"""OPTIONAL: polite OpenAlex crawler for the freshness / cold-start experiment.

Fetches recent arXiv cs.CL-style papers (with abstracts) that are newer than
the unarXive snapshot, so they have few or no citations.

    export CITEDS_CONTACT_EMAIL=you@university.edu     # REQUIRED, see config.py
    python -m citeds.crawler --since 2024-01-01 --max 2000

Politeness:
  * Contact email in the User-Agent and `mailto` parameter (OpenAlex polite pool).
    The crawler REFUSES to run while config.CONTACT_EMAIL is the placeholder.
  * robots.txt of the API host is checked before the first request.
  * <= 1 request per config.CRAWL_MIN_INTERVAL_S seconds; backs off on 429/5xx.
  * Every response is cached on disk (data/external/crawl_cache/) keyed by
    URL; cached pages are never re-fetched.

Output: data/processed/fresh_papers.parquet (schema.FRESH_PAPERS).
NOTE: written against the OpenAlex API docs; not run against the live API in
development (the dev environment had no network access to it). Test with a
small --max first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlencode, urlparse

import pandas as pd
import pyarrow.parquet as pq
import requests

from . import config, schema
from .clean import clean_text, normalise_arxiv_id

log = logging.getLogger(__name__)
API = "https://api.openalex.org/works"
# OpenAlex concept "Natural language processing". Concepts are being replaced by
# topics; if this filter stops working, switch to a topics.id / primary_topic filter.
DEFAULT_FILTER = "concepts.id:C204321447,has_abstract:true,locations.source.id:S4306400194"
PLACEHOLDER = "YOUR_EMAIL@example.com"


def _check_email(email: str) -> None:
    if not email or email == PLACEHOLDER or "@" not in email:
        raise SystemExit(
            "\n*** CONTACT EMAIL NOT SET ***\n"
            "OpenAlex asks crawlers to identify themselves. Set it with\n"
            "    export CITEDS_CONTACT_EMAIL=you@university.edu\n"
            "or edit CONTACT_EMAIL in citeds/config.py, then re-run.\n")


class PoliteClient:
    def __init__(self, email: str, cache_dir: Path = config.CRAWL_CACHE_DIR,
                 min_interval: float = config.CRAWL_MIN_INTERVAL_S):
        _check_email(email)
        self.email = email
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = f"citeds-course-project/0.1 (mailto:{email})"
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    def _allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.session.get(f"https://{host}/robots.txt", timeout=20)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                rp.parse([])  # unreachable robots.txt: treat as allow-all (RFC 9309)
            self._robots[host] = rp
        return self._robots[host].can_fetch(self.session.headers["User-Agent"], url)

    def get_json(self, url: str) -> dict:
        key = hashlib.sha256(url.encode()).hexdigest()
        cache = self.cache_dir / f"{key}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        if not self._allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = self.session.get(url, timeout=60)
            if r.status_code == 200:
                data = r.json()
                cache.write_text(json.dumps(data))
                return data
            if r.status_code in (429, 500, 502, 503, 504):
                backoff = float(r.headers.get("Retry-After", 2 ** attempt))
                log.warning("HTTP %s, retrying in %.0fs", r.status_code, backoff)
                time.sleep(backoff)
                continue
            r.raise_for_status()
        raise RuntimeError(f"giving up on {url}")


def abstract_from_inverted_index(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def arxiv_id_of(work: dict) -> str | None:
    for loc in work.get("locations") or []:
        for k in ("landing_page_url", "pdf_url"):
            aid = normalise_arxiv_id(loc.get(k))
            if aid and "arxiv.org" in (loc.get(k) or ""):
                return aid
    return normalise_arxiv_id((work.get("ids") or {}).get("arxiv"))


def work_to_row(w: dict) -> dict | None:
    aid = arxiv_id_of(w)
    abstract = clean_text(abstract_from_inverted_index(w.get("abstract_inverted_index")))
    if not aid or len(abstract.split()) < config.MIN_ABSTRACT_TOKENS:
        return None
    return {"paper_id": aid, "title": clean_text(w.get("title") or ""), "abstract": abstract,
            "year": w.get("publication_year"), "categories": "",
            "citation_count": 0, "num_contexts": 0,  # in-corpus counts: none by construction
            "openalex_id": w.get("id", ""), "openalex_cited_by_count": int(w.get("cited_by_count") or 0),
            "referenced_works": "|".join(w.get("referenced_works") or []),
            "publication_date": w.get("publication_date") or ""}


def crawl(email: str, since: str, max_papers: int, flt: str = DEFAULT_FILTER,
          exclude_ids: set[str] | None = None) -> pd.DataFrame:
    client = PoliteClient(email)
    rows, cursor, seen = [], "*", set(exclude_ids or ())
    while cursor and len(rows) < max_papers:
        params = {"filter": f"{flt},from_publication_date:{since}", "per-page": 200,
                  "cursor": cursor, "mailto": email, "sort": "publication_date:desc",
                  "select": "id,ids,title,publication_year,publication_date,abstract_inverted_index,"
                            "locations,cited_by_count,referenced_works"}
        data = client.get_json(f"{API}?{urlencode(params)}")
        for w in data.get("results", []):
            row = work_to_row(w)
            if row and row["paper_id"] not in seen:
                seen.add(row["paper_id"])
                rows.append(row)
        cursor = (data.get("meta") or {}).get("next_cursor")
        log.info("fetched %d fresh papers", len(rows))
        if not data.get("results"):
            break
    df = pd.DataFrame(rows[:max_papers], columns=schema.FRESH_PAPERS.names)
    df["year"] = df["year"].astype("Int32")
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2024-01-01", help="only papers published on/after this date")
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--filter", default=DEFAULT_FILTER)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _check_email(config.CONTACT_EMAIL)
    config.ensure_dirs()
    exclude = set(pd.read_parquet(config.PAPERS_PATH, columns=["paper_id"])["paper_id"]) \
        if config.PAPERS_PATH.exists() else set()
    df = crawl(config.CONTACT_EMAIL, a.since, a.max, a.filter, exclude)
    pq.write_table(schema.conform(df, schema.FRESH_PAPERS), config.FRESH_PAPERS_PATH)
    log.info("wrote %s (%d rows)", config.FRESH_PAPERS_PATH, len(df))


if __name__ == "__main__":
    main()
