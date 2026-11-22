"""Streaming reader for the unarXive 2022 JSONL format.

One JSON object per line, one paper per object. Fields used:

  paper_id                       arXiv ID of the paper
  metadata.title / .abstract     arXiv metadata
  metadata.categories            "cs.CL cs.LG"
  metadata.versions[0].created   date of v1 (for the year)
  abstract.text                  abstract (preferred over metadata.abstract)
  body_text[]                    paragraphs: {section, text, cite_spans[]}
                                 text contains {{cite:<ref_id>}} markers
  bib_entries{ref_id}            {ids: {arxiv_id, ...}, contained_arXiv_ids: [...]}

Files may be plain, .gz, .bz2 or .xz compressed.
See https://github.com/IllDepence/unarXive for the full format.
"""
from __future__ import annotations

import bz2
import gzip
import json
import lzma
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .clean import find_cite_ids, normalise_arxiv_id, split_sentences

_OPENERS = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_RAW_ARXIV = re.compile(r"arxiv(?:\.org/(?:abs|pdf)/|[^0-9a-z]{0,25}(?:arxiv:)?\s*)(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[a-z]{2})?/\d{7})",
                        re.IGNORECASE)


@dataclass
class RawSentence:
    sentence_number: int
    text: str                    # raw text, markers intact
    section: str
    cited_arxiv_ids: list[str]   # normalised, de-duplicated, in order


@dataclass
class RawPaper:
    paper_id: str
    title: str
    abstract: str
    year: int | None
    categories: str
    # Every sentence of every paragraph is kept (not only citing ones) so that
    # prev/next context sentences are available.
    paragraphs: list[list[RawSentence]] = field(default_factory=list)


def iter_files(raw_dir: Path) -> list[Path]:
    pats = ("*.jsonl", "*.jsonl.gz", "*.jsonl.bz2", "*.jsonl.xz")
    files = sorted({p for pat in pats for p in Path(raw_dir).rglob(pat)})
    if not files:
        raise FileNotFoundError(f"no unarXive *.jsonl[.gz|.bz2|.xz] files under {raw_dir}")
    return files


def _open(path: Path):
    opener = _OPENERS.get(path.suffix, open)
    return opener(path, "rt", encoding="utf-8")


def _year(meta: dict, paper_id: str) -> int | None:
    versions = meta.get("versions") or []
    if versions:
        m = _YEAR.search(str(versions[0].get("created", "")))
        if m:
            return int(m.group(0))
    m = re.match(r"(\d{2})(\d{2})\.", paper_id)  # new-style IDs: YYMM.NNNNN
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r"/(\d{2})\d{5}$", paper_id)    # old-style: archive/YYMMNNN
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy > 90 else 2000 + yy
    return None


def _bib_to_arxiv(bib_entries: dict) -> dict[str, str]:
    out = {}
    for ref_id, entry in (bib_entries or {}).items():
        aid = normalise_arxiv_id((entry.get("ids") or {}).get("arxiv_id"))
        if not aid:
            for c in entry.get("contained_arXiv_ids") or []:
                aid = normalise_arxiv_id(c.get("id") if isinstance(c, dict) else c)
                if aid:
                    break
        if not aid:
            # Last resort: "arXiv:1810.04805", "arXiv preprint arXiv:1810.04805",
            # "arxiv.org/abs/1810.04805" in the raw reference string.
            m = _RAW_ARXIV.search(entry.get("bib_entry_raw") or "")
            if m:
                aid = normalise_arxiv_id(m.group(1))
        if aid:
            out[ref_id] = aid
    return out


def parse_record(rec: dict) -> RawPaper | None:
    meta = rec.get("metadata") or {}
    pid = normalise_arxiv_id(rec.get("paper_id") or meta.get("id"))
    if not pid:
        return None
    abstract = (rec.get("abstract") or {}).get("text") or meta.get("abstract") or ""
    paper = RawPaper(
        paper_id=pid,
        title=(meta.get("title") or "").replace("\n", " ").strip(),
        abstract=abstract,
        year=_year(meta, pid),
        categories=(meta.get("categories") or "").strip(),
    )
    ref2arxiv = _bib_to_arxiv(rec.get("bib_entries") or {})
    n = 0
    for para in rec.get("body_text") or []:
        if para.get("content_type", "paragraph") != "paragraph":
            continue
        section = (para.get("section") or "").strip().lower()
        sents = []
        for s in split_sentences(para.get("text") or ""):
            cited = []
            for ref in find_cite_ids(s):
                aid = ref2arxiv.get(ref)
                if aid and aid not in cited:
                    cited.append(aid)
            sents.append(RawSentence(n, s, section, cited))
            n += 1
        if sents:
            paper.paragraphs.append(sents)
    return paper


def iter_papers(raw_dir: Path, field_filter: str | None = None) -> Iterator[RawPaper]:
    """Yield parsed papers, optionally only those listing `field_filter` as a category."""
    for path in iter_files(raw_dir):
        with _open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if field_filter:
                    cats = ((rec.get("metadata") or {}).get("categories") or "").split()
                    if field_filter not in cats:
                        continue
                p = parse_record(rec)
                if p is not None:
                    yield p
