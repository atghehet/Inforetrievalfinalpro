"""Audit how many bibliography references resolve to arXiv IDs under different policies (a READ-ONLY study).

    python -m p4.dataprep.resolution_audit --subset <folder of JSONL files> --out <report.json>

Policies compared (each adds a source of arXiv IDs to the previous one):
    ids        bib_entries[].ids.arxiv_id
    current    ids + bib_entries[].contained_arXiv_ids          <- what citeds/unarxive.py does today
    proposed   current + an arXiv ID found in the raw reference text (`bib_entry_raw`)

For every policy the audit reports resolved references, in-corpus citation edges (cited paper is in the folder) and
usable citation contexts (using P1's own cleaning functions, imported read-only). It also checks the raw-text source:
  * agreement with ids.arxiv_id where both exist,
  * a title-consistency test: when a raw-text ID lands on a corpus paper, does that paper's title appear in the
    reference text? The same test on the other sources is the control,
  * ambiguity (several different IDs in one reference) and implausible IDs (cited paper newer than the citing paper).

`extract_raw_ids` is the proposed fallback as a pure function, so it can be reviewed and tested on its own. Nothing here
modifies P1; the tool only reads the subset folder and writes the report file you name.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from citeds import config
from citeds.clean import clean_text, find_cite_ids, normalise_arxiv_id, normalise_for_match, num_tokens, split_sentences

_NEW = r"\d{4}\.\d{4,5}"
_OLD = r"[a-z][a-z\-]*(?:\.[A-Za-z]{2})?/\d{7}"
# 1) "arXiv:2104.06921", "arXiv preprint arXiv:2104.06921v2", "arxiv.org/abs/2104.06921", "arxiv.org/pdf/hep-th/9901001"
# 2) DOI form "10.48550/arXiv.2104.06921"
_RAW = re.compile(rf"(?i)(?:arxiv(?:\.org)?(?:\s*preprint)?(?:\s*arxiv)?\s*[:/.]?\s*(?:abs/|pdf/)?|10\.48550/arxiv\.)({_NEW}|{_OLD})(?:v\d+)?")
_WORD = re.compile(r"[a-z0-9]+")


def valid_arxiv_id(identifier: str) -> bool:
    """Shape check: plausible month, and 4 digits before 2015 / 5 digits from 2015 for new-style IDs."""
    m = re.fullmatch(r"(\d{2})(\d{2})\.(\d{4,5})", identifier)
    if m:
        yy, mm, seq = int(m.group(1)), int(m.group(2)), m.group(3)
        yymm = yy * 100 + mm
        return 1 <= mm <= 12 and yymm >= 704 and (len(seq) == 4) == (yymm < 1501)
    m = re.fullmatch(r"[a-z][a-z\-]*/(\d{2})(\d{2})\d{3}", identifier)
    return bool(m) and 1 <= int(m.group(2)) <= 12


def extract_raw_ids(text: str) -> list[str]:
    """Distinct, valid, normalised arXiv IDs mentioned in a raw reference string, in order of appearance."""
    found: list[str] = []
    for match in _RAW.finditer(text or ""):
        normalised = normalise_arxiv_id(match.group(1))
        if normalised and valid_arxiv_id(normalised) and normalised not in found:
            found.append(normalised)
    return found


def _yymm(identifier: str) -> int | None:
    m = re.match(r"(\d{2})(\d{2})\.", identifier)
    if m:
        return int(m.group(1)) * 100 + int(m.group(2))
    return 0 if "/" in identifier else None                      # old-style IDs all predate 0704


def _id_year(identifier: str) -> str:
    """Year bucket of an arXiv ID: 2007-2023 from the YYMM prefix, or 'pre-2007' for old-style IDs."""
    m = re.match(r"(\d{2})\d{2}\.", identifier)
    return str(2000 + int(m.group(1))) if m else "pre-2007"


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def title_consistent(reference_text: str, title: str, threshold: float = 0.8) -> bool:
    """True if at least `threshold` of the title's words occur in the reference text."""
    title_words = _tokens(title)
    return bool(title_words) and len(title_words & _tokens(reference_text)) / len(title_words) >= threshold


def _iter_records(folder: Path):
    for path in sorted(folder.rglob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def _first_contained(entry: dict) -> str | None:
    for c in entry.get("contained_arXiv_ids") or []:
        found = normalise_arxiv_id(c.get("id") if isinstance(c, dict) else c)
        if found:
            return found
    return None


def collect(folder: Path, category: str = "cs.CL") -> dict:
    """One pass over the folder: per citing paper its references (all sources) and its usable citing sentences."""
    papers, titles = [], {}
    for rec in _iter_records(folder):
        if category not in ((rec.get("metadata") or {}).get("categories") or "").split():
            continue
        pid = normalise_arxiv_id(rec.get("paper_id") or (rec.get("metadata") or {}).get("id"))
        if not pid:
            continue
        titles[pid] = (rec.get("metadata") or {}).get("title") or ""
        refs = {}
        for key, entry in (rec.get("bib_entries") or {}).items():
            raw = entry.get("bib_entry_raw") or ""
            refs[key] = {"ids": normalise_arxiv_id((entry.get("ids") or {}).get("arxiv_id")),
                         "contained": _first_contained(entry), "raw_ids": extract_raw_ids(raw), "raw": raw}
        sentences = []
        for para in rec.get("body_text") or []:
            if para.get("content_type", "paragraph") != "paragraph":
                continue
            for s in split_sentences(para.get("text") or ""):
                keys = list(dict.fromkeys(find_cite_ids(s)))
                if not keys:
                    continue
                clean = clean_text(s)
                if config.MIN_SENTENCE_TOKENS <= num_tokens(clean) <= config.MAX_SENTENCE_TOKENS:
                    sentences.append((normalise_for_match(clean), keys))
        papers.append({"pid": pid, "refs": refs, "sentences": sentences})
    return {"papers": papers, "titles": titles}


def resolve(ref: dict, policy: str) -> str | None:
    if ref["ids"]:
        return ref["ids"]
    if policy == "ids":
        return None
    if ref["contained"]:
        return ref["contained"]
    if policy == "proposed" and len(ref["raw_ids"]) == 1:       # ambiguous references (several IDs) are not resolved
        return ref["raw_ids"][0]
    return None


def audit(folder, category: str = "cs.CL") -> dict:
    started = time.time()
    data = collect(Path(folder), category)
    papers, titles = data["papers"], data["titles"]
    corpus = set(titles)
    n_refs = sum(len(p["refs"]) for p in papers)
    report = {"subset": str(folder), "category": category, "papers": len(papers), "references": n_refs, "policies": {}}
    for policy in ("ids", "current", "proposed"):
        resolved = edges = 0
        contexts = set()
        edge_set = set()
        cited_ids = set()                                            # every distinct resolved cited ID (inside or outside the corpus)
        for p in papers:
            mapping = {k: resolve(r, policy) for k, r in p["refs"].items()}
            resolved += sum(v is not None for v in mapping.values())
            cited_ids |= {v for v in mapping.values() if v and v != p["pid"]}
            for key, cited in mapping.items():
                if cited and cited in corpus and cited != p["pid"]:
                    edge_set.add((p["pid"], cited))
            for norm_text, keys in p["sentences"]:                   # P1 rule: one context per (citing, cited, sentence text)
                for key in keys:
                    cited = mapping.get(key)
                    if cited and cited in corpus and cited != p["pid"]:
                        contexts.add((p["pid"], cited, norm_text))
        report["policies"][policy] = {"resolved_references": resolved, "resolved_share": resolved / n_refs if n_refs else 0,
                                      "in_corpus_edges": len(edge_set), "usable_contexts": len(contexts),
                                      "distinct_cited_ids": len(cited_ids),
                                      "distinct_cited_in_corpus": len(cited_ids & corpus),
                                      "distinct_cited_outside_corpus": len(cited_ids - corpus),
                                      "outside_corpus_by_id_year": dict(sorted(Counter(
                                          _id_year(i) for i in cited_ids - corpus).items(), key=lambda kv: str(kv[0])))}
    # --- quality of the raw-text source
    agree = disagree = raw_only = ambiguous = implausible = 0
    hits = {"raw_only": Counter(), "control": Counter()}
    examples = []
    for p in papers:
        own = _yymm(p["pid"])
        for r in p["refs"].values():
            if len(r["raw_ids"]) > 1:
                ambiguous += 1
            known = r["ids"] or r["contained"]
            if r["raw_ids"] and known:                               # the raw text and another source both name an ID
                if known in r["raw_ids"]:
                    agree += 1
                else:
                    disagree += 1
            if len(r["raw_ids"]) == 1 and not known:
                raw_only += 1
                cited = r["raw_ids"][0]
                a, b = _yymm(cited), own
                implausible += bool(a is not None and b is not None and a > b)
                if cited in corpus and cited != p["pid"]:
                    ok = title_consistent(r["raw"], titles[cited])
                    hits["raw_only"][ok] += 1
                    if not ok and len(examples) < 12:
                        examples.append({"citing": p["pid"], "resolved_to": cited, "title": titles[cited][:120], "reference": r["raw"][:240]})
            if known and known in corpus and known != p["pid"]:
                hits["control"][title_consistent(r["raw"], titles[known])] += 1
    rate = lambda c: (c[True] / (c[True] + c[False])) if (c[True] + c[False]) else None
    report["raw_text_quality"] = {
        "references_resolved_only_by_raw_text": raw_only, "ambiguous_references_several_ids": ambiguous,
        "agree_with_other_sources": agree, "disagree_with_other_sources": disagree,
        "implausible_cited_newer_than_citing": implausible,
        "in_corpus_hits_raw_only": sum(hits["raw_only"].values()), "title_consistent_share_raw_only": rate(hits["raw_only"]),
        "in_corpus_hits_control": sum(hits["control"].values()), "title_consistent_share_control": rate(hits["control"]),
        "title_inconsistent_raw_examples": examples}
    report["seconds"] = round(time.time() - started, 1)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="JSON report to write (must not exist)")
    p.add_argument("--category", default="cs.CL")
    args = p.parse_args(argv)
    if args.out.exists():
        print(f"error: {args.out} already exists", file=sys.stderr)
        return 2
    if not args.subset.is_dir():
        print(f"error: subset folder not found: {args.subset}", file=sys.stderr)
        return 2
    report = audit(args.subset, args.category)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
