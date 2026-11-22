"""Export P1's Parquet tables in the JSONL format P2's lexical runner reads.

    python -m citeds.export_p2            # -> data/p2_input/

Replaces P3's earlier ad-hoc converter. Differences:
  * `label` is the predicted citation function (background/method/result) from
    context_functions.parquet, not the section heading, so P2's --method-filter
    works directly (the section is kept in `section`).
  * `before` / `after` hold up to TWO neighbour sentences each, in reading
    order, so P2's window-0/1/2 comparison is meaningful.
  * Only leakage-controlled index contexts are exported.
  * test.qrels is written for P4; P2's runner never opens it.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from . import config, io

P2_INPUT_DIR = config.DATA_DIR / "p2_input"
SPLIT_NAMES = {"val": "validation", "test": "test"}


def _write_jsonl(path: Path, rows) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def export(out_dir: Path = P2_INPUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    papers = io.load_papers(["paper_id", "title", "abstract"])
    ctx = io.load_index_contexts()
    labels = io.load_context_functions().set_index("context_id")["function"]
    missing = ~ctx["context_id"].isin(labels.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} index contexts have no citation-function label; "
                         f"run `python -m citeds.citation_function predict` first")

    counts = {"papers": _write_jsonl(out_dir / "papers.jsonl", (
        {"paper_id": r.paper_id, "title": r.title, "abstract": r.abstract}
        for r in papers.itertuples(index=False)))}

    def ctx_rows():
        for r in ctx.itertuples(index=False):
            before = [t for t in (getattr(r, "prev2_text", ""), r.prev_text) if t]
            after = [t for t in (r.next_text, getattr(r, "next2_text", "")) if t]
            yield {"context_id": r.context_id, "cited_paper_id": r.cited_paper_id,
                   "sentence": r.text, "before": before, "after": after,
                   "label": labels[r.context_id], "section": r.section}
    counts["contexts"] = _write_jsonl(out_dir / "contexts.jsonl", ctx_rows())

    queries = io.load_queries()
    for p1_split, p2_split in SPLIT_NAMES.items():
        q = queries[queries["split"] == p1_split]
        counts[f"{p2_split}_queries"] = _write_jsonl(
            out_dir / f"{p2_split}.jsonl",
            ({"query_id": r.query_id, "text": r.text} for r in q.itertuples(index=False)))
        shutil.copyfile(io.qrels_path(p1_split), out_dir / f"{p2_split}.qrels")
    info = config.PROCESSED_DIR / "build_info.json"
    (out_dir / "SOURCE.txt").write_text(f"Exported from {config.PROCESSED_DIR} by citeds.export_p2\n"
                                        + (info.read_text() if info.exists() else ""))
    return counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=P2_INPUT_DIR)
    a = ap.parse_args(argv)
    counts = export(a.out)
    print(f"Wrote P2 input to {a.out}: {counts}")


if __name__ == "__main__":
    main()
