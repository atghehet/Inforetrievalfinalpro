"""Validate a teammate's TREC run file against the shared data, and score it.

    python scripts/check_run.py runs/bm25_abstract.val.txt --split val

Checks: 6 columns (qid Q0 docid rank score run_name), qids exist in the split's
qrels, docids exist in papers.parquet, no duplicate docids per query, ranks
start at 1 and follow descending score. Then prints MRR@10, Recall@10/100,
nDCG@10 (one relevant doc per query) so you can sanity-check against Member 4's
harness. Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from citeds import io  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    a = ap.parse_args()

    rel = {}
    for line in open(io.qrels_path(a.split)):
        qid, _, did, _ = line.split()
        rel[qid] = did
    docids = set(io.load_papers(["paper_id"])["paper_id"])

    errors, runs = [], defaultdict(list)
    for n, line in enumerate(open(a.run), 1):
        parts = line.split()
        if len(parts) != 6:
            errors.append(f"line {n}: expected 6 columns, got {len(parts)}")
            continue
        qid, q0, did, rank, score, _ = parts
        if qid not in rel:
            errors.append(f"line {n}: qid {qid} not in {a.split} qrels")
        if did not in docids:
            errors.append(f"line {n}: docid {did} not in papers.parquet")
        try:
            runs[qid].append((int(rank), float(score), did))
        except ValueError:
            errors.append(f"line {n}: rank/score not numeric")
    for qid, rows in runs.items():
        rows.sort()
        if rows and rows[0][0] != 1:
            errors.append(f"{qid}: ranks start at {rows[0][0]}, not 1")
        if len({d for _, _, d in rows}) != len(rows):
            errors.append(f"{qid}: duplicate docids")
        if any(rows[i][1] < rows[i + 1][1] for i in range(len(rows) - 1)):
            errors.append(f"{qid}: scores not descending with rank")

    missing = set(rel) - set(runs)
    print(f"run: {a.run}  split: {a.split}")
    print(f"queries in qrels: {len(rel)}  in run: {len(runs)}  missing from run: {len(missing)}")
    mrr = r10 = r100 = ndcg = 0.0
    for qid, did in rel.items():
        ranked = [d for _, _, d in sorted(runs.get(qid, []))]
        if did in ranked:
            r = ranked.index(did) + 1
            mrr += 1 / r if r <= 10 else 0
            r10 += r <= 10
            r100 += r <= 100
            ndcg += 1 / math.log2(r + 1) if r <= 10 else 0
    k = max(len(rel), 1)
    print(f"MRR@10 {mrr / k:.4f}  Recall@10 {r10 / k:.4f}  Recall@100 {r100 / k:.4f}  nDCG@10 {ndcg / k:.4f}")
    if errors:
        print(f"\n{len(errors)} FORMAT ERRORS, first 10:")
        for e in errors[:10]:
            print("  " + e)
        sys.exit(1)
    print("format OK")


if __name__ == "__main__":
    main()
