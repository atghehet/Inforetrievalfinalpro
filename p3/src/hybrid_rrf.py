"""Hybrid retrieval: Reciprocal Rank Fusion of P2's BM25 run and a P3 dense run.

    python p3/src/hybrid_rrf.py

Everything is chosen on VALIDATION, then frozen and applied to test:
  1. the dense run (best validation MRR@10 among runs/p3/dense/*.val.run,
     unless --dense names one),
  2. the RRF constant k from --k-grid.
The BM25 side is P2's frozen fusion run (runs/p2/fusion/bm25.{validation,test}.run),
which P2 already selected on validation.

Writes runs/p3/hybrid/hybrid_rrf.{val,test}.run, the validation grid
(hybrid_validation_metrics.csv) and selection.json, which
evaluate_hybrid_test.py reads instead of a hard-coded constant.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common import (DENSE_DIR, HYBRID_DIR, P2_RUNS_DIR, P2_SPLIT, evaluate, load_qrels,
                    load_run, qrels_path, rrf, write_run)


def bm25_run_path(split: str):
    return P2_RUNS_DIR / "fusion" / f"bm25.{P2_SPLIT[split]}.run"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", help="dense system name, e.g. minilm_expanded (default: best on validation)")
    ap.add_argument("--k-grid", nargs="+", type=int, default=[10, 20, 40, 60, 100])
    a = ap.parse_args(argv)

    for split in ("val", "test"):
        if not bm25_run_path(split).exists():
            raise SystemExit(f"Missing {bm25_run_path(split)}: run P2's experiment with --out runs/p2 first")
    rel_val = load_qrels(qrels_path("val"))
    bm25_val = load_run(bm25_run_path("val"))

    dense_scores = {p.name.removesuffix(".val.run"): evaluate(load_run(p), rel_val)["MRR@10"]
                    for p in sorted(DENSE_DIR.glob("*.val.run"))}
    if not dense_scores:
        raise SystemExit("No dense validation runs; run p3/src/dense_baseline.py first")
    dense_name = a.dense or max(dense_scores, key=dense_scores.get)
    dense_val = load_run(DENSE_DIR / f"{dense_name}.val.run")

    rows = []
    for k in a.k_grid:
        rows.append({"k": k, **evaluate({q: {d: r for r, (d, _) in enumerate(v, 1)}
                                         for q, v in rrf([dense_val, bm25_val], k).items()}, rel_val)})
    grid = pd.DataFrame(rows)
    best_k = int(grid.sort_values(["MRR@10", "k"], ascending=[False, True]).iloc[0]["k"])

    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    grid.round(4).to_csv(HYBRID_DIR / "hybrid_validation_metrics.csv", index=False)
    bm25_sel = (P2_RUNS_DIR / "fusion" / "selection.txt")
    selection = {"dense_system": dense_name, "rrf_k": best_k,
                 "bm25_system": bm25_sel.read_text().strip() if bm25_sel.exists() else "bm25 (P2 fusion run)",
                 "selected_on": "validation MRR@10",
                 "dense_validation_mrr10": {k: round(v, 4) for k, v in dense_scores.items()}}
    (HYBRID_DIR / "selection.json").write_text(json.dumps(selection, indent=2))

    tag = f"hybrid_rrf_k{best_k}"
    for split in ("val", "test"):
        dense = load_run(DENSE_DIR / f"{dense_name}.{split}.run")
        bm25 = load_run(bm25_run_path(split))
        write_run(HYBRID_DIR / f"hybrid_rrf.{split}.run", rrf([dense, bm25], best_k), tag)

    print(f"Dense system (by validation MRR@10): {dense_name}")
    print("RRF validation grid:")
    print(grid.round(4).to_string(index=False))
    print(f"\nSelected k = {best_k}; frozen in {HYBRID_DIR / 'selection.json'}")
    print("Wrote hybrid_rrf.val.run and hybrid_rrf.test.run (test not scored here).")


if __name__ == "__main__":
    main()
