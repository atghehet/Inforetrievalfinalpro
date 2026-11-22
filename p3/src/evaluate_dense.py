"""Evaluate every dense run on the VALIDATION split (for P3's own model selection).

    python p3/src/evaluate_dense.py              # -> runs/p3/dense/dense_validation_metrics.csv

Test-split numbers are produced once, by evaluate_hybrid_test.py, after all
choices are frozen. P4's harness computes the official metrics.
"""
from __future__ import annotations

import pandas as pd

from common import DENSE_DIR, evaluate, load_qrels, load_run, qrels_path


def main() -> None:
    rel = load_qrels(qrels_path("val"))
    rows = []
    for path in sorted(DENSE_DIR.glob("*.val.run")):
        system = path.name.removesuffix(".val.run")
        rows.append({"system": system, **evaluate(load_run(path), rel)})
    if not rows:
        raise SystemExit(f"No *.val.run files in {DENSE_DIR}; run p3/src/dense_baseline.py first")
    df = pd.DataFrame(rows).sort_values("MRR@10", ascending=False)
    out = DENSE_DIR / "dense_validation_metrics.csv"
    df.round(4).to_csv(out, index=False)
    print("Dense retrieval, validation split")
    print(df.round(4).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
