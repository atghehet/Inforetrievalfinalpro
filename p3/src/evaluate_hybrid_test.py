"""Score the frozen dense, BM25 and hybrid systems ONCE on the held-out test split.

    python p3/src/evaluate_hybrid_test.py

Reads the choices frozen on validation from runs/p3/hybrid/selection.json
(written by hybrid_rrf.py); nothing is tuned here. P4's harness produces the
official numbers and significance tests; this is P3's own check.
"""
from __future__ import annotations

import json

import pandas as pd

from common import DENSE_DIR, HYBRID_DIR, P2_RUNS_DIR, evaluate, load_qrels, load_run, qrels_path


def main() -> None:
    sel_path = HYBRID_DIR / "selection.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} missing: run p3/src/hybrid_rrf.py first")
    sel = json.loads(sel_path.read_text())
    rel = load_qrels(qrels_path("test"))
    systems = {
        f"Dense ({sel['dense_system']})": DENSE_DIR / f"{sel['dense_system']}.test.run",
        "BM25 (P2 fusion run)": P2_RUNS_DIR / "fusion" / "bm25.test.run",
        f"Hybrid RRF k={sel['rrf_k']}": HYBRID_DIR / "hybrid_rrf.test.run",
    }
    rows = [{"system": name, **evaluate(load_run(path), rel)} for name, path in systems.items()]
    df = pd.DataFrame(rows).round(4)
    df.to_csv(HYBRID_DIR / "hybrid_test_metrics.csv", index=False)
    print("Held-out test evaluation (choices frozen on validation)")
    print(df.to_string(index=False))
    print(f"\nSaved {HYBRID_DIR / 'hybrid_test_metrics.csv'}")


if __name__ == "__main__":
    main()
