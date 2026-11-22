"""Run the whole project pipeline end to end, in order (P1 -> P2 -> P3 -> P4 inputs).

    python scripts/run_all.py                    # use existing data/processed, P2 quick grid
    python scripts/run_all.py --p2 full          # P2's full tuning grid (hours)
    python scripts/run_all.py --build            # also rebuild P1 data from data/raw first
    python scripts/run_all.py --p2 skip          # reuse existing runs/p2

Steps
  1. P1  (optional --build) build tables/splits/leakage control, label citation functions
  2. P1  export JSONL for P2 (data/p2_input) and verify it
  3. P2  lexical experiment -> runs/p2  (validation runs, frozen test runs, fusion runs)
  4. P3  dense retrieval for each --models -> runs/p3/dense
  5. P3  validation metrics, hybrid RRF (choices frozen on validation), test check
  6.     leakage check, format check of every frozen run, runs/MANIFEST.csv for P4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def step(title: str, cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\n=== {title}\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {title}")
    print(f"--- done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="rebuild P1 data from data/raw (needs the raw files)")
    ap.add_argument("--p2", choices=["quick", "full", "skip"], default="quick")
    ap.add_argument("--models", nargs="+", default=["sentence-transformers/all-MiniLM-L6-v2"],
                    help="dense models; add sentence-transformers/allenai-specter for SPECTER (slower)")
    a = ap.parse_args()

    if a.build:
        step("P1 build", [PY, "-m", "citeds.build"])
        step("P1 citation functions", [PY, "-m", "citeds.citation_function", "predict"])
    step("P1 leakage check", [PY, "-c", "from citeds import io; from citeds.leakage import assert_no_leakage; "
                              "assert_no_leakage(io.load_index_contexts(), io.load_queries()); print('leakage OK')"])
    step("P1 export for P2", [PY, "-m", "citeds.export_p2"])
    step("Verify P2 input", [PY, "p3/src/verify_p2_data.py"])

    if a.p2 != "skip":
        out = ROOT / "runs" / "p2"
        if out.exists():
            shutil.rmtree(out)  # P2 requires a fresh output directory
        cmd = [PY, "-m", "p2.experiment", "--data", str(ROOT / "data" / "p2_input"), "--out", str(out),
               "--method-filter"] + (["--quick"] if a.p2 == "quick" else [])
        step(f"P2 lexical experiment ({a.p2})", cmd, cwd=ROOT / "lexical")

    for model in a.models:
        step(f"P3 dense ({model})", [PY, "p3/src/dense_baseline.py", "--model", model])
    step("P3 dense validation metrics", [PY, "p3/src/evaluate_dense.py"])
    step("P3 hybrid RRF (selected on validation)", [PY, "p3/src/hybrid_rrf.py"])
    step("P3 test check (frozen choices)", [PY, "p3/src/evaluate_hybrid_test.py"])

    frozen = sorted((ROOT / "runs" / "p2" / "test").glob("*.run")) + \
        sorted((ROOT / "runs" / "p2" / "fusion").glob("*.run")) + \
        sorted((ROOT / "runs" / "p3").rglob("*.run"))
    for run in frozen:
        split = "test" if (".test." in run.name or run.parent.name == "test") else "val"
        r = subprocess.run([PY, "scripts/check_run.py", str(run), "--split", split],
                           cwd=ROOT, capture_output=True, text=True)
        status = "OK " if r.returncode == 0 else "BAD"
        print(f"{status} {run.relative_to(ROOT)}")
        if r.returncode != 0:
            print(r.stdout[-800:])
            sys.exit("Format check failed")
    step("Manifest for P4", [PY, "scripts/make_manifest.py"])
    print("\nAll steps finished. P4: start from runs/MANIFEST.csv and docs/P4_HANDOFF.md")


if __name__ == "__main__":
    main()
