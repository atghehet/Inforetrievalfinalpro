"""Cross-check P4's harness against the team's scripts/check_run.py (P4_HANDOFF.md asks that both give the same numbers on the same file).

    python -m p4.experiments.crosscheck_check_run --config runs.json --processed data/processed --check-run <check_run.py> \
        --integration-src <folder holding citeds/> --split val --out crosscheck.csv

check_run.py is run unchanged as a subprocess (it needs the team's `citeds` package, whose folder is given with --integration-src, and reads
the qrels from CITEDS_DATA = the folder that contains `processed/`). It prints MRR@10, Recall@10, Recall@100 and nDCG@10 to four decimals;
P4's harness must agree to that precision. Any difference above 5e-5 is reported and the exit code is 2.
Use --split val for tuning-time checks; the test split is only scored by the final evaluation.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from p4.eval.harness import evaluate_runs
from p4.eval.trec_io import read_qrels, read_run

TOLERANCE = 5e-5                                                  # check_run prints four decimals
PATTERN = re.compile(r"MRR@10 ([\d.]+)\s+Recall@10 ([\d.]+)\s+Recall@100 ([\d.]+)\s+nDCG@10 ([\d.]+)")
COLUMNS = ("MRR@10", "Recall@10", "Recall@100", "nDCG@10")


def run_check_run(check_run: Path, run: Path, split: str, integration_src: Path, data_dir: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": str(integration_src), "CITEDS_DATA": str(data_dir), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-B", str(check_run), str(run), "--split", split], capture_output=True, text=True, env=env)
    match = PATTERN.search(r.stdout)
    if r.returncode != 0 or not match or "format OK" not in r.stdout:
        raise RuntimeError(f"check_run.py failed on {run.name} (exit {r.returncode}): {(r.stdout + r.stderr)[-400:]}")
    return dict(zip(COLUMNS, map(float, match.groups())))


def crosscheck(systems: dict[str, Path], qrels_path: Path, split: str, check_run: Path, integration_src: Path, data_dir: Path) -> pd.DataFrame:
    qrels = read_qrels(qrels_path)
    rows = []
    for name, path in systems.items():
        theirs = run_check_run(check_run, path, split, integration_src, data_dir)
        ours = evaluate_runs({name: read_run(path)}, qrels, (10, 100)).summary.iloc[0]
        row = {"system": name, "split": split}
        for col in COLUMNS:
            row[f"{col} (check_run)"], row[f"{col} (P4)"] = theirs[col], float(ours[col])
            row[f"{col} diff"] = abs(float(ours[col]) - theirs[col])
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="runs.json (systems with val/test run files)")
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--check-run", required=True, type=Path)
    ap.add_argument("--integration-src", required=True, type=Path, help="folder that contains the team's citeds/ package")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--out", required=True, type=Path, help="CSV to write (must not exist)")
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"error: {args.out} already exists; refusing to overwrite", file=sys.stderr)
        return 2
    systems = {n: Path(f[args.split]) for n, f in json.loads(args.config.read_text(encoding="utf-8"))["systems"].items()}
    table = crosscheck(systems, args.processed / "qrels" / f"qrels.{args.split}.txt", args.split, args.check_run, args.integration_src,
                       args.processed.resolve().parent)
    table.to_csv(args.out, index=False)
    worst = max(table[f"{c} diff"].max() for c in COLUMNS)
    print(f"{len(table)} runs on the {args.split} split; largest difference between P4 and check_run.py: {worst:.2e} (tolerance {TOLERANCE:g})")
    return 0 if worst <= TOLERANCE else 2


if __name__ == "__main__":
    sys.exit(main())
