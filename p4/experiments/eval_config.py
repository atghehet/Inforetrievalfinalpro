"""Evaluate every system of a runs.json on one split with the normal evaluation tool (p4.eval.cli).

    python -m p4.experiments.eval_config --config runs.json --processed data/processed --split val --out <NEW folder> [--n-boot 5000] [--note TEXT]

Used for the VALIDATION tables (analysis and tuning). The frozen TEST evaluation is done once by p4.experiments.run_all.
Extra arguments after `--` go to p4.eval.cli unchanged (for example `-- --test-metrics MRR@10`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from p4.eval import cli as eval_cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--note", default="")
    args, extra = ap.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    try:
        raw = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        print(f"error: cannot read {args.config}: {err}", file=sys.stderr)
        return 2
    systems = raw.get("systems") or {}
    if not systems:
        print("error: the configuration has no systems", file=sys.stderr)
        return 2
    run_args = [a for name, files in systems.items() for a in ("--run", f"{name}={files[args.split]}")]
    baseline = raw.get("baseline") or next(iter(systems))
    qrels = args.processed / "qrels" / f"qrels.{args.split}.txt"
    note = f"{args.note} split: {args.split}; systems from {args.config.name}".strip()
    return eval_cli.main(["--qrels", str(qrels), *run_args, "--out", str(args.out), "--baseline", baseline, "--n-boot", str(args.n_boot),
                          "--note", note, "--quiet", *extra])


if __name__ == "__main__":
    sys.exit(main())
