"""Launcher for the P4 evaluation CLI (same as `python -m p4.eval.cli`). Run with --help for options.

    python scripts/run_eval.py --qrels QRELS --run NAME=RUNFILE --out OUTDIR

The program itself lives in p4/eval/cli.py; this file only makes the repository root importable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p4.eval.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
