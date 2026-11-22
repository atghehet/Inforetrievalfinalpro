"""Command-line evaluation runner.

    python -m p4.eval.cli --qrels QRELS --run NAME=RUNFILE [--run ...] --out OUTDIR

(`python scripts/run_eval.py ...` is the same program.) See p4/README.md for examples.

What it does: reads the qrels and every run file, scores all systems on the same
queries with the P4 harness, compares each system with a baseline using paired
bootstrap + a paired test, and writes into OUTDIR:

    summary.csv       one row per system, every metric (means)
    per_query.csv     one row per (system, query): the raw scores
    comparisons.csv   paired significance results
    report.md         the readable summary
    efficiency_index.csv, efficiency_runs.csv     only with --efficiency

Exit code 0 = success, 2 = bad input (a one-line "error: ..." message is printed).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from .efficiency import EfficiencyFormatError, summarise
from .harness import (DEFAULT_CUTOFFS, DEFAULT_MIN_COVERAGE, RunQrelsMismatchError, evaluate_runs,
                      per_query_column, summary_name)
from .report import build_report
from .significance import (DEFAULT_ALPHA, DEFAULT_CONFIDENCE, DEFAULT_N_BOOT, DEFAULT_SEED, TESTS,
                           compare_result)
from .trec_io import TrecFormatError, read_qrels, read_run, run_tag

OUTPUT_FILES = ("summary.csv", "per_query.csv", "comparisons.csv", "report.md")


class CliError(Exception):
    """Bad command-line input. The message is shown to the user without a traceback."""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qrels", required=True, type=Path, help="qrels file: 'qid 0 docid relevance' per line")
    p.add_argument("--run", action="append", default=[], metavar="NAME=PATH",
                   help="a TREC run file; NAME=PATH gives it a table name, a bare PATH uses the file name. "
                        "Repeat for several systems")
    p.add_argument("--runs-glob", action="append", default=[], metavar="PATTERN",
                   help="add every run file matching a pattern (quote it); names come from the file names")
    p.add_argument("--name-from-tag", action="store_true",
                   help="name systems given without NAME= (and --runs-glob matches) by the run tag in the sixth column of the file")
    p.add_argument("--out", required=True, type=Path, help="output folder (created if needed)")
    p.add_argument("--cutoffs", default=",".join(map(str, DEFAULT_CUTOFFS)),
                   help="comma-separated cutoffs, default %(default)s")
    p.add_argument("--baseline", help="system the others are compared with (default: the first one)")
    p.add_argument("--test-metrics", help="metrics to test, e.g. MRR@10,nDCG@10 (default: MRR, nDCG and Recall "
                                          "at the headline cutoff)")
    p.add_argument("--test", choices=TESTS, default="wilcoxon", help="paired test (default %(default)s)")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="bootstrap resamples (default %(default)s)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default %(default)s)")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="significance level (default %(default)s)")
    p.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                   help="fraction of judged queries a run must contain, else it is rejected as the wrong split "
                        "(default %(default)s; 0 disables)")
    p.add_argument("--efficiency", type=Path, help="P2's efficiency.csv (optional)")
    p.add_argument("--tuning", type=Path, help="P2's tuning.csv, adds query times (needs --efficiency)")
    p.add_argument("--note", help="free text printed at the top of the report, e.g. 'SYNTHETIC sample data'")
    p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    p.add_argument("--quiet", action="store_true", help="do not print the summary table")
    p.add_argument("--debug", action="store_true", help="show the full traceback on errors")
    return p.parse_args(argv)


def collect_runs(run_args: list[str], glob_args: list[str], name_from_tag: bool = False) -> dict[str, Path]:
    """Build {system name: run file path} from --run and --runs-glob, validating every path.

    With `name_from_tag`, a file given without NAME= (and every --runs-glob match) is named by the run tag in its sixth column
    instead of its file name: P2 names frozen test runs like `context_bm25.run`, hiding which configuration won, while the tag
    (for example `context-w0-plain-method-bm25-0`) says it.
    """
    runs: dict[str, Path] = {}

    def add(name: str, path: Path) -> None:
        if not name:
            raise CliError(f"empty system name for run file {path}")
        if name in runs:
            raise CliError(f"two run files are called {name!r} ({runs[name]} and {path}); give them distinct "
                           "names with NAME=PATH")
        if not path.exists():
            raise CliError(f"run file not found: {path} (system {name!r})")
        if not path.is_file():
            raise CliError(f"run path is not a file: {path} (system {name!r})")
        runs[name] = path

    for spec in run_args:
        name, sep, path = spec.partition("=")
        if sep:
            add(name.strip(), Path(path.strip()))
        else:
            add(run_tag(spec) if name_from_tag and Path(spec).is_file() else Path(spec).stem, Path(spec))
    for pattern in glob_args:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise CliError(f"--runs-glob pattern matched no files: {pattern}")
        for match in matches:
            if Path(match).is_file():
                add(run_tag(match) if name_from_tag else Path(match).stem, Path(match))
    if not runs:
        raise CliError("no run files given: use --run NAME=PATH and/or --runs-glob PATTERN")
    return runs


def parse_cutoffs(text: str) -> tuple[int, ...]:
    try:
        cutoffs = tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError:
        raise CliError(f"--cutoffs must be whole numbers separated by commas, got {text!r}") from None
    if not cutoffs or any(k < 1 for k in cutoffs):
        raise CliError(f"--cutoffs must be whole numbers >= 1, got {text!r}")
    return tuple(sorted(set(cutoffs)))


def choose_test_metrics(spec: str | None, cutoffs: tuple[int, ...]) -> tuple[list[str], int]:
    """Return (per-query metric columns to test, headline cutoff)."""
    headline = 10 if 10 in cutoffs else max(cutoffs)
    names = [m.strip() for m in spec.split(",") if m.strip()] if spec else \
        [f"MRR@{headline}", f"nDCG@{headline}", f"Recall@{headline}"]
    columns = []
    for name in names:
        try:
            column = per_query_column(name)
        except ValueError as err:
            raise CliError(str(err)) from None
        if column != "ap" and int(column.split("@")[1]) not in cutoffs:
            raise CliError(f"metric {name!r} needs cutoff {column.split('@')[1]}, which is not in --cutoffs {list(cutoffs)}")
        columns.append(column)
    return columns, headline


def check_outputs(out: Path, overwrite: bool, extra: tuple[str, ...]) -> None:
    if out.exists() and not out.is_dir():
        raise CliError(f"--out is not a folder: {out}")
    existing = [f for f in OUTPUT_FILES + extra if (out / f).exists()]
    if existing and not overwrite:
        raise CliError(f"{out} already contains {existing}; use --overwrite or a new folder")


def run(args: argparse.Namespace) -> int:
    runs = collect_runs(args.run, args.runs_glob, args.name_from_tag)
    cutoffs = parse_cutoffs(args.cutoffs)
    metrics, headline = choose_test_metrics(args.test_metrics, cutoffs)
    if args.baseline is not None and args.baseline not in runs:
        raise CliError(f"--baseline {args.baseline!r} is not one of the systems {list(runs)}")
    if args.tuning and not args.efficiency:
        raise CliError("--tuning needs --efficiency as well")
    if not args.qrels.is_file():
        raise CliError(f"qrels file not found: {args.qrels}")
    extra = ("efficiency_index.csv", "efficiency_runs.csv") if args.efficiency else ()
    check_outputs(args.out, args.overwrite, extra)

    qrels = read_qrels(args.qrels)
    result = evaluate_runs({name: read_run(path) for name, path in runs.items()}, qrels, cutoffs,
                           min_coverage=args.min_coverage)
    baseline = args.baseline or next(iter(runs))
    if len(runs) > 1:
        comparisons = compare_result(result, metrics, baseline=baseline, n_boot=args.n_boot, seed=args.seed,
                                     confidence=DEFAULT_CONFIDENCE, alpha=args.alpha, test=args.test)
    else:
        comparisons = compare_result(result, [], baseline=baseline)     # empty table with the right columns
    efficiency = summarise(args.efficiency, args.tuning) if args.efficiency else None

    args.out.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(args.out / "summary.csv", index=False)
    result.per_query.to_csv(args.out / "per_query.csv", index=False)
    comparisons.to_csv(args.out / "comparisons.csv", index=False)
    if efficiency is not None:
        efficiency.index_cost.to_csv(args.out / "efficiency_index.csv", index=False)
        if efficiency.runs is not None:
            efficiency.runs.to_csv(args.out / "efficiency_runs.csv", index=False)
    settings = {"qrels": args.qrels, "run files": "; ".join(f"{n} = {p}" for n, p in runs.items()),
                "cutoffs": list(cutoffs), "baseline": baseline, "tested metrics": ", ".join(summary_name(m) for m in metrics),
                "paired test": args.test, "bootstrap resamples": args.n_boot, "seed": args.seed,
                "alpha": args.alpha, "minimum query coverage": args.min_coverage}
    report = build_report(result=result, comparisons=comparisons, settings=settings, headline_k=headline,
                          efficiency=efficiency, note=args.note)
    (args.out / "report.md").write_text(report, encoding="utf-8")

    if not args.quiet:
        shown = [c for c in ("system", "n_queries", "n_no_results", f"Recall@{headline}", f"MRR@{headline}",
                             f"nDCG@{headline}", "MAP") if c in result.summary.columns]
        print(result.summary[shown].round(4).to_string(index=False))
    print(f"\nWrote {', '.join(OUTPUT_FILES + extra)} to {args.out}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (CliError, TrecFormatError, RunQrelsMismatchError, EfficiencyFormatError,
            FileNotFoundError, IsADirectoryError, PermissionError, ValueError) as err:
        if args.debug:
            raise
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
