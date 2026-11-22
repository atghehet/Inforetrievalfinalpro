"""Command line for the P4 analyses.

    python -m p4.analysis.cli popularity|coldstart|year|function|errors --data-label real|synthetic|demo \\
        --qrels QRELS --run NAME=RUNFILE [--run ...] --queries QUERIES --papers PAPERS --out DIR [options]

--data-label is REQUIRED and is written into every output. It is your declaration and is not verified.
Inputs: P1's queries.parquet and papers.parquet (and context_functions.parquet for `function`, index_contexts.parquet
for the optional context-count axis of `coldstart`). A missing input is an error; nothing is substituted.
Outputs in DIR: <table>.csv files (each with a data_label column) and <analysis>_report.md.
Exit code 0 = success, 2 = bad input (one "error: ..." line).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from p4.eval.cli import CliError, collect_runs, parse_cutoffs
from p4.eval.harness import RunQrelsMismatchError
from p4.eval.significance import DEFAULT_ALPHA, DEFAULT_N_BOOT, DEFAULT_SEED, TESTS
from p4.eval.trec_io import TrecFormatError, read_qrels, read_run

from .analyses import ANALYSES, Options, prepare_context
from .data import (DATA_LABELS, AnalysisDataError, load_functions, load_index_context_counts, load_papers, load_queries)
from .groups import DEFAULT_MIN_GROUP_SIZE


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p4.analysis.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("analysis", choices=sorted(ANALYSES), help="which analysis to run")
    p.add_argument("--data-label", required=True, choices=DATA_LABELS,
                   help="what the data is (declared by you, not verified); written into every output")
    p.add_argument("--qrels", required=True, type=Path)
    p.add_argument("--run", action="append", default=[], metavar="NAME=PATH", help="a TREC run file; repeat per system")
    p.add_argument("--runs-glob", action="append", default=[], metavar="PATTERN")
    p.add_argument("--name-from-tag", action="store_true", help="name systems by the run tag (sixth column) instead of the file name")
    p.add_argument("--queries", type=Path, help="P1 queries.parquet (text, cited_citation_count, context_id)")
    p.add_argument("--papers", type=Path, help="P1 papers.parquet (title, year, citation_count)")
    p.add_argument("--citation-counts", type=Path, metavar="PRIORS.parquet",
                   help="priors.parquet from p4.graph.eval_graph: use its leakage-safe `citation_count_eval` INSTEAD of "
                        "papers.citation_count (which includes each query's own citing paper). Recommended for every real result")
    p.add_argument("--functions", type=Path, help="P1 context_functions.parquet (needed for `function`)")
    p.add_argument("--index-contexts", type=Path, help="P1 index_contexts.parquet (optional, for `coldstart`)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--baseline", help="system the others are compared with (default: the first --run)")
    p.add_argument("--k", type=int, default=10, help="cutoff for the scores and for 'failure' (default %(default)s)")
    p.add_argument("--cutoffs", default="1,5,10,20,50,100")
    p.add_argument("--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE, help="default %(default)s")
    p.add_argument("--year-bins", help="comma-separated upper edges, e.g. 2019,2021 gives <=2019, 2020-2021, >=2022")
    p.add_argument("--min-confidence", type=float, help="citation-function labels below this confidence count as unlabelled")
    p.add_argument("--test", choices=TESTS, default="wilcoxon")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--min-coverage", type=float, default=0.5)
    p.add_argument("--top-n", type=int, default=20, help="rows in the hardest-queries / missed-papers lists")
    p.add_argument("--analysis-date", help="YYYY-MM-DD printed in the report (default: today, UTC); set it for identical reruns")
    p.add_argument("--note", help="free text printed at the top of the report")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def load_priors_table(path) -> "pd.Series":
    """Read `citation_count_eval` (index = paper_id) from a priors.parquet written by p4.graph.eval_graph."""
    import pandas as pd
    path = Path(path)
    if not path.is_file():
        raise CliError(f"citation-counts file not found: {path}")
    df = pd.read_parquet(path)
    if "paper_id" not in df.columns or "citation_count_eval" not in df.columns:
        raise CliError(f"{path.name} has no paper_id / citation_count_eval columns (expected priors.parquet from p4.graph.eval_graph)")
    return df.set_index("paper_id")["citation_count_eval"]


def _year_bins(text):
    if not text:
        return None
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise CliError(f"--year-bins must be whole years separated by commas, got {text!r}") from None


def run(args) -> int:
    runs = collect_runs(args.run, args.runs_glob, args.name_from_tag)
    cutoffs = parse_cutoffs(args.cutoffs)
    if args.k < 1:
        raise CliError(f"--k must be >= 1, got {args.k}")
    if not args.qrels.is_file():
        raise CliError(f"qrels file not found: {args.qrels}")
    if args.analysis == "function" and not args.functions:
        raise CliError("the `function` analysis needs --functions (P1's context_functions.parquet); "
                       "no labels are invented when it is missing")
    if args.analysis != "errors" and not args.papers:
        raise CliError(f"the `{args.analysis}` analysis needs --papers (P1's papers.parquet)")
    if args.out.exists() and not args.out.is_dir():
        raise CliError(f"--out is not a folder: {args.out}")
    report_name = f"{args.analysis}_report.md"
    if (args.out / report_name).exists() and not args.overwrite:
        raise CliError(f"{args.out / report_name} already exists; use --overwrite or a new folder")
    options = Options(k=args.k, cutoffs=cutoffs, min_group_size=args.min_group_size, n_boot=args.n_boot, seed=args.seed,
                      alpha=args.alpha, test=args.test, baseline=args.baseline, year_bins=_year_bins(args.year_bins),
                      min_confidence=args.min_confidence, min_coverage=args.min_coverage, top_n=args.top_n,
                      analysis_date=args.analysis_date, note=args.note)
    qrels = read_qrels(args.qrels)
    loaded_runs = {name: read_run(path) for name, path in runs.items()}
    queries = load_queries(args.queries) if args.queries else None
    papers = load_papers(args.papers) if args.papers else None
    counts_source = "papers.citation_count (includes each query's own citing paper; NOT leakage-safe)"
    if args.citation_counts:
        if papers is None:
            raise CliError("--citation-counts needs --papers")
        safe = load_priors_table(args.citation_counts)
        papers = papers.assign(citation_count=papers["paper_id"].map(safe))
        counts_source = f"citation_count_eval from {args.citation_counts} (leakage-safe evaluation graph)"
    functions = load_functions(args.functions) if args.functions else None
    index_counts = load_index_context_counts(args.index_contexts) if args.index_contexts else None
    input_paths = {"qrels": args.qrels, **{f"run '{n}'": p for n, p in runs.items()}}
    for label, path in (("queries", args.queries), ("papers", args.papers), ("context functions", args.functions),
                        ("index contexts", args.index_contexts)):
        if path:
            input_paths[label] = path
    ctx = prepare_context(data_label=args.data_label, runs=loaded_runs, qrels=qrels, queries=queries, papers=papers,
                          functions=functions, index_counts=index_counts, options=options, input_paths=input_paths)
    if papers is not None:
        ctx.stats["citation counts used"] = counts_source
    output = ANALYSES[args.analysis](ctx)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in output.tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
    (args.out / report_name).write_text(output.report, encoding="utf-8")
    if not args.quiet:
        print(f"[data label: {args.data_label}] {args.analysis}: {len(output.tables)} tables")
    print(f"Wrote {report_name} and {len(output.tables)} CSV files to {args.out}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (CliError, TrecFormatError, RunQrelsMismatchError, AnalysisDataError, FileNotFoundError, IsADirectoryError,
            PermissionError, ValueError) as err:
        if args.debug:
            raise
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
