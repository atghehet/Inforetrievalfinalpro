"""Command-line interface for the graph algorithms.

    python -m p4.graph.cli pagerank  --edges EDGES [--papers PAPERS] [--top-k 20] [--out FILE.csv]
    python -m p4.graph.cli cocitation --edges EDGES --paper PAPER_ID [--top-k 10] [--out FILE.csv]

EDGES is P1's edges.parquet (columns citing_paper_id, cited_paper_id, num_contexts) or a .csv/.tsv with those columns.
PAPERS (optional) is P1's papers.parquet: it adds papers that have no edges and shows titles.
Exit code 0 = success, 2 = bad input (one line "error: ..." is printed, no traceback).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .cocitation import CoCitationIndex, UnknownPaperError
from .edges import EdgeFormatError, read_edges, read_paper_ids
from .pagerank import DEFAULT_DAMPING, DEFAULT_MAX_ITER, DEFAULT_TOL, pagerank


class CliError(Exception):
    """Bad command-line input; shown without a traceback."""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p4.graph.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, default_k):
        sp.add_argument("--edges", required=True, type=Path, help="edges.parquet (or .csv/.tsv)")
        sp.add_argument("--papers", type=Path, help="papers.parquet (or .csv/.tsv): adds papers without edges, shows titles")
        sp.add_argument("--top-k", type=int, default=default_k, help="rows to show/save (default %(default)s)")
        sp.add_argument("--out", type=Path, help="also write the table to this CSV file")
        sp.add_argument("--overwrite", action="store_true", help="replace --out if it exists")
        sp.add_argument("--debug", action="store_true", help="show the full traceback on errors")

    pr = sub.add_parser("pagerank", help="rank papers by PageRank", description=pagerank.__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    common(pr, 20)
    pr.add_argument("--damping", type=float, default=DEFAULT_DAMPING, help="default %(default)s")
    pr.add_argument("--tol", type=float, default=DEFAULT_TOL, help="convergence tolerance, default %(default)s")
    pr.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="iteration limit, default %(default)s")
    pr.add_argument("--weighted", action="store_true", help="weight citations by num_contexts")

    co = sub.add_parser("cocitation", help="recommend papers cited together with a given paper")
    common(co, 10)
    co.add_argument("--paper", required=True, help="paper_id to find recommendations for")
    co.add_argument("--max-references", type=int, help="ignore citing papers with more references than this")
    return p.parse_args(argv)


def _load(args):
    if args.out is not None and args.out.exists() and not args.overwrite:
        raise CliError(f"{args.out} already exists; use --overwrite or another file name")
    weight_col = "num_contexts" if getattr(args, "weighted", False) else None
    table = read_edges(args.edges, weight_col=weight_col)
    papers = read_paper_ids(args.papers) if args.papers else None
    return table, papers


def _with_titles(frame: pd.DataFrame, papers: pd.DataFrame | None) -> pd.DataFrame:
    if papers is not None and "title" in papers.columns and len(frame):
        return frame.merge(papers[["paper_id", "title"]], on="paper_id", how="left")
    return frame


def _emit(frame: pd.DataFrame, args) -> None:
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.6f}") if len(frame) else "(no rows)")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(f"\nWrote {len(frame)} rows to {args.out}")


def run(args) -> int:
    table, papers = _load(args)
    nodes = None if papers is None else papers["paper_id"]
    stats = table.stats
    if args.command == "pagerank":
        result = pagerank(table, nodes=nodes, damping=args.damping, tol=args.tol, max_iter=args.max_iter,
                          weighted=args.weighted, top_k=args.top_k)
        print(f"PageRank: {result.n_nodes} papers, {result.n_edges} edges "
              f"({stats.n_self_loops} self-citations and {stats.n_duplicates} duplicate edges removed), "
              f"{result.n_dangling} cite nothing; damping {result.damping}; "
              f"{'converged' if result.converged else 'DID NOT CONVERGE'} after {result.iterations} iterations "
              f"(last change {result.final_delta:.2e}).\n")
        if not result.converged:
            print("warning: PageRank did not converge within --max-iter; scores are approximate", file=sys.stderr)
        _emit(_with_titles(result.ranking, papers), args)
        return 0
    index = CoCitationIndex.from_edges(table, nodes=nodes, max_references=args.max_references)
    recommendations = index.recommend(args.paper, top_k=args.top_k)
    print(f"Co-citation: {index.stats.n_papers} papers, {index.stats.n_pairs} co-cited pairs from "
          f"{index.stats.n_citing_papers} citing papers ({index.stats.n_citing_skipped} skipped for too many references).\n"
          f"Recommendations for {args.paper}:\n")
    _emit(_with_titles(recommendations, papers), args)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (CliError, EdgeFormatError, UnknownPaperError, FileNotFoundError, IsADirectoryError, PermissionError,
            ValueError) as err:
        if args.debug:
            raise
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
