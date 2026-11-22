"""Diagnostic for leak B: how much do systems score higher on queries whose citing paper has OTHER sentences in the retrieval index?

    python -m p4.experiments.leak_b_diagnostic --processed data/processed --per-query <eval>/per_query.csv --split test --out <NEW folder>

P1 removes the query sentence (and near-duplicates) from the index but keeps other sentences written by the same citing paper. For each query:
  same_paper_same_target  the index holds another sentence of the SAME citing paper that cites the SAME target paper (the strongest form)
  same_paper_any          the index holds any other sentence of the same citing paper
Per system we compare the mean per-query score of queries WITH and WITHOUT the flag, with an unpaired bootstrap interval for the difference.
This is an observational comparison, not a controlled experiment: queries with the flag may also differ in other ways (for example longer papers
with several citations of the same work). It shows how large the exposure is and whether the scores move with it; it cannot remove the effect.
A system that does not use citation contexts (abstract-only) is a useful control: it cannot benefit from other sentences of the citing paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRIC_COLUMNS = {"MRR@10": "rr@10", "Recall@10": "recall@10"}


def query_flags(processed: Path, split: str) -> pd.DataFrame:
    queries = pd.read_parquet(processed / "queries.parquet", columns=["query_id", "split", "citing_paper_id", "cited_paper_id", "context_id"])
    index = pd.read_parquet(processed / "index_contexts.parquet", columns=["context_id", "citing_paper_id", "cited_paper_id"])
    q = queries[queries["split"] == split].copy()
    pairs = set(zip(index["citing_paper_id"], index["cited_paper_id"]))
    citers = set(index["citing_paper_id"])
    q["same_paper_same_target"] = [(c, t) in pairs for c, t in zip(q["citing_paper_id"], q["cited_paper_id"])]
    q["same_paper_any"] = q["citing_paper_id"].isin(citers)
    return q[["query_id", "same_paper_same_target", "same_paper_any"]]


BUCKET_BINS = [-1, 0, 2, 10, 50, np.inf]
BUCKET_LABELS = ["0", "1-2", "3-10", "11-50", ">50"]


def by_bucket(processed: Path, split: str, flags: pd.DataFrame, priors: pd.DataFrame, per_query: pd.DataFrame, systems: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exposure and MRR@10 by the target's evaluation-graph citation bucket (0, 1-2, 3-10, 11-50, >50).

    A target with 0 citations in the evaluation graph is cited only by validation/test papers, so every index sentence about it comes from a
    held-out citing paper, possibly the very paper that wrote the query. Returns (exposure table, MRR@10 by bucket and exposure per system).
    """
    queries = pd.read_parquet(processed / "queries.parquet", columns=["query_id", "cited_paper_id"])
    counts = priors.set_index("paper_id")["citation_count_eval"]
    q = flags.merge(queries, on="query_id")
    q["bucket"] = pd.cut(q["cited_paper_id"].map(counts), BUCKET_BINS, labels=BUCKET_LABELS).astype(str)
    exposure = q.groupby("bucket").agg(n_queries=("query_id", "size"), n_same_paper_same_target=("same_paper_same_target", "sum")).reindex(BUCKET_LABELS).reset_index()
    exposure[["n_queries", "n_same_paper_same_target"]] = exposure[["n_queries", "n_same_paper_same_target"]].fillna(0).astype(int)   # an empty bucket: 0 queries
    exposure["share_same_paper_same_target"] = exposure["n_same_paper_same_target"] / exposure["n_queries"].where(exposure["n_queries"] > 0)    # share undefined (NaN)
    scores = per_query[per_query["system"].isin(systems)].merge(q[["query_id", "bucket", "same_paper_same_target"]], on="query_id")
    cells = (scores.groupby(["system", "bucket", "same_paper_same_target"])["rr@10"].agg(["size", "mean"]).reset_index()
             .rename(columns={"size": "n_queries", "mean": "MRR@10", "same_paper_same_target": "same_paper_same_target_in_index"}))
    return exposure, cells


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    low, high = np.quantile(diffs, [0.025, 0.975])
    return float(low), float(high)


def diagnostic(per_query: pd.DataFrame, flags: pd.DataFrame, systems: list[str], *, n_boot: int = 5000, seed: int = 42) -> pd.DataFrame:
    rows = []
    merged = per_query[per_query["system"].isin(systems)].merge(flags, on="query_id", how="inner")
    for system in systems:
        part = merged[merged["system"] == system]
        for flag in ("same_paper_same_target", "same_paper_any"):
            for metric, column in METRIC_COLUMNS.items():
                a, b = part.loc[part[flag], column].to_numpy(), part.loc[~part[flag], column].to_numpy()
                row = {"system": system, "flag": flag, "metric": metric, "n_with": len(a), "n_without": len(b),
                       "mean_with": float(a.mean()) if len(a) else np.nan, "mean_without": float(b.mean()) if len(b) else np.nan}
                if len(a) >= 10 and len(b) >= 10:
                    row["diff"] = row["mean_with"] - row["mean_without"]
                    row["ci_low"], row["ci_high"] = _bootstrap_diff(a, b, n_boot, seed)
                else:
                    row["diff"] = row["ci_low"] = row["ci_high"] = np.nan                 # too few queries in a group to compare
                rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--per-query", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--systems", nargs="+", help="systems to include (default: all that are not fused)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--priors", type=Path, help="priors.parquet of the evaluation graph: also reports exposure and MRR@10 by citation bucket")
    ap.add_argument("--out", required=True, type=Path, help="NEW folder")
    args = ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"error: {args.out} is not empty", file=sys.stderr)
        return 2
    try:
        per_query = pd.read_csv(args.per_query)
        flags = query_flags(args.processed, args.split)
    except (FileNotFoundError, KeyError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    systems = args.systems or [s for s in dict.fromkeys(per_query["system"]) if not s.endswith("+authority")]
    unknown = sorted(set(systems) - set(per_query["system"]))
    if unknown:
        print(f"error: systems not in {args.per_query.name}: {unknown}", file=sys.stderr)
        return 2
    table = diagnostic(per_query, flags, systems, n_boot=args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "leak_b_diagnostic.csv", index=False)
    exposure = {"queries": len(flags), "same_paper_same_target": int(flags["same_paper_same_target"].sum()), "same_paper_any": int(flags["same_paper_any"].sum())}
    (args.out / "exposure.json").write_text(json.dumps(exposure, indent=2), encoding="utf-8")
    for name, mask in (("queries_without_same_paper_same_target.txt", ~flags["same_paper_same_target"]),
                       ("queries_with_same_paper_same_target.txt", flags["same_paper_same_target"])):
        (args.out / name).write_text("".join(f"{q}\n" for q in flags.loc[mask, "query_id"]), encoding="utf-8")
    if args.priors:
        exposure_table, cells = by_bucket(args.processed, args.split, flags, pd.read_parquet(args.priors), per_query, systems)
        exposure_table.to_csv(args.out / "exposure_by_bucket.csv", index=False)
        cells.to_csv(args.out / "mrr_by_bucket_and_exposure.csv", index=False)
    print(f"{args.split}: {exposure['queries']} queries; {exposure['same_paper_same_target']} with another sentence of the same paper citing the same target, "
          f"{exposure['same_paper_any']} with any other sentence of the same paper; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
