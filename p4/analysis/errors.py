"""Error analysis: where do systems fail, which relevant papers are missed, and how do systems differ?

Terms (K is the cutoff, default 10)
  rank          1-based position of the relevant paper in a system's run, NaN if it is not in the run at all
                (it was outside the run's depth, or the system returned nothing for the query)
  hit@K         the relevant paper is in the top K;  failure = no hit@K
  category      "rank 1" | "rank 2-K" | "rank K+1 or deeper" | "not retrieved"
  hub paper     a paper that a system puts at rank 1 for many queries where it is NOT the relevant paper

Rank depth caveat: a run lists only its top results (P2: up to 100, P3 dense/hybrid: every paper). "not retrieved" means
"not in this run", and runs of different depth are not equally informative about deep ranks.

The comparisons between systems are descriptive counts (how many queries one system hits and the other misses) with an
exact binomial test on those discordant queries. They are associations in the sample, not explanations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from p4.eval.significance import holm_adjust


def categorise(rank: float, k: int) -> str:
    if rank != rank:
        return "not retrieved"
    if rank == 1:
        return "rank 1"
    return f"rank 2-{k}" if rank <= k else f"rank {k + 1} or deeper"


def category_order(k: int) -> list[str]:
    return ["rank 1", f"rank 2-{k}", f"rank {k + 1} or deeper", "not retrieved"]


def relevant_ranks(runs: dict, qrels: dict, k: int = 10) -> pd.DataFrame:
    """One row per (system, judged query): rank of the best-ranked relevant paper, top-1 document, category."""
    rows = []
    for system, run in runs.items():
        for qid in sorted(qrels):
            relevant = qrels[qid]
            ranked = [doc for doc, _ in run.get(qid, [])]
            rank = next((i for i, d in enumerate(ranked, start=1) if d in relevant), float("nan"))
            top1 = ranked[0] if ranked else None
            rows.append({"system": system, "query_id": qid, "target_id": sorted(relevant)[0], "rank": rank,
                         "run_depth": len(ranked), "top1_id": top1, "top1_is_relevant": top1 in relevant,
                         "hit": bool(rank == rank and rank <= k), "category": categorise(rank, k)})
    return pd.DataFrame(rows)


def error_summary(ranks: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per system: queries, hits, failure rate and how the failures are distributed over the categories."""
    rows = []
    for system, part in ranks.groupby("system", sort=False):
        counts = part["category"].value_counts()
        found = part["rank"].dropna()
        row = {"system": system, "n_queries": len(part), "k": k, "n_hits": int(part["hit"].sum()),
               "n_failures": int((~part["hit"]).sum()), "failure_rate": float((~part["hit"]).mean()),
               "median_rank_when_found": float(found.median()) if len(found) else np.nan}
        for name in category_order(k):
            row[f"n_{name}"] = int(counts.get(name, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_outcomes(ranks: pd.DataFrame, baseline: str, alpha: float = 0.05) -> pd.DataFrame:
    """Each system against the baseline on the same queries: both hit, only the system, only the baseline, neither.

    p_value: exact two-sided binomial (sign) test on the discordant queries (only one of the two hits). NaN when there
    are no discordant queries. p_holm corrects across the systems compared.
    """
    wide = ranks.pivot(index="query_id", columns="system", values="hit")
    if baseline not in wide.columns:
        raise ValueError(f"baseline {baseline!r} is not one of the systems {list(wide.columns)}")
    rows = []
    for system in wide.columns:
        if system == baseline:
            continue
        a, b = wide[system].astype(bool), wide[baseline].astype(bool)
        only_system, only_baseline = int((a & ~b).sum()), int((~a & b).sum())
        discordant = only_system + only_baseline
        p = float(stats.binomtest(only_system, discordant, 0.5).pvalue) if discordant else float("nan")
        rows.append({"system": system, "baseline": baseline, "n_queries": len(wide), "both_hit": int((a & b).sum()),
                     "only_system_hits": only_system, "only_baseline_hits": only_baseline,
                     "neither_hits": int((~a & ~b).sum()), "n_discordant": discordant, "p_value": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out.assign(p_holm=[], significant_holm=[])
    out["p_holm"] = holm_adjust(out["p_value"])
    out["significant_holm"] = out["p_holm"] < alpha
    return out


def hard_queries(ranks: pd.DataFrame, query_table: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Queries that the most systems fail, with text and the relevant paper's title. Ties: lowest query_id first."""
    n_systems = ranks["system"].nunique()
    grouped = ranks.groupby("query_id")
    table = pd.DataFrame({
        "n_systems": n_systems, "n_systems_failing": grouped["hit"].apply(lambda s: int((~s).sum())),
        "systems_failing": grouped.apply(lambda g: "; ".join(g.loc[~g["hit"], "system"]), include_groups=False),
        "best_rank_any_system": grouped["rank"].min()})
    table["failed_by_all"] = table["n_systems_failing"] == n_systems
    table = table.join(query_table[["target_id", "text", "title", "year", "citation_count", "function"]], how="left")
    return table.reset_index().sort_values(["n_systems_failing", "query_id"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def hub_papers(ranks: pd.DataFrame, papers: pd.DataFrame | None, top_n: int = 10) -> pd.DataFrame:
    """Per system: the papers most often returned at rank 1 although they are not the relevant paper."""
    wrong = ranks[ranks["top1_id"].notna() & ~ranks["top1_is_relevant"]]
    counts = wrong.groupby(["system", "top1_id"]).size().rename("n_queries_wrong_top1").reset_index()
    counts = counts.sort_values(["system", "n_queries_wrong_top1", "top1_id"], ascending=[True, False, True])
    counts = counts.groupby("system", sort=False).head(top_n).rename(columns={"top1_id": "paper_id"})
    if papers is not None:
        info = papers.set_index("paper_id")
        for col in ("title", "citation_count", "year"):
            counts[col] = counts["paper_id"].map(info[col]) if col in info.columns else np.nan
    return counts.reset_index(drop=True)


def missed_papers(ranks: pd.DataFrame, query_table: pd.DataFrame) -> pd.DataFrame:
    """Per relevant paper: in how many (system, query) pairs was it missed, and was it missed by every system."""
    grouped = ranks.groupby("target_id")
    table = pd.DataFrame({
        "n_queries": grouped["query_id"].nunique(),
        "n_pairs": grouped.size(), "n_pairs_missed": grouped["hit"].apply(lambda s: int((~s).sum())),
        "systems_that_missed": grouped.apply(lambda g: "; ".join(sorted(set(g.loc[~g["hit"], "system"]))), include_groups=False)})
    table["missed_in_every_pair"] = table["n_pairs_missed"] == table["n_pairs"]
    info = query_table.drop_duplicates("target_id").set_index("target_id")
    for col in ("title", "year", "citation_count"):
        table[col] = table.index.map(info[col]) if col in info.columns else np.nan
    table = table[table["n_pairs_missed"] > 0]
    return table.reset_index().sort_values(["n_pairs_missed", "target_id"], ascending=[False, True]).reset_index(drop=True)
