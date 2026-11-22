"""Tables for the analysis sections of the report (popularity, leak B, extended-lambda sensitivity, ...), built from output files only.

Used by p4.experiments.make_report. Each builder returns Markdown; a table whose input folder does not exist is simply not offered, so a report
that needs it fails with "no table for placeholder" instead of showing a blank.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from p4.experiments.make_report import df_to_md

BUCKETS = ["0", "1-2", "3-10", "11-50", ">50"]
SHOWN_SYSTEMS = ["p2_abstract_bm25", "p2_context_bm25", "p2_expanded_bm25", "p3_minilm_abstract", "p3_minilm_contexts", "p3_minilm_expanded", "p3_hybrid_rrf",
                 "p2_abstract_bm25+authority", "p2_expanded_bm25+authority", "p3_minilm_expanded+authority", "p3_hybrid_rrf+authority"]


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def group_table(groups: pd.DataFrame, metric: str, order: list[str], systems=SHOWN_SYSTEMS, label_n: bool = True,
                corner: str = "citations of the target") -> str:
    """Mean score per group for each system; the header carries the number of queries and marks groups too small to compare."""
    g = groups[groups["metric"] == metric]
    sizes = g.drop_duplicates("group").set_index("group")
    present = [b for b in order if b in set(g["group"].astype(str))]
    g = g.assign(group=g["group"].astype(str))
    sizes = g.drop_duplicates("group").set_index("group")
    pivot = g.pivot(index="system", columns="group", values="mean").reindex([s for s in systems if s in set(g["system"])])[present]
    names = {b: f"{b} (n={int(sizes.loc[b, 'n_queries'])}{', small' if bool(sizes.loc[b, 'small_group']) else ''})" for b in present} if label_n else {}
    return df_to_md(pivot.rename(columns=names).round(3).reset_index().rename(columns={"system": f"system \\ {corner}"}), "{:.3f}")


def popularity_tables(analysis: Path) -> dict[str, str]:
    groups = _csv(analysis / "popularity_query_side_groups.csv")
    retrieval = _csv(analysis / "popularity_retrieval.csv").set_index("system").reindex(SHOWN_SYSTEMS).reset_index()
    cols = ["system", "top_decile_share_topk", "median_citations_topk", "n_wrong_top1", "wrong_top1_more_cited_rate"]
    # With 79% of papers uncited, the "most-cited 10%" cutoff collapses to "cited at least once" (about 21% of the corpus): label it as that.
    retrieval = retrieval[cols].rename(columns={"top_decile_share_topk": "share of top-10 papers cited at least once (corpus 21.4%)",
                                                "median_citations_topk": "median citations of a top-10 paper",
                                                "n_wrong_top1": "queries with a wrong top-1",
                                                "wrong_top1_more_cited_rate": "wrong top-1 more cited than the target"})
    return {"pop_mrr_by_bucket": group_table(groups, "rr@10", BUCKETS), "pop_recall_by_bucket": group_table(groups, "recall@10", BUCKETS),
            "pop_retrieval": df_to_md(retrieval, "{:.3f}")}


def leak_b_tables(real: Path) -> dict[str, str]:
    diag = _csv(real / "leak_b_diagnostic" / "leak_b_diagnostic.csv")
    d = diag[(diag["metric"] == "MRR@10") & (diag["flag"] == "same_paper_same_target") & diag["system"].isin(SHOWN_SYSTEMS[:7])].copy()
    d["difference (95% CI)"] = [f"{r.diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" for r in d.itertuples()]
    d = d.rename(columns={"n_with": "n with", "n_without": "n without", "mean_with": "MRR@10 with", "mean_without": "MRR@10 without"})
    d = d.set_index("system").reindex(SHOWN_SYSTEMS[:7]).reset_index()
    from p4.experiments import key_comparisons as KC
    unexposed = _csv(real / "key_comparisons_unexposed" / "key_comparisons.csv")
    return {"leak_b_table": df_to_md(d[["system", "n with", "n without", "MRR@10 with", "MRR@10 without", "difference (95% CI)"]], "{:.3f}"),
            "key_comparisons_unexposed_mrr": KC.to_markdown(unexposed[unexposed["metric"] == "MRR@10"])}


def sensitivity_table(real: Path) -> str:
    rows = []
    for system in ("p2_abstract_bm25", "p2_expanded_bm25"):
        t = _csv(real / "sensitivity_extended_lambda_val" / system / "fusion_tuning.csv")
        for prior, part in t.groupby("prior", sort=False):
            top = part.sort_values(["MRR@10", "lambda"], ascending=[False, True]).iloc[0]
            rows.append({"system": system, "prior": prior, "val MRR@10 (lambda 0)": part[part["lambda"] == 0.0]["MRR@10"].iloc[0],
                         "best val MRR@10, lambda 0-100": top["MRR@10"], "at lambda": top["lambda"], "val MRR@10 at lambda 100": part[part["lambda"] == 100.0]["MRR@10"].iloc[0]})
    return df_to_md(pd.DataFrame(rows))


def build(real: Path) -> dict[str, str]:
    tables: dict[str, str] = {}
    builders = [(lambda: popularity_tables(real / "run_main" / "analysis")), (lambda: leak_b_tables(real)),
                (lambda: {"sensitivity_lambda": sensitivity_table(real)}), (lambda: coldstart_tables(real / "run_main" / "analysis")), (lambda: exposure_tables(real))]
    for builder in builders:
        try:
            tables.update(builder())
        except FileNotFoundError:
            continue
    tables.update(build_extra(real))
    return tables


# ---------------------------------------------------------------------------------------------- cold start and exposure by bucket
CONTEXT_BUCKETS = ["0", "1", "2-5", "6-20", "21-100", ">100"]


def coldstart_tables(analysis: Path) -> dict[str, str]:
    groups = _csv(analysis / "coldstart_contexts_groups.csv")
    trend = _csv(analysis / "coldstart_contexts_trend.csv")
    t = trend[trend["metric"] == "rr@10"].set_index("system").reindex(SHOWN_SYSTEMS).reset_index()
    t["Spearman rho (95% CI)"] = [f"{r.spearman_rho:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" for r in t.itertuples()]
    t["significant (Holm)"] = ["yes" if r.significant_holm else "no" for r in t.itertuples()]
    return {"cold_mrr_by_contexts": group_table(groups, "rr@10", CONTEXT_BUCKETS, corner="indexed contexts of the target"),
            "cold_trend": df_to_md(t[["system", "Spearman rho (95% CI)", "p_holm", "significant (Holm)"]].rename(columns={"p_holm": "p (Holm)"}), "{:.3g}")}


def exposure_tables(real: Path) -> dict[str, str]:
    exposure = _csv(real / "leak_b_diagnostic" / "exposure_by_bucket.csv")
    exposure["share_same_paper_same_target"] = exposure["share_same_paper_same_target"].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    cells = _csv(real / "leak_b_diagnostic" / "mrr_by_bucket_and_exposure.csv")
    rows = []
    for system in ("p2_abstract_bm25", "p2_expanded_bm25", "p3_hybrid_rrf"):
        for bucket in BUCKETS:
            part = cells[(cells["system"] == system) & (cells["bucket"] == bucket)].set_index("same_paper_same_target_in_index")
            get = lambda flag, col: (part.loc[flag, col] if flag in part.index else float("nan"))
            rows.append({"system": system, "citations of target": bucket, "n exposed": int(get(True, "n_queries")) if True in part.index else 0,
                         "MRR@10 exposed": get(True, "MRR@10"), "n not exposed": int(get(False, "n_queries")) if False in part.index else 0,
                         "MRR@10 not exposed": get(False, "MRR@10")})
    return {"exposure_by_bucket": df_to_md(exposure.rename(columns={"bucket": "citations of the target (evaluation graph)", "n_queries": "test queries",
                                                                      "n_same_paper_same_target": "with a same-paper, same-target sentence in the index",
                                                                      "share_same_paper_same_target": "share"})),
            "mrr_bucket_exposure": df_to_md(pd.DataFrame(rows), "{:.3f}")}


# ---------------------------------------------------------------------------------------------- function, year, errors
def function_tables(analysis: Path) -> dict[str, str]:
    groups = _csv(analysis / "function_groups.csv")
    tests = _csv(analysis / "function_group_tests.csv")
    t = tests[tests["metric"] == "rr@10"].set_index("system").reindex(SHOWN_SYSTEMS).reset_index()
    t["significant (Holm)"] = ["yes" if r.significant_holm else "no" for r in t.itertuples()]
    return {"function_mrr": group_table(groups, "rr@10", ["background", "method", "result"], corner="predicted citation function"),
            "function_tests": df_to_md(t[["system", "h_statistic", "p_value", "p_holm", "significant (Holm)"]].rename(
                columns={"h_statistic": "Kruskal-Wallis H", "p_value": "p", "p_holm": "p (Holm)"}), "{:.3g}")}


def year_tables(analysis: Path) -> dict[str, str]:
    groups = _csv(analysis / "year_groups.csv")
    groups["group"] = groups["group"].astype(str)
    years = sorted(set(groups["group"]), key=int)
    trend = _csv(analysis / "year_trend.csv")
    t = trend[trend["metric"] == "rr@10"].set_index("system").reindex(SHOWN_SYSTEMS).reset_index()
    t["Spearman rho (95% CI)"] = [f"{r.spearman_rho:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" for r in t.itertuples()]
    t["significant (Holm)"] = ["yes" if r.significant_holm else "no" for r in t.itertuples()]
    return {"year_mrr": group_table(groups, "rr@10", years, corner="publication year of the target"), "year_trend": df_to_md(t[["system", "Spearman rho (95% CI)", "p_holm", "significant (Holm)"]].rename(
        columns={"p_holm": "p (Holm)"}), "{:.3g}")}


def error_tables(analysis: Path) -> dict[str, str]:
    s = _csv(analysis / "errors_summary.csv").set_index("system").reindex(SHOWN_SYSTEMS).reset_index()
    s = s[["system", "n_hits", "failure_rate", "median_rank_when_found", "n_rank 1", "n_rank 2-10", "n_rank 11 or deeper", "n_not retrieved"]].rename(columns={
        "n_hits": "hits in top 10", "failure_rate": "miss@10 rate", "median_rank_when_found": "median rank when found", "n_rank 1": "rank 1",
        "n_rank 2-10": "rank 2-10", "n_rank 11 or deeper": "rank 11+", "n_not retrieved": "not in the returned list"})
    return {"error_summary": df_to_md(s, "{:.3f}")}


_BUILDERS_EXTRA = [(lambda real: function_tables(real / "run_main" / "analysis")), (lambda real: year_tables(real / "run_main" / "analysis")),
                   (lambda real: error_tables(real / "run_main" / "analysis"))]


def build_extra(real: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for builder in _BUILDERS_EXTRA:
        try:
            out.update(builder(real))
        except FileNotFoundError:
            continue
    return out
