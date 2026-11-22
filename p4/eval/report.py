"""Turn evaluation results into a readable Markdown report (no extra libraries needed)."""
from __future__ import annotations

import math

import pandas as pd

from .harness import summary_name


def markdown_table(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Render a DataFrame as a Markdown table. NaN is shown as 'n/a'."""
    def cell(value) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return "n/a" if math.isnan(value) else float_format.format(value)
        return str(value).replace("|", "/")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, rule, *body])


def _p(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def build_report(*, result, comparisons: pd.DataFrame, settings: dict, headline_k: int,
                 efficiency=None, note: str | None = None) -> str:
    """Assemble the Markdown report.

    result       EvalResult from the harness
    comparisons  DataFrame from significance.compare_result (may be empty)
    settings     dict of run settings to print (qrels path, run files, seed, ...)
    headline_k   cutoff shown in the summary table (the CSV files hold every cutoff)
    efficiency   EfficiencySummary or None
    note         free text printed at the top, e.g. which dataset these numbers come from
    """
    out = ["# Evaluation report", ""]
    if note:
        out += [f"> **Note:** {note}", ""]

    out += ["## Inputs", ""]
    for key, value in settings.items():
        out.append(f"- **{key}:** {value}")
    n_queries = int(result.summary["n_queries"].iloc[0])
    out += [f"- **judged queries scored:** {n_queries}", ""]

    columns = ["system", "n_queries", "n_no_results", f"Recall@{headline_k}", f"MRR@{headline_k}",
               f"nDCG@{headline_k}", "MAP"]
    summary = result.summary[[c for c in columns if c in result.summary.columns]]
    out += [f"## Summary metrics (cutoff {headline_k})", "",
            markdown_table(summary), "",
            "`n_no_results` = judged queries for which the system returned nothing (they score 0). "
            "All cutoffs are in `summary.csv`. MAP is computed over whatever depth each run file lists, "
            "so it is only comparable between runs of similar depth.", ""]

    flagged = {s: w for s, w in result.warnings.items() if w["missing"] or w["extra"]}
    if flagged:
        out += ["## Query coverage warnings", ""]
        for system, w in flagged.items():
            out.append(f"- **{system}:** {len(w['missing'])} judged queries had no results (scored 0); "
                       f"{len(w['extra'])} queries in the run have no judgments (ignored).")
        out.append("")

    out += ["## Statistical comparisons", ""]
    if comparisons.empty:
        out += ["Not computed (needs at least two systems).", ""]
    else:
        out += [f"Each system is compared with the baseline **{settings.get('baseline', '?')}** on the same "
                "queries. `mean_diff` = system minus baseline; the interval is a paired bootstrap confidence "
                "interval; `p` comes from a paired per-query test; `p_holm` corrects for the several systems "
                "compared, and `sig. (Holm)` is the safer verdict when many systems are compared with one "
                "baseline. Significance is judged from the per-query test, never from the averages alone.", ""]
        for metric, group in comparisons.groupby("metric", sort=False):
            level = f"{group['ci_level'].iloc[0]:.0%}"
            table = pd.DataFrame({
                "system": group["system"], "n": group["n_queries"], "mean_system": group["mean_system"],
                "mean_baseline": group["mean_baseline"], "mean_diff": group["mean_diff"],
                f"CI {level} low": group["ci_low"], f"CI {level} high": group["ci_high"],
                "p": [_p(v) for v in group["p_value"]], "p_holm": [_p(v) for v in group["p_holm"]],
                "sig. (raw p)": group["significant"], "sig. (Holm)": group["significant_holm"]})
            out += [f"### {summary_name(metric)}", "", markdown_table(table), ""]
            out += [f"- {r.system}: {r.conclusion}" + (f" ({r.note})" if r.note else "")
                    for r in group.itertuples()]
            out.append("")

    out += ["## Efficiency (from P2's logs)", ""]
    if efficiency is None:
        out += ["Not requested (pass `--efficiency` with P2's efficiency.csv).", ""]
    else:
        cost = efficiency.index_cost.rename(columns={"size_mib": "size_MiB"})[["index", "build_seconds", "size_MiB"]]
        out += ["Index build cost:", "", markdown_table(cost), ""]
        if efficiency.best_per_variant is not None:
            best = efficiency.best_per_variant[["variant", "model", "validation_mrr_at_10", "median_ms", "mean_ms"]]
            out += ["Query time of the best-validation configuration per (representation, model):", "",
                    markdown_table(best), ""]
        out += [f"- {n}" for n in efficiency.notes] + [""]

    out += ["## How to read this report", "",
            "- Numbers from synthetic or demo data say nothing about real retrieval quality.",
            "- A p-value is the chance of seeing a difference this large if the systems were really equal; "
            "it is not the chance that the systems are equal.",
            "- With one relevant paper per query, Recall@K is a yes/no score per query, so Wilcoxon p-values "
            "on it are only approximate; MRR and nDCG are better suited.", ""]
    return "\n".join(out)
