"""The five analyses. Each takes a prepared `AnalysisContext` and returns an `AnalysisOutput` (tables + Markdown report).

    run_popularity   what popular papers do to retrieval (query side trend, retrieval side bias)
    run_coldstart    performance for papers with few citations / few indexable contexts, expanded vs abstract-only
    run_year         performance by publication year of the relevant paper
    run_function     performance by citation function of the query sentence
    run_errors       query-level failures, missed papers, differences between systems

Rules shared by all of them
  * A data label (real / synthetic / demo) is required. It is declared by the user and NOT verified. It is written into
    every CSV (column `data_label`) and at the top of every report. Synthetic and demo outputs are never findings.
  * Input paths, dataset statistics and the analysis date are written into every report.
  * Systems are compared only when their runs are MATCHED: the same judged queries (qrels), all systems evaluated by the
    same harness call with the same cutoffs, and every system returned results for every judged query. Otherwise the
    comparisons are skipped and the report says why. The code cannot check that two systems differ ONLY in the document
    representation (for example P2 tunes BM25 separately for each representation); that is a limitation to state.
  * Missing metadata is reported, small groups are shown but excluded from inference, and wording stays descriptive.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from p4.eval.harness import DEFAULT_CUTOFFS, EvalResult, evaluate_runs, summary_name
from p4.eval.report import markdown_table
from p4.eval.significance import DEFAULT_ALPHA, DEFAULT_CONFIDENCE, DEFAULT_N_BOOT, DEFAULT_SEED

from . import errors as err_mod
from . import groups as G
from .data import AnalysisDataError, build_query_table, check_data_label, dataset_stats
from .popularity import retrieval_popularity


@dataclass
class Options:
    k: int = 10
    cutoffs: tuple = DEFAULT_CUTOFFS
    min_group_size: int = G.DEFAULT_MIN_GROUP_SIZE
    n_boot: int = DEFAULT_N_BOOT
    seed: int = DEFAULT_SEED
    alpha: float = DEFAULT_ALPHA
    confidence: float = DEFAULT_CONFIDENCE
    test: str = "wilcoxon"
    baseline: str | None = None
    year_bins: list | None = None
    min_confidence: float | None = None
    min_coverage: float = 0.5
    top_n: int = 20
    analysis_date: str | None = None
    note: str | None = None


@dataclass
class AnalysisContext:
    data_label: str
    runs: dict
    qrels: dict
    query_table: pd.DataFrame
    papers: pd.DataFrame | None
    result: EvalResult
    matched: bool
    unmatched: list
    options: Options
    baseline: str
    stats: dict
    input_paths: dict
    have_functions: bool
    have_index_counts: bool

    @property
    def metrics(self) -> list[str]:
        return [f"rr@{self.options.k}", f"ndcg@{self.options.k}", f"recall@{self.options.k}"]


@dataclass
class AnalysisOutput:
    name: str
    tables: dict = field(default_factory=dict)
    report: str = ""


def prepare_context(*, data_label, runs, qrels, queries=None, papers=None, functions=None, index_counts=None,
                    options: Options | None = None, input_paths: dict | None = None) -> AnalysisContext:
    """Evaluate every run with the P4 harness and join the per-query metadata. Nothing is substituted for missing inputs."""
    options = options or Options()
    check_data_label(data_label)
    if options.k not in options.cutoffs:
        options.cutoffs = tuple(sorted(set(options.cutoffs) | {options.k}))
    if not runs:
        raise AnalysisDataError("no runs given")
    result = evaluate_runs(runs, qrels, options.cutoffs, min_coverage=options.min_coverage)
    unmatched = [s for s, w in result.warnings.items() if w["missing"]]
    baseline = options.baseline or next(iter(runs))
    if baseline not in runs:
        raise AnalysisDataError(f"baseline {baseline!r} is not one of the systems {list(runs)}")
    table = build_query_table(qrels, queries, papers, functions, index_counts)
    stats = dataset_stats(table, papers, have_queries=queries is not None, have_functions=functions is not None,
                          have_index_counts=index_counts is not None)
    return AnalysisContext(data_label, runs, qrels, table, papers, result, not unmatched, unmatched, options, baseline,
                           stats, dict(input_paths or {}), functions is not None, index_counts is not None)


# ------------------------------------------------------------------------------------------ helpers
def _tag(tables: dict, label: str) -> dict:
    out = {}
    for name, df in tables.items():
        df = df.copy()
        df.insert(0, "data_label", label)
        out[name] = df
    return out


def _require(ctx: AnalysisContext, column: str, what: str) -> None:
    if ctx.papers is None or ctx.query_table[column].isna().all():
        raise AnalysisDataError(f"this analysis needs {what}, but it is missing or empty in the inputs; "
                                "nothing was substituted")


def _banner(label: str) -> list[str]:
    if label == "real":
        return ["> **Data label: REAL** (declared by the user, not verified by this tool).", ""]
    return [f"> **Data label: {label.upper()} - these numbers are NOT real findings.** They only demonstrate that the "
            "analysis code works. Do not quote them as results.", ""]


def _header(title: str, ctx: AnalysisContext) -> list[str]:
    o = ctx.options
    date = o.analysis_date or _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    lines = [f"# {title}", ""] + _banner(ctx.data_label)
    if o.note:
        lines += [f"> **Note:** {o.note}", ""]
    lines += ["## Metadata", "", f"- **analysis date (UTC):** {date}", f"- **data label:** {ctx.data_label} (user-declared)"]
    for name, path in ctx.input_paths.items():
        lines.append(f"- **{name}:** {path}")
    lines.append(f"- **systems:** {', '.join(ctx.runs)} (baseline: {ctx.baseline})")
    lines.append(f"- **settings:** k={o.k}, minimum group size={o.min_group_size}, bootstrap resamples={o.n_boot}, "
                 f"seed={o.seed}, alpha={o.alpha}, paired test={o.test}")
    lines += ["", "### Dataset statistics", ""] + [f"- {k}: {v}" for k, v in ctx.stats.items()]
    if ctx.matched:
        lines += ["", "- **Matched runs:** yes - every system returned results for every judged query; all systems were "
                  "evaluated together with the same qrels and cutoffs."]
    else:
        lines += ["", f"- **Matched runs: NO.** These systems returned nothing for some judged queries: {ctx.unmatched}. "
                  "System comparisons are skipped; per-system group summaries still count those queries as 0."]
    return lines + [""]


def _whole(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Show counts, years and ranks as whole numbers in a report table (the CSV files keep the raw values)."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(lambda v: "n/a" if pd.isna(v) else str(int(v)))
    return df


def _fmt(value, digits=4):
    return "n/a" if value != value else f"{value:.{digits}f}"


def format_groups(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Wide, readable table: one row per (system, group) with 'mean [low, high]' for each metric."""
    rows = []
    for (system, group), part in summary.groupby(["system", "group"], sort=False):
        row = {"system": system, "group": group, "n_queries": int(part["n_queries"].iloc[0])}
        row["small"] = "yes (n too small to interpret)" if part["small_group"].iloc[0] else ""
        for metric in metrics:
            r = part[part["metric"] == metric].iloc[0]
            interval = "" if r["ci_low"] != r["ci_low"] else f" [{_fmt(r['ci_low'], 3)}, {_fmt(r['ci_high'], 3)}]"
            row[summary_name(metric)] = "n/a" if r["mean"] != r["mean"] else f"{_fmt(r['mean'], 3)}{interval}"
        rows.append(row)
    return pd.DataFrame(rows)


def format_comparisons(comp: pd.DataFrame, metric: str) -> pd.DataFrame:
    part = comp[comp["metric"] == metric]
    return pd.DataFrame({"system": part["system"], "baseline": part["baseline"], "group": part["group"],
                         "n": part["n_queries"], "mean_diff": part["mean_diff"],
                         "CI low": part["ci_low"], "CI high": part["ci_high"], "p": part["p_value"].map(lambda v: _fmt(v, 4)),
                         "p_holm": part["p_holm"].map(lambda v: _fmt(v, 4)), "sig. (Holm)": part["significant_holm"]})


def _comparison_block(ctx: AnalysisContext, per_query, groups, order, metrics) -> pd.DataFrame:
    o = ctx.options
    if not ctx.matched or len(ctx.runs) < 2:
        return pd.DataFrame(columns=["system", "baseline", "group", "metric", "n_queries", "mean_system", "mean_baseline",
                                     "mean_diff", "ci_low", "ci_high", "p_value", "p_holm", "significant_holm", "direction",
                                     "direction_agrees", "note"])
    return G.compare_within_groups(per_query, groups, order, metrics, baseline=ctx.baseline,
                                   min_group_size=o.min_group_size, n_boot=o.n_boot, seed=o.seed, alpha=o.alpha, test=o.test)


def _group_tables(ctx: AnalysisContext, name: str, groups, order, *, x=None, x_name=None, ordinal=True) -> dict:
    o, pq, metrics = ctx.options, ctx.result.per_query, ctx.metrics
    tables = {f"{name}_groups": G.summarise_groups(pq, groups, order, metrics, min_group_size=o.min_group_size,
                                                    n_boot=o.n_boot, seed=o.seed, confidence=o.confidence),
              f"{name}_comparisons": _comparison_block(ctx, pq, groups, order, metrics)}
    if ordinal and x is not None:
        tables[f"{name}_trend"] = G.trend_tests(pq, x, groups, order, metrics, x_name=x_name, min_group_size=o.min_group_size,
                                                n_boot=o.n_boot, seed=o.seed, confidence=o.confidence, alpha=o.alpha)
    elif not ordinal:
        tables[f"{name}_group_tests"] = G.group_difference_tests(pq, groups, order, metrics,
                                                                  min_group_size=o.min_group_size, alpha=o.alpha)
    return tables


def _group_report(ctx: AnalysisContext, title: str, name: str, tables: dict, sizes: pd.DataFrame | None = None) -> list[str]:
    metrics = ctx.metrics
    lines = [f"## {title}", ""]
    if sizes is not None:
        lines += ["Queries per group (a group with fewer than "
                  f"{ctx.options.min_group_size} queries is small: shown, but excluded from comparisons and trend tests):", "",
                  markdown_table(sizes), ""]
    lines += ["Mean score per group, with a bootstrap interval where the group is large enough:", "",
              markdown_table(format_groups(tables[f"{name}_groups"], metrics)), ""]
    comp = tables[f"{name}_comparisons"]
    if comp.empty:
        reason = ("systems are not matched" if not ctx.matched else "only one system" if len(ctx.runs) < 2
                  else "no group reaches the minimum size")
        lines += [f"System comparisons within groups: not computed ({reason}).", ""]
    else:
        lines += [f"System comparisons within groups (paired, same queries; each system minus {ctx.baseline}):", ""]
        for metric in metrics:
            lines += [f"**{summary_name(metric)}**", "", markdown_table(format_comparisons(comp, metric)), ""]
    return lines


def _trend_report(trend: pd.DataFrame, what: str) -> list[str]:
    show = trend[["system", "metric", "n_queries_used", "n_queries_excluded", "n_groups_used", "spearman_rho", "ci_low",
                  "ci_high", "p_value", "p_holm", "supported", "note"]].copy()
    show["metric"] = show["metric"].map(summary_name)
    return [f"Trend: Spearman correlation between {what} and the per-query score (only queries in non-small groups; "
            "an association, not a cause):", "", markdown_table(show), ""]


def _caveats(*items: str) -> list[str]:
    return ["## Limitations", ""] + [f"- {i}" for i in items] + [
        "- Groups are defined on the relevant paper; several properties (popularity, age, number of indexable contexts) "
        "go together, so a difference between groups cannot be attributed to any one of them.",
        "- Small samples give wide intervals; absence of a significant difference is not evidence of no difference.", ""]


def _finish(ctx: AnalysisContext, name: str, tables: dict, lines: list[str]) -> AnalysisOutput:
    return AnalysisOutput(name, _tag(tables, ctx.data_label), "\n".join(lines))


# ------------------------------------------------------------------------------------------ analyses
def run_popularity(ctx: AnalysisContext) -> AnalysisOutput:
    """Popularity bias: query side (score vs the relevant paper's popularity) and retrieval side (what is returned)."""
    _require(ctx, "citation_count", "papers.citation_count for the relevant papers")
    if "citation_count" not in ctx.papers.columns:
        raise AnalysisDataError("papers table has no citation_count column")
    o = ctx.options
    groups, order = G.citation_groups(ctx.query_table["citation_count"])
    tables = _group_tables(ctx, "popularity_query_side", groups, order, x=ctx.query_table["citation_count"],
                           x_name="papers.citation_count")
    summary, per_query, ref = retrieval_popularity(ctx.runs, ctx.qrels, ctx.papers, k=o.k, n_boot=o.n_boot, seed=o.seed,
                                                   confidence=o.confidence)
    tables["popularity_retrieval"] = summary
    tables["popularity_retrieval_per_query"] = per_query
    sizes = G.group_sizes(groups, order, o.min_group_size)
    lines = _header("Popularity bias analysis", ctx)
    lines += ["Popularity = `papers.citation_count`: citing papers inside the corpus, including the query's own citing paper. "
              "`queries.cited_citation_count` is the same number copied onto the query row (supplementary; compared with "
              "papers.citation_count in the dataset statistics above).", ""]
    lines += _group_report(ctx, "1. Query side: does the score change with the relevant paper's popularity?",
                           "popularity_query_side", tables, sizes)
    lines += _trend_report(tables["popularity_query_side_trend"], "the relevant paper's citation count")
    lines += ["## 2. Retrieval side: how popular are the papers each system returns?", "",
              f"Corpus reference: {ref['n_papers']} papers, median {ref['median_citations']:.1f} citations, the top 10% start at "
              f"{ref['threshold']:.0f} citations and make up {ref['top_decile_share']:.3f} of the corpus; "
              f"{ref['n_zero_citation']} papers ({ref['zero_citation_share']:.3f}) have 0 citations.", ""]
    show = summary[["system", "n_queries", "n_docs_topk", "n_docs_unknown_citation_count", "mean_log10_citations_topk",
                    "median_citations_topk", "top_decile_share_topk", "top_decile_lift", "zero_citation_share_topk",
                    "relevant_mean_log10_citations"]]
    lines += [markdown_table(show), "",
              "`top_decile_lift` = share of returned documents in the most-cited 10% divided by that group's share of the corpus "
              "(1 = no preference). Papers with 0 citations are never query targets, so they are visible only here, as returned "
              "distractors.", "",
              "When the top-1 result is wrong, how often is it more or less cited than the relevant paper?", "",
              markdown_table(summary[["system", "n_wrong_top1", "wrong_top1_more_cited_rate", "wrong_top1_more_cited_ci_low",
                                      "wrong_top1_more_cited_ci_high", "wrong_top1_less_cited_rate"]]), ""]
    lines += _caveats("The relevant papers are not a random sample of the corpus (each has at least one citation), so the "
                      "corpus shares are reference points, not a neutral expectation.",
                      "Popular papers have more citing sentences, so a context-expanded index gives them more text; a link "
                      "between popularity and score may reflect that and not popularity itself.",
                      "Citation counts are corpus-internal (cs.CL papers citing cs.CL papers), not global citation counts.")
    return _finish(ctx, "popularity", tables, lines)


def run_coldstart(ctx: AnalysisContext) -> AnalysisOutput:
    """Cold start: performance for papers with few incoming citations or few indexable contexts."""
    _require(ctx, "citation_count", "papers.citation_count for the relevant papers")
    o = ctx.options
    groups, order = G.citation_groups(ctx.query_table["citation_count"])
    tables = _group_tables(ctx, "coldstart_citation", groups, order, x=ctx.query_table["citation_count"],
                           x_name="papers.citation_count")
    corpus_groups, _ = G.citation_groups(ctx.papers["citation_count"])
    corpus_counts = corpus_groups.value_counts()
    sizes = G.group_sizes(groups, order, o.min_group_size).rename(columns={"n_queries": "n_query_targets"})
    sizes.insert(1, "n_papers_in_corpus", [int(corpus_counts.get(g, 0)) for g in order])
    tables["coldstart_corpus_buckets"] = sizes
    lines = _header("Cold-start analysis", ctx)
    zero_papers = int(corpus_counts.get("0", 0))
    zero_targets = int((groups == "0").sum())
    lines += ["## Zero-citation papers", "",
              f"{zero_papers} papers in the corpus have `citation_count` 0 and {zero_targets} judged queries have such a paper as "
              "their relevant paper. A paper can only be a query target if some paper cites it, and `citation_count` includes "
              "that citing paper, so **zero-citation target papers cannot be evaluated** from this data. They can appear in "
              "rankings only as distractors (see the popularity analysis, `zero_citation_share_topk`).", ""]
    lines += _group_report(ctx, "1. By citation count of the relevant paper (papers.citation_count)", "coldstart_citation",
                           tables, sizes)
    lines += _trend_report(tables["coldstart_citation_trend"], "the relevant paper's citation count")
    if ctx.have_index_counts:
        cgroups, corder = G.context_groups(ctx.query_table["n_index_contexts"])
        tables.update(_group_tables(ctx, "coldstart_contexts", cgroups, corder, x=ctx.query_table["n_index_contexts"],
                                    x_name="indexable contexts of the relevant paper"))
        lines += _group_report(ctx, "2. By number of indexable contexts of the relevant paper (the expansion text actually "
                               "available)", "coldstart_contexts", tables, G.group_sizes(cgroups, corder, o.min_group_size))
        lines += _trend_report(tables["coldstart_contexts_trend"], "the number of indexable contexts")
    else:
        lines += ["## 2. By number of indexable contexts", "",
                  "Not computed: no `--index-contexts` file was given, and nothing was substituted.", ""]
    lines += ["## Expanded versus abstract-only", "",
              "Compare a context-expanded system with an abstract-only system by naming them in `--run` and choosing the "
              "abstract-only system as `--baseline`. The comparison above is valid only for matched runs (see Metadata). "
              "With P2's frozen test runs the two representations are tuned separately (`best.json`), so the pair differs in "
              "tuned parameters as well as in representation; a strictly matched comparison needs P2's validation logs.", ""]
    lines += _caveats("Cold start is judged on papers that are query targets, so every one has at least one citation; the "
                      "truly uncited papers are invisible as targets.",
                      "`citation_count` counts the query's own citing paper, so the lowest bucket is 1-2 citations in total, "
                      "not 'no other citations'.")
    return _finish(ctx, "coldstart", tables, lines)


def run_year(ctx: AnalysisContext) -> AnalysisOutput:
    """Publication-year analysis (year of the relevant paper), plus citation age as supplementary groups."""
    _require(ctx, "year", "papers.year for the relevant papers")
    o = ctx.options
    groups, order = G.year_groups(ctx.query_table["year"], o.year_bins)
    tables = _group_tables(ctx, "year", groups, order, x=ctx.query_table["year"], x_name="publication year")
    sizes = G.group_sizes(groups, order, o.min_group_size)
    lines = _header("Publication-year analysis", ctx)
    lines += _group_report(ctx, "1. By publication year of the relevant paper", "year", tables, sizes)
    lines += _trend_report(tables["year_trend"], "the relevant paper's publication year")
    pop, pop_order = G.citation_groups(ctx.query_table["citation_count"])
    cross = pd.crosstab(groups.rename("year group"), pop.rename("citation bucket")).reindex(
        index=order, columns=[c for c in pop_order if c in set(pop)], fill_value=0)
    tables["year_by_popularity_crosstab"] = cross.reset_index()
    lines += ["## Year and popularity overlap", "",
              "Number of queries per year group and citation bucket. Older papers have had longer to collect citations, so "
              "year and popularity are entangled; this table shows how much:", "",
              markdown_table(cross.reset_index()), ""]
    if ctx.query_table["citation_age"].notna().any():
        agroups, aorder = G.age_groups(ctx.query_table["citation_age"])
        tables.update(_group_tables(ctx, "citation_age", agroups, aorder))
        lines += _group_report(ctx, "2. Supplementary: citation age (citing year minus relevant paper's year)", "citation_age",
                               tables, G.group_sizes(agroups, aorder, o.min_group_size))
    lines += _caveats("Year groups of the relevant paper say nothing about when the query sentence was written; see the "
                      "citation-age groups.",
                      "The corpus covers a limited date range; years with few papers produce small groups.")
    return _finish(ctx, "year", tables, lines)


def run_function(ctx: AnalysisContext) -> AnalysisOutput:
    """Citation-function analysis: groups defined by the label of the QUERY sentence (P1's context_functions)."""
    if not ctx.have_functions or ctx.query_table["function"].isna().all():
        raise AnalysisDataError("the citation-function analysis needs P1's context_functions.parquet joined to the queries "
                                "(by context_id); no labels are available, and none were invented")
    o = ctx.options
    groups, order = G.function_groups(ctx.query_table["function"], ctx.query_table["function_confidence"], o.min_confidence)
    tables = _group_tables(ctx, "function", groups, order, ordinal=False)
    lines = _header("Citation-function analysis", ctx)
    lines += ["The label describes the query sentence: whether it cites the paper as background, as a method used, or for a "
              "result. Labels are predictions from P1's classifier (trained on SciCite, a different corpus), not human "
              "annotations; accuracy on this corpus is unknown.", ""]
    lines += _group_report(ctx, "1. By citation function of the query sentence", "function", tables,
                           G.group_sizes(groups, order, o.min_group_size))
    tests = tables["function_group_tests"].copy()
    tests["metric"] = tests["metric"].map(summary_name)
    lines += ["Kruskal-Wallis test across the non-small labelled groups (do scores differ between functions?):", "",
              markdown_table(tests), ""]
    lines += _caveats("Function labels are machine-predicted; label noise blurs differences between groups.",
                      "Queries are sentences that cite exactly one corpus paper, which may over-represent some functions.")
    return _finish(ctx, "function", tables, lines)


def run_errors(ctx: AnalysisContext) -> AnalysisOutput:
    """Error analysis: failures, missed papers, differences between systems."""
    o, k = ctx.options, ctx.options.k
    ranks = err_mod.relevant_ranks(ctx.runs, ctx.qrels, k)
    summary = err_mod.error_summary(ranks, k)
    tables = {"errors_summary": summary}
    qt = ctx.query_table
    wide = qt[["target_id", "text", "title", "year", "citation_count", "function"]].copy()
    for system in ctx.runs:
        part = ranks[ranks["system"] == system].set_index("query_id")
        wide[f"rank__{system}"] = part["rank"]
        wide[f"top1__{system}"] = part["top1_id"]
    tables["errors_per_query"] = wide.reset_index()
    pair = err_mod.pairwise_outcomes(ranks, ctx.baseline, o.alpha) if len(ctx.runs) > 1 else pd.DataFrame()
    if len(ctx.runs) > 1:
        tables["errors_pairwise"] = pair
    hard = err_mod.hard_queries(ranks, qt, o.top_n)
    tables["errors_hard_queries"] = hard
    tables["errors_hubs"] = err_mod.hub_papers(ranks, ctx.papers)
    missed = err_mod.missed_papers(ranks, qt)
    tables["errors_missed_papers"] = missed
    miss = ranks.assign(**{f"miss@{k}": (~ranks["hit"]).astype(float)})
    blocks = []
    for label, (groups, order) in (("citation bucket", G.citation_groups(qt["citation_count"])),
                                   ("year", G.year_groups(qt["year"], o.year_bins)),
                                   ("citation function", G.function_groups(qt["function"], qt["function_confidence"],
                                                                           o.min_confidence))):
        if label == "citation function" and not ctx.have_functions:
            continue
        if label != "citation function" and qt[{"citation bucket": "citation_count", "year": "year"}[label]].isna().all():
            continue
        s = G.summarise_groups(miss, groups, order, [f"miss@{k}"], min_group_size=o.min_group_size, n_boot=o.n_boot,
                               seed=o.seed, confidence=o.confidence)
        s.insert(0, "grouping", label)
        blocks.append(s)
    if blocks:
        tables["errors_failure_by_group"] = pd.concat(blocks, ignore_index=True)
    lines = _header("Error analysis", ctx)
    lines += [f"A **failure** means the relevant paper is not in a system's top {k}. `rank` is the position of the relevant "
              "paper in the run (not retrieved = not in the run at all; runs have different depths, so deep ranks are not "
              "equally informative across systems).", "", "## 1. Failures per system", "", markdown_table(summary), ""]
    if len(ctx.runs) > 1:
        lines += [f"## 2. Differences between systems (each against {ctx.baseline})", "",
                  "Counts of queries where the two systems disagree on hit/miss, with an exact binomial test on those "
                  "discordant queries (p_holm corrects across systems):", "", markdown_table(pair), ""]
    shown = _whole(hard[["query_id", "n_systems_failing", "failed_by_all", "best_rank_any_system", "title",
                         "citation_count", "year"]], ["best_rank_any_system", "citation_count", "year"])
    lines += [f"## 3. Hardest queries (top {o.top_n}: failed by the most systems)", "", markdown_table(shown), "",
              "Query texts, the systems that failed, and ranks of the relevant paper are in `errors_hard_queries.csv` and "
              "`errors_per_query.csv`.", ""]
    lines += [f"## 4. Relevant papers that were missed ({len(missed)} papers missed by at least one system)", "",
              markdown_table(_whole(missed.head(o.top_n)[["target_id", "n_queries", "n_pairs_missed", "n_pairs",
                                                          "missed_in_every_pair", "title", "citation_count"]],
                                    ["citation_count"])) if len(missed) else
              "No relevant paper was missed.", ""]
    lines += ["## 5. Hub papers (wrong papers most often returned at rank 1)", "",
              markdown_table(_whole(tables["errors_hubs"], ["citation_count", "year"])) if len(tables["errors_hubs"])
              else "No wrong top-1 results.", ""]
    if "errors_failure_by_group" in tables:
        fg = tables["errors_failure_by_group"].copy()
        fg = fg[fg["n_queries"] > 0]
        lines += [f"## 6. Failure rate by group (miss@{k}; small groups have no interval)", "", markdown_table(fg), ""]
    lines += _caveats("Failures are described, not explained; reading the query texts and relevant papers is needed to judge "
                      "why a system failed.",
                      "With one relevant paper per query, other papers that would also satisfy the query count as failures, so "
                      "failure rates overstate how often a system returns nothing useful.")
    return _finish(ctx, "errors", tables, lines)


ANALYSES = {"popularity": run_popularity, "coldstart": run_coldstart, "year": run_year, "function": run_function,
            "errors": run_errors}
