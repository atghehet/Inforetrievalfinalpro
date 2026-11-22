"""Tests for p4.analysis.analyses: context preparation, the five analyses, labels, metadata and refusals."""
from functools import lru_cache

import pandas as pd
import pytest

from p4.analysis import analyses as A
from p4.analysis.data import AnalysisDataError
from p4.eval.harness import RunQrelsMismatchError
from p4.tests import analysis_fixtures as F

DATE = "2026-01-02"


def cached(name, **options):
    """Run an analysis once per distinct option set (each run does many paired tests, which are slow on tiny samples)."""
    return _cached(name, tuple(sorted(options.items(), key=lambda kv: kv[0])))


@lru_cache(maxsize=None)
def _cached(name, options):
    return A.ANALYSES[name](context(baseline="good", **{k: (list(v) if isinstance(v, tuple) else v) for k, v in options}))


def context(label="synthetic", *, runs=None, functions=True, index_counts=True, papers=True, queries=True, **options):
    opts = A.Options(n_boot=200, analysis_date=DATE, **options)
    return A.prepare_context(
        data_label=label, runs=runs or F.make_runs(), qrels=F.make_qrels(),
        queries=F.make_queries() if queries else None, papers=F.make_papers() if papers else None,
        functions=F.make_functions() if functions else None, index_counts=F.make_index_counts() if index_counts else None,
        options=opts, input_paths={"qrels": "qrels.txt", "papers": "papers.parquet"})


# ---- preparation --------------------------------------------------------------------------------------
def test_prepare_context_checks_the_label_and_evaluates_all_systems_together():
    ctx = context(baseline="good")
    assert ctx.matched and ctx.unmatched == [] and ctx.baseline == "good" and 10 in ctx.options.cutoffs
    assert list(ctx.result.summary["system"]) == ["good", "weak"] and ctx.metrics == ["rr@10", "ndcg@10", "recall@10"]
    with pytest.raises(AnalysisDataError, match="data label must be one of"):
        context("maybe")
    with pytest.raises(AnalysisDataError, match="no runs given"):
        A.prepare_context(data_label="demo", runs={}, qrels=F.make_qrels())
    with pytest.raises(AnalysisDataError, match="baseline 'nope'"):
        context(baseline="nope")


def test_runs_with_missing_queries_are_flagged_as_not_matched():
    partial = {q: v for q, v in F.make_runs()["good"].items() if int(q[1:]) <= 30}
    ctx = context(runs={"good": F.make_runs()["good"], "partial": partial})
    assert not ctx.matched and ctx.unmatched == ["partial"]


def test_a_run_for_the_wrong_split_is_rejected_by_the_existing_coverage_check():
    wrong = {f"v{i}": [("T01", 1.0)] for i in range(1, 6)}
    with pytest.raises(RunQrelsMismatchError, match="wrong split"):
        context(runs={"good": F.make_runs()["good"], "wrong": wrong})


# ---- data label, metadata, determinism ---------------------------------------------------------------
@pytest.mark.parametrize("label, banner", [("synthetic", "NOT real findings"), ("demo", "NOT real findings"),
                                           ("real", "declared by the user, not verified")])
def test_every_report_and_every_csv_carries_the_data_label(label, banner):
    ctx = context(label, baseline="good", min_group_size=50)         # no group reaches 50: no paired tests, label is the point
    for name, run in A.ANALYSES.items():
        out = run(ctx)
        assert banner in out.report and f"data label:** {label}" in out.report, name
        assert out.tables and all(t["data_label"].eq(label).all() and list(t.columns)[0] == "data_label"
                                  for t in out.tables.values()), name


def test_reports_contain_metadata_paths_statistics_date_and_settings():
    out = A.run_popularity(context(baseline="good", note="SYNTHETIC fixture", min_group_size=10))
    r = out.report
    assert "analysis date (UTC):** 2026-01-02" in r and "**qrels:** qrels.txt" in r and "**papers:** papers.parquet" in r
    assert "judged queries: 40" in r and "papers with citation_count 0: 5" in r and "queries where cited_citation_count != papers.citation_count: 0" in r
    assert "minimum group size=10" in r and "bootstrap resamples=200" in r and "seed=42" in r
    assert "Note:** SYNTHETIC fixture" in r and "Matched runs:** yes" in r


def test_default_analysis_date_is_today_in_utc_and_reports_are_otherwise_deterministic():
    import datetime
    ctx = A.prepare_context(data_label="demo", runs=F.make_runs(), qrels=F.make_qrels(), papers=F.make_papers(),
                            options=A.Options(n_boot=200))
    assert datetime.datetime.now(datetime.timezone.utc).date().isoformat() in A.run_popularity(ctx).report
    a, b = (A.run_popularity(context(baseline="good", min_group_size=15)) for _ in range(2))
    assert a.report == b.report
    for name in a.tables:
        pd.testing.assert_frame_equal(a.tables[name], b.tables[name])


# ---- popularity ----------------------------------------------------------------------------------------
def test_popularity_tables_and_query_side_results():
    out = cached("popularity")
    assert set(out.tables) == {"popularity_query_side_groups", "popularity_query_side_comparisons",
                               "popularity_query_side_trend", "popularity_retrieval", "popularity_retrieval_per_query"}
    groups = out.tables["popularity_query_side_groups"]
    weak = groups[(groups.system == "weak") & (groups.metric == "rr@10")].set_index("group")
    assert weak.loc["3-10", "mean"] == pytest.approx(0.25) and weak.loc[">50", "mean"] == pytest.approx(0.75)
    assert weak.loc["3-10", "n_queries"] == 20
    retrieval = out.tables["popularity_retrieval"].set_index("system")
    assert retrieval.loc["good", "zero_citation_share_topk"] == pytest.approx(0.4)
    assert "zero_citation_share_topk" in out.report and "Papers with 0 citations are never query targets" in out.report


# ---- cold start ----------------------------------------------------------------------------------------
def test_coldstart_states_that_zero_citation_targets_cannot_be_evaluated():
    out = cached("coldstart")
    assert "**zero-citation target papers cannot be evaluated**" in out.report
    assert "5 papers in the corpus have `citation_count` 0 and 0 judged queries" in out.report
    corpus = out.tables["coldstart_corpus_buckets"].set_index("group")
    assert corpus.loc["0", "n_papers_in_corpus"] == 5 and corpus.loc["0", "n_query_targets"] == 0
    assert corpus.loc["3-10", "n_query_targets"] == 20 and corpus.loc[">50", "n_papers_in_corpus"] == 20


def test_coldstart_second_axis_uses_indexable_contexts_when_given():
    out = cached("coldstart")
    ctx_groups = out.tables["coldstart_contexts_groups"]
    weak = ctx_groups[(ctx_groups.system == "weak") & (ctx_groups.metric == "rr@10")].set_index("group")
    assert weak.loc["1", "n_queries"] == 10 and weak.loc["6-20", "n_queries"] == 10 and weak.loc["21-100", "n_queries"] == 20
    assert weak.loc["1", "mean"] == pytest.approx(0.5)                                  # q01-q05 hit of q01-q10
    assert "indexable contexts" in out.report
    without = A.run_coldstart(context(baseline="good", index_counts=False, min_group_size=50))
    assert "coldstart_contexts_groups" not in without.tables
    assert "no `--index-contexts` file was given, and nothing was substituted" in without.report


def test_expanded_versus_abstract_comparison_needs_matched_runs():
    runs = F.make_runs()
    matched = cached("coldstart")
    assert not matched.tables["coldstart_citation_comparisons"].empty
    partial = {"good": runs["good"], "weak": {q: v for q, v in runs["weak"].items() if int(q[1:]) <= 30}}
    unmatched = A.run_coldstart(context(runs=partial, baseline="good"))
    assert unmatched.tables["coldstart_citation_comparisons"].empty                      # skipped, not computed on mismatched runs
    assert "Matched runs: NO" in unmatched.report and "systems are not matched" in unmatched.report
    assert A.run_coldstart(context(runs={"good": runs["good"]})).tables["coldstart_citation_comparisons"].empty


# ---- year ----------------------------------------------------------------------------------------------
def test_year_groups_bins_and_crosstab():
    out = cached("year")
    groups = out.tables["year_groups"]
    assert set(groups["group"]) == {"2019", "2020"} and set(groups["n_queries"]) == {20}
    cross = out.tables["year_by_popularity_crosstab"].set_index("year group")
    assert cross.loc["2019", "3-10"] == 10 and cross.loc["2019", ">50"] == 10 and cross.loc["2020", "3-10"] == 10
    binned = A.run_year(context(baseline="good", year_bins=[2019], min_group_size=50))
    assert set(binned.tables["year_groups"]["group"]) == {"<=2019", ">=2020"}


def test_year_analysis_requires_year_data_and_reports_unknown_years():
    with pytest.raises(AnalysisDataError, match="needs papers.year"):
        A.run_year(context(papers=False, baseline="good"))
    papers = F.make_papers().astype({"year": float})
    papers.loc[papers.paper_id.isin(["T01", "T02", "T03"]), "year"] = float("nan")
    ctx = A.prepare_context(data_label="demo", runs=F.make_runs(), qrels=F.make_qrels(), queries=F.make_queries(),
                            papers=papers, options=A.Options(n_boot=200, analysis_date=DATE, baseline="good"))
    out = A.run_year(ctx)
    g = out.tables["year_groups"]
    assert g[(g.group == "unknown") & (g.system == "good")]["n_queries"].iloc[0] == 3
    assert "queries with unknown target year: 3" in out.report


# ---- citation function ---------------------------------------------------------------------------------
def test_function_analysis_groups_by_label_and_refuses_without_labels():
    out = cached("function")
    g = out.tables["function_groups"]
    sizes = g[(g.system == "good") & (g.metric == "rr@10")].set_index("group")["n_queries"].to_dict()
    assert sizes == {"background": 13, "method": 14, "result": 13}
    assert "function_group_tests" in out.tables and "predictions from P1's classifier" in out.report
    with pytest.raises(AnalysisDataError, match="none were invented"):
        A.run_function(context(functions=False, baseline="good"))


def test_low_confidence_labels_can_be_set_aside():
    out = A.run_function(context(baseline="good", min_confidence=0.7, min_group_size=50))   # even-numbered queries have confidence 0.5
    g = out.tables["function_groups"]
    sizes = g[(g.system == "good") & (g.metric == "rr@10")].set_index("group")["n_queries"]
    assert sizes["unlabelled (low confidence)"] == 20 and sizes[["background", "method", "result"]].sum() == 20


# ---- errors --------------------------------------------------------------------------------------------
def test_error_analysis_tables_and_report():
    out = cached("errors")
    assert {"errors_summary", "errors_per_query", "errors_pairwise", "errors_hard_queries", "errors_hubs",
            "errors_missed_papers", "errors_failure_by_group"} <= set(out.tables)
    per_query = out.tables["errors_per_query"].set_index("query_id")
    assert per_query.loc["q06", "rank__weak"] == 13 and per_query.loc["q06", "text"] == "query sentence 6"
    fb = out.tables["errors_failure_by_group"]
    weak = fb[(fb.system == "weak") & (fb.grouping == "citation bucket")].set_index("group")
    assert weak.loc["3-10", "mean"] == pytest.approx(0.75) and weak.loc[">50", "mean"] == pytest.approx(0.25)   # failure rate = 1 - recall
    assert set(fb["grouping"]) == {"citation bucket", "year", "citation function"}
    for heading in ("1. Failures per system", "2. Differences between systems", "3. Hardest queries", "5. Hub papers",
                    "Limitations"):
        assert heading in out.report
    assert "Failures are described, not explained" in out.report


def test_error_analysis_without_optional_inputs_still_runs_and_skips_what_it_cannot_group():
    out = A.run_errors(context(papers=False, functions=False, queries=False, index_counts=False, baseline="good"))
    assert "errors_failure_by_group" not in out.tables                                  # nothing to group by, nothing invented
    assert out.tables["errors_summary"].set_index("system").loc["weak", "n_failures"] == 20
    single = A.run_errors(context(runs={"good": F.make_runs()["good"]}))
    assert "errors_pairwise" not in single.tables
