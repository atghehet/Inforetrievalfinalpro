"""Tests for p4.analysis.groups: group definitions, per-group summaries, paired comparisons, trends."""
import math

import numpy as np
import pandas as pd
import pytest

from p4.analysis import groups as G
from p4.eval.harness import evaluate_runs
from p4.tests import analysis_fixtures as F


@pytest.fixture(scope="module")
def per_query():
    return evaluate_runs(F.make_runs(), F.make_qrels(), cutoffs=(10,), min_coverage=0).per_query


@pytest.fixture(scope="module")
def table():
    from p4.analysis.data import build_query_table
    return build_query_table(F.make_qrels(), F.make_queries(), F.make_papers(), F.make_functions(), F.make_index_counts())


# ---- group definitions ----------------------------------------------------------------------------
def test_citation_buckets_follow_p1_boundaries_and_unknown():
    values = pd.Series([0, 1, 2, 3, 10, 11, 50, 51, np.nan], index=list("abcdefghi"))
    groups, order = G.citation_groups(values)
    assert groups.tolist() == ["0", "1-2", "1-2", "3-10", "3-10", "11-50", "11-50", ">50", "unknown"]
    assert order == ["0", "1-2", "3-10", "11-50", ">50", "unknown"]


def test_empty_buckets_stay_in_the_order_and_unknown_appears_only_when_needed():
    groups, order = G.citation_groups(pd.Series([5, 6]))
    assert order == ["0", "1-2", "3-10", "11-50", ">50"] and set(groups) == {"3-10"}


def test_context_and_age_buckets():
    assert G.context_groups(pd.Series([0, 1, 2, 5, 6, 20, 21, 100, 101]))[0].tolist() == \
        ["0", "1", "2-5", "2-5", "6-20", "6-20", "21-100", "21-100", ">100"]
    assert G.age_groups(pd.Series([-2, 0, 1, 2, 3, 4, 5, 6]))[0].tolist() == ["<0", "0-1", "0-1", "2-3", "2-3", "4-5", "4-5", ">5"]


def test_year_groups_default_to_one_group_per_year_with_unknown_last():
    groups, order = G.year_groups(pd.Series([2020.0, 2018.0, np.nan, 2020.0]))
    assert groups.tolist() == ["2020", "2018", "unknown", "2020"] and order == ["2018", "2020", "unknown"]


def test_year_bins_make_coarser_groups():
    groups, order = G.year_groups(pd.Series([2017, 2019, 2020, 2021, 2022, 2025, np.nan]), bins=[2019, 2021])
    assert groups.tolist() == ["<=2019", "<=2019", "2020-2021", "2020-2021", ">=2022", ">=2022", "unknown"]
    assert order == ["<=2019", "2020-2021", ">=2022", "unknown"]
    for bad in ([], [2019, 2019]):
        with pytest.raises(ValueError, match="distinct whole numbers"):
            G.year_groups(pd.Series([2019]), bins=bad)


def test_function_groups_with_missing_labels_and_low_confidence():
    function = pd.Series(["method", None, "result", "background"])
    groups, order = G.function_groups(function)
    assert groups.tolist() == ["method", "unlabelled", "result", "background"] and order == ["background", "method", "result", "unlabelled"]
    groups, order = G.function_groups(function, pd.Series([0.9, np.nan, 0.4, 0.8]), min_confidence=0.5)
    assert groups.tolist() == ["method", "unlabelled", "unlabelled (low confidence)", "background"]
    assert "unlabelled (low confidence)" in order


# ---- summaries ---------------------------------------------------------------------------------------
def test_group_means_match_the_hand_calculation(per_query, table):
    groups, order = G.citation_groups(table["citation_count"])
    s = G.summarise_groups(per_query, groups, order, ["rr@10"], n_boot=200)
    get = lambda system, group, col: s[(s.system == system) & (s.group == group)][col].iloc[0]
    assert get("weak", "3-10", "mean") == pytest.approx(5 / 20) and get("weak", ">50", "mean") == pytest.approx(15 / 20)
    assert get("good", "3-10", "mean") == 1.0 and get("good", ">50", "n_queries") == 20
    assert not get("weak", "3-10", "small_group")
    lo, hi = get("weak", "3-10", "ci_low"), get("weak", "3-10", "ci_high")
    assert lo <= 0.25 <= hi and 0 < lo and hi < 0.6                              # a sensible interval around 5/20


def test_empty_and_small_groups_are_reported_without_intervals(per_query, table):
    groups, order = G.citation_groups(table["citation_count"])
    s = G.summarise_groups(per_query, groups, order, ["rr@10"], min_group_size=25, n_boot=200)     # nobody reaches 25
    assert s["small_group"].all() and s["ci_low"].isna().all()
    assert s[s.group == "3-10"]["mean"].notna().all()                              # means are still shown
    empty = s[(s.group == "0") & (s.system == "weak")].iloc[0]
    assert empty["n_queries"] == 0 and math.isnan(empty["mean"])                   # the empty bucket is visible, not silently absent


def test_invalid_min_group_size_is_rejected(per_query, table):
    groups, order = G.citation_groups(table["citation_count"])
    with pytest.raises(ValueError, match="min_group_size"):
        G.summarise_groups(per_query, groups, order, ["rr@10"], min_group_size=0)


def test_group_sizes_table(table):
    groups, order = G.citation_groups(table["citation_count"])
    sizes = G.group_sizes(groups, order, 10).set_index("group")
    assert sizes.loc["3-10", "n_queries"] == 20 and sizes.loc[">50", "n_queries"] == 20 and sizes.loc["0", "n_queries"] == 0
    assert not sizes.loc["3-10", "small_group"] and sizes.loc["0", "small_group"]


# ---- paired comparisons within groups ----------------------------------------------------------------
def test_comparisons_within_groups_have_the_hand_calculated_differences(per_query, table):
    groups, order = G.citation_groups(table["citation_count"])
    c = G.compare_within_groups(per_query, groups, order, ["rr@10", "recall@10"], baseline="good", n_boot=200)
    one = c[(c.metric == "rr@10")].set_index("group")
    assert set(one.index) == {"3-10", ">50"}                                       # small groups (0, 1-2, 11-50) are skipped
    assert one.loc["3-10", "mean_diff"] == pytest.approx(-0.75) and one.loc[">50", "mean_diff"] == pytest.approx(-0.25)
    assert one.loc["3-10", "n_queries"] == 20 and (one["direction"] == "baseline_better").all()
    assert one.loc["3-10", "significant_holm"]                                      # 15 of 20 queries lose a full point
    assert not one.loc[">50", "significant_holm"]                                   # only 5 queries differ: p cannot go below 0.0625
    assert one.loc[">50", "p_value"] == pytest.approx(0.0625) and "fewer than 6" in one.loc[">50", "note"]
    assert (one["p_holm"] >= one["p_value"]).all()


def test_holm_is_applied_per_metric_over_the_comparisons_of_the_call(per_query, table):
    groups, order = G.citation_groups(table["citation_count"])
    c = G.compare_within_groups(per_query, groups, order, ["rr@10", "recall@10"], baseline="good", n_boot=200)
    from p4.eval.significance import holm_adjust
    for metric, part in c.groupby("metric"):
        assert part["p_holm"].tolist() == pytest.approx(holm_adjust(part["p_value"]))
    assert len(c) == 4                                                              # 2 metrics x 2 groups x 1 system


def test_comparison_errors(per_query, table):
    groups, order = G.citation_groups(table["citation_count"])
    with pytest.raises(ValueError, match="baseline 'nope'"):
        G.compare_within_groups(per_query, groups, order, ["rr@10"], baseline="nope")
    none = G.compare_within_groups(per_query, groups, order, ["rr@10"], baseline="good", min_group_size=50, n_boot=200)
    assert none.empty and "p_holm" in none.columns                                  # no group reaches 50: nothing to compare


# ---- trends --------------------------------------------------------------------------------------------
def test_trend_uses_only_non_small_groups_and_reports_the_counts(per_query, table):
    groups, order = G.citation_groups(table["citation_count"])
    t = G.trend_tests(per_query, table["citation_count"], groups, order, ["rr@10"], x_name="citations", n_boot=200)
    weak = t[t.system == "weak"].iloc[0]
    assert weak["n_groups_used"] == 2 and weak["n_queries_used"] == 40 and weak["n_queries_excluded"] == 0
    from scipy import stats
    x = table["citation_count"].reindex(per_query.loc[per_query.system == "weak", "query_id"]).to_numpy(float)
    y = per_query.loc[per_query.system == "weak", "rr@10"].to_numpy(float)
    rho, p = stats.spearmanr(x, y)                                                   # independent computation on the same data
    assert weak["supported"] and weak["spearman_rho"] == pytest.approx(rho) and weak["p_value"] == pytest.approx(p)
    assert 0.2 < rho < 0.4                                                           # a modest positive association
    assert weak["ci_low"] <= weak["spearman_rho"] <= weak["ci_high"]
    good = t[t.system == "good"].iloc[0]
    assert not good["supported"] and "constant" in good["note"]                      # every query scores 1.0: nothing to correlate


def test_trend_excludes_small_groups_and_says_not_testable_with_too_few_groups(per_query, table):
    groups, order = G.citation_groups(table["citation_count"])
    t = G.trend_tests(per_query, table["citation_count"], groups, order, ["rr@10"], x_name="citations",
                      min_group_size=30, n_boot=200)                                 # no group reaches 30
    assert (~t["supported"]).all() and t["note"].str.contains("fewer than 2 groups").all()
    assert (t["n_queries_used"] == 0).all() and (t["n_queries_excluded"] == 40).all()


def test_trend_rho_matches_a_hand_calculation_on_a_tiny_monotone_case():
    pq = pd.DataFrame({"system": "s", "query_id": [f"q{i}" for i in range(12)], "rr@10": np.arange(12) / 11})
    groups = pd.Series(["low"] * 6 + ["high"] * 6, index=pq["query_id"])
    x = pd.Series(np.arange(12), index=pq["query_id"])
    t = G.trend_tests(pq, x, groups, ["low", "high"], ["rr@10"], x_name="x", min_group_size=5, n_boot=200).iloc[0]
    assert t["spearman_rho"] == pytest.approx(1.0) and t["n_queries_used"] == 12      # a perfectly increasing relationship


def test_trend_ignores_queries_with_unknown_x():
    pq = pd.DataFrame({"system": "s", "query_id": [f"q{i}" for i in range(14)], "rr@10": np.r_[np.arange(12) / 11, 0.5, 0.5]})
    groups = pd.Series(["low"] * 6 + ["high"] * 6 + ["low", "low"], index=pq["query_id"])
    x = pd.Series(np.r_[np.arange(12), np.nan, np.nan], index=pq["query_id"])
    t = G.trend_tests(pq, x, groups, ["low", "high"], ["rr@10"], x_name="x", min_group_size=5, n_boot=200).iloc[0]
    assert t["n_queries_used"] == 12 and t["n_queries_excluded"] == 2


def test_group_difference_test_across_functions(per_query, table):
    groups, order = G.function_groups(table["function"])
    t = G.group_difference_tests(per_query, groups, order, ["rr@10"])
    weak = t[t.system == "weak"].iloc[0]
    assert weak["n_groups_used"] == 3 and weak["n_queries_used"] == 40 and weak["supported"]
    assert 0 <= weak["p_value"] <= 1 and weak["h_statistic"] >= 0
    assert not t[t.system == "good"].iloc[0]["supported"]                            # constant scores
    small = G.group_difference_tests(per_query, groups, order, ["rr@10"], min_group_size=20)
    assert (~small["supported"]).all() and small["note"].str.contains("fewer than 2").all()


def test_a_trend_needs_two_groups_so_exactly_one_eligible_group_is_not_testable():
    n = 40
    pq = pd.DataFrame({"system": "s", "query_id": [f"q{i}" for i in range(n)], "rr@10": np.arange(n) / n})
    # one big group (20 queries) and three small ones (9, 9, 2): with min size 10 only "big" is eligible
    groups = pd.Series(["big"] * 20 + ["a"] * 9 + ["b"] * 9 + ["c"] * 2, index=pq["query_id"])
    x = pd.Series(np.arange(n), index=pq["query_id"])
    t = G.trend_tests(pq, x, groups, ["big", "a", "b", "c"], ["rr@10"], x_name="x", min_group_size=10, n_boot=200).iloc[0]
    assert t["n_groups_used"] == 1 and t["n_queries_used"] == 20 and t["n_queries_excluded"] == 20
    assert not t["supported"] and np.isnan(t["spearman_rho"]) and "fewer than 2 groups" in t["note"]
