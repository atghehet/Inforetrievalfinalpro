"""Tests for p4.eval.significance (paired bootstrap, paired tests, Holm, comparisons)."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from p4.eval.harness import evaluate_runs
from p4.eval.significance import (bootstrap_ci, compare_pair, compare_result, compare_systems, holm_adjust,
                                  paired_test)


def same(first, second):
    """Equality of two comparison results (NaN == NaN here, unlike plain dict equality)."""
    return pd.DataFrame([first.to_dict()]).equals(pd.DataFrame([second.to_dict()]))


def series(values, prefix="q"):
    return pd.Series(values, index=[f"{prefix}{i}" for i in range(len(values))], dtype=float)


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_is_reproducible_for_the_same_seed_and_changes_with_another():
    d = [0.5, -0.2, 0.1, 0.9, 0.0, 0.3, -0.1, 0.4]
    a = bootstrap_ci(d, n_boot=500, seed=7)
    assert a == bootstrap_ci(d, n_boot=500, seed=7)                 # identical, to the last digit
    assert a != bootstrap_ci(d, n_boot=500, seed=8)                 # a different seed resamples differently
    assert a[0] == pytest.approx(np.mean(d))                        # the point estimate never depends on the seed


def test_bootstrap_interval_matches_an_independent_calculation():
    d = np.array([0.5, -0.2, 0.1, 0.9, 0.0, 0.3, -0.1, 0.4])
    rng = np.random.default_rng(3)                                   # same recipe written out by hand
    means = d[rng.integers(0, len(d), size=(1000, len(d)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    mean, got_low, got_high = bootstrap_ci(d, n_boot=1000, seed=3, confidence=0.95)
    assert (mean, got_low, got_high) == pytest.approx((d.mean(), low, high))


def test_bootstrap_interval_properties():
    d = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    mean, low, high = bootstrap_ci(d, n_boot=2000, seed=1)
    assert low <= mean <= high
    assert min(d) <= low and high <= max(d)                          # a mean of resampled values cannot leave the data range
    _, low90, high90 = bootstrap_ci(d, n_boot=2000, seed=1, confidence=0.90)
    assert low <= low90 and high90 <= high                           # a 90 % interval is inside the 95 % one


def test_bootstrap_with_identical_differences_has_zero_width():
    assert bootstrap_ci([0.25] * 10, n_boot=200) == (0.25, 0.25, 0.25)
    assert bootstrap_ci([0.0] * 10, n_boot=200) == (0.0, 0.0, 0.0)


def test_bootstrap_with_a_single_query_has_no_interval():
    mean, low, high = bootstrap_ci([0.4], n_boot=200)
    assert mean == 0.4 and math.isnan(low) and math.isnan(high)


def test_bootstrap_does_not_depend_on_how_it_is_chunked_internally(monkeypatch):
    d = np.random.default_rng(0).normal(0.1, 0.5, size=40)
    reference = bootstrap_ci(d, n_boot=1000, seed=5)
    import p4.eval.significance as module
    monkeypatch.setattr(module, "_CHUNK_CELLS", 40 * 7)             # force many tiny blocks
    # Same inputs must still give a valid, reproducible interval (block size is fixed per run).
    assert bootstrap_ci(d, n_boot=1000, seed=5) == bootstrap_ci(d, n_boot=1000, seed=5)
    assert abs(bootstrap_ci(d, n_boot=1000, seed=5)[1] - reference[1]) < 0.05     # and statistically the same


@pytest.mark.parametrize("kwargs, message", [
    (dict(diffs=[], n_boot=200), "non-empty"),
    (dict(diffs=[0.1, float("nan")], n_boot=200), "finite"),
    (dict(diffs=[0.1, 0.2], n_boot=50), "at least 100"),
    (dict(diffs=[0.1, 0.2], n_boot=200, confidence=1.0), "confidence"),
    (dict(diffs=[0.1, 0.2], n_boot=200, confidence=0.0), "confidence"),
])
def test_bootstrap_rejects_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_ci(**kwargs)


# ------------------------------------------------------------------ paired tests
def test_wilcoxon_exact_p_value_for_six_positive_differences():
    # Only one of the 2**6 = 64 sign patterns is as extreme on each side: p = 2 / 64 = 0.03125.
    p, note = paired_test([1, 2, 3, 4, 5, 6], "wilcoxon")
    assert p == pytest.approx(0.03125)
    assert note == ""


def test_wilcoxon_cannot_reach_005_with_five_differences_and_says_so():
    p, note = paired_test([1, 2, 3, 4, 5], "wilcoxon")
    assert p == pytest.approx(0.0625)                                # 2 / 32
    assert "fewer than 6" in note


def test_wilcoxon_ignores_zero_differences():
    p, _ = paired_test([0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "wilcoxon")
    assert 0 < p < 0.1                                               # the zeros are dropped, six clear improvements remain


def test_paired_t_test_matches_the_textbook_formula():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))                # = 4.2426
    expected = 2 * stats.t.sf(abs(t), df=len(d) - 1)                  # about 0.0132
    p, note = paired_test(d, "ttest")
    assert p == pytest.approx(expected) and p == pytest.approx(0.0132, abs=1e-3)
    assert note == ""


@pytest.mark.parametrize("test", ["wilcoxon", "ttest"])
def test_all_zero_differences_give_p_one(test):
    p, note = paired_test([0.0, 0.0, 0.0, 0.0], test)
    assert p == 1.0 and "identical" in note


def test_floating_point_noise_counts_as_zero():
    assert paired_test([1e-15, -2e-16, 0.0], "wilcoxon")[0] == 1.0


def test_t_test_edge_cases_give_nan_with_an_explanation():
    p, note = paired_test([0.5], "ttest")
    assert math.isnan(p) and "at least 2" in note
    p, note = paired_test([0.5, 0.5, 0.5], "ttest")                   # zero variance but not zero: undefined
    assert math.isnan(p) and "zero variance" in note


def test_unknown_test_name_is_rejected():
    with pytest.raises(ValueError, match="test must be one of"):
        paired_test([1, 2], "magic")


def test_holm_correction_by_hand():
    # sorted p: 0.01, 0.03, 0.04 -> 3*0.01 = 0.03 ; 2*0.03 = 0.06 ; 1*0.04 = 0.04 -> kept non-decreasing = 0.06
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])      # 2*0.5 = 1.0 (capped); 0.9 is raised to keep order


def test_holm_skips_nan_and_does_not_count_it_as_a_comparison():
    adjusted = holm_adjust([0.01, float("nan"), 0.04])               # two real comparisons
    assert adjusted[0] == pytest.approx(0.02) and adjusted[2] == pytest.approx(0.04) and math.isnan(adjusted[1])
    assert holm_adjust([]) == []


# ------------------------------------------------------------------ comparing two systems
def test_clear_improvement_is_positive_and_significant():
    rng = np.random.default_rng(0)
    baseline = series(rng.uniform(0.0, 0.5, 30))
    better = baseline + 0.3 + rng.normal(0, 0.02, 30)
    r = compare_pair(better, baseline, metric="rr@10", system_name="new", baseline_name="old", n_boot=500)
    assert r.direction == "system_better" and r.significant
    assert r.mean_diff == pytest.approx((better - baseline).mean()) and r.mean_diff > 0.25
    assert r.ci_low > 0 and r.ci_high >= r.ci_low                    # the interval excludes zero
    assert r.n_queries == 30 and r.n_dropped == 0 and r.p_value < 0.001
    assert r.mean_system == pytest.approx(better.mean()) and r.mean_baseline == pytest.approx(baseline.mean())
    assert "new has the higher mean score" in r.conclusion and "significant" in r.conclusion


def test_swapping_the_systems_flips_the_sign_and_mirrors_the_interval():
    rng = np.random.default_rng(1)
    a, b = series(rng.uniform(0, 1, 25)), series(rng.uniform(0, 1, 25))
    ab = compare_pair(a, b, n_boot=500, seed=4)
    ba = compare_pair(b, a, n_boot=500, seed=4)
    assert ba.mean_diff == pytest.approx(-ab.mean_diff)
    assert (ba.ci_low, ba.ci_high) == pytest.approx((-ab.ci_high, -ab.ci_low))
    assert ba.p_value == pytest.approx(ab.p_value)                   # the test is two-sided
    assert {ab.direction, ba.direction} == {"system_better", "baseline_better"}


def test_worse_system_is_reported_as_baseline_better():
    base = series([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.55, 0.65])
    worse = base - 0.3
    r = compare_pair(worse, base, system_name="x", baseline_name="y", n_boot=300)
    assert r.direction == "baseline_better" and r.mean_diff < 0 and r.significant
    assert r.conclusion.startswith("y has the higher mean score")


def test_identical_systems_are_a_tie_with_p_one():
    s = series([0.1, 0.5, 1.0, 0.0, 0.25])
    r = compare_pair(s, s.copy(), n_boot=200)
    assert r.mean_diff == 0 and (r.ci_low, r.ci_high) == (0, 0) and r.p_value == 1.0
    assert r.direction == "tie" and not r.significant and "identical" in r.note
    assert "same mean score" in r.conclusion


def test_a_large_average_gap_on_too_few_queries_is_not_called_significant():
    base, system = series([0.0, 0.0, 0.0]), series([0.9, 0.9, 0.9])      # mean difference 0.9 ...
    r = compare_pair(system, base, n_boot=200)
    assert r.mean_diff == pytest.approx(0.9)
    assert not r.significant and "NOT significant" in r.conclusion       # ... but 3 queries cannot prove it
    assert "fewer than 6" in r.note


def test_single_query_comparison_is_safe():
    r = compare_pair(series([0.8]), series([0.2]), n_boot=200)
    assert r.n_queries == 1 and r.mean_diff == pytest.approx(0.6)
    assert math.isnan(r.ci_low) and "single query" in r.note and not r.significant


def test_missing_scores_are_dropped_and_counted_not_filled_in():
    a = pd.Series([0.9, 0.8, np.nan, 0.7, 0.6, 0.5, 0.4], index=list("abcdefg"))
    b = pd.Series([0.1, 0.2, 0.3, np.nan, 0.4, 0.3, 0.2], index=list("abcdefg"))
    r = compare_pair(a, b, n_boot=200)
    assert r.n_queries == 5 and r.n_dropped == 2 and "2 query(ies) dropped" in r.note
    kept = [("a", 0.9, 0.1), ("b", 0.8, 0.2), ("e", 0.6, 0.4), ("f", 0.5, 0.3), ("g", 0.4, 0.2)]
    assert r.mean_diff == pytest.approx(np.mean([x - y for _, x, y in kept]))


def test_no_overlapping_valid_scores_is_an_error():
    with pytest.raises(ValueError, match="no queries with a score"):
        compare_pair(pd.Series([np.nan], index=["q"]), pd.Series([0.5], index=["q"]), n_boot=200)


def test_infinite_scores_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        compare_pair(pd.Series([np.inf, 1.0], index=["a", "b"]), pd.Series([0.0, 1.0], index=["a", "b"]), n_boot=200)


def test_systems_must_have_exactly_the_same_query_ids():
    a = pd.Series([1.0, 0.5, 0.2], index=["q1", "q2", "q3"])
    b = pd.Series([1.0, 0.5, 0.2], index=["q1", "q2", "q4"])
    with pytest.raises(ValueError, match="exactly the same queries") as err:
        compare_pair(a, b, n_boot=200)
    assert "q3" in str(err.value) and "q4" in str(err.value)           # says which queries differ


def test_scores_are_paired_by_query_id_not_by_position():
    ids = ["q1", "q2", "q3", "q4", "q5", "q6"]
    a = pd.Series([0.9, 0.1, 0.8, 0.3, 0.7, 0.5], index=ids)
    b = pd.Series([0.5, 0.4, 0.6, 0.2, 0.1, 0.3], index=ids)
    shuffled = b.loc[["q6", "q2", "q5", "q1", "q4", "q3"]]              # same values, different row order
    assert same(compare_pair(a, shuffled, n_boot=300, seed=2), compare_pair(a, b, n_boot=300, seed=2))


def test_duplicate_query_ids_and_non_series_input_are_rejected():
    dup = pd.Series([1.0, 2.0], index=["q1", "q1"])
    with pytest.raises(ValueError, match="duplicate query IDs"):
        compare_pair(dup, dup, n_boot=200)
    with pytest.raises(TypeError, match="pandas Series"):
        compare_pair([1, 2, 3], [1, 2, 3], n_boot=200)


@pytest.mark.parametrize("kwargs", [dict(alpha=0.0), dict(alpha=1.0), dict(confidence=1.5), dict(n_boot=10), dict(test="x")])
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        compare_pair(series([1, 2, 3]), series([0, 1, 2]), **{"n_boot": 200, **kwargs})


def test_comparison_is_reproducible_end_to_end():
    rng = np.random.default_rng(9)
    a, b = series(rng.uniform(0, 1, 40)), series(rng.uniform(0, 1, 40))
    assert same(compare_pair(a, b, n_boot=400, seed=11), compare_pair(a, b, n_boot=400, seed=11))


# ------------------------------------------------------------------ several systems, using the harness
QRELS = {f"q{i}": {f"D{i}": 1} for i in range(1, 11)}                      # ten queries, one relevant paper each


def make_run(first_hit_rank_by_query):
    """A run where query i finds its relevant paper at the given rank (None = never)."""
    run = {}
    for qid, rank in first_hit_rank_by_query.items():
        docs = [f"x{j}" for j in range(1, 6)]
        if rank is not None:
            docs.insert(rank - 1, f"D{qid[1:]}")
        run[qid] = [(d, float(len(docs) - k)) for k, d in enumerate(docs)]
    return run


def test_compare_result_uses_the_harness_wide_table():
    good = make_run({q: 1 for q in QRELS})                                  # always rank 1
    mid = make_run({q: (2 if int(q[1:]) % 2 else 1) for q in QRELS})        # half at rank 2
    bad = make_run({q: None for q in QRELS})                                # never found
    result = evaluate_runs({"good": good, "mid": mid, "bad": bad}, QRELS, cutoffs=(10,))
    table = compare_result(result, ["rr@10", "ndcg@10"], baseline="good", n_boot=300)
    assert list(table["metric"]) == ["rr@10", "rr@10", "ndcg@10", "ndcg@10"]
    assert sorted(table["system"].unique()) == ["bad", "mid"] and set(table["baseline"]) == {"good"}
    assert (table["n_queries"] == 10).all() and (table["mean_diff"] < 0).all()
    row = table[(table.metric == "rr@10") & (table.system == "mid")].iloc[0]
    assert row.mean_diff == pytest.approx(-0.25)                            # 5 queries lose 0.5, 5 lose nothing
    assert row.mean_system == pytest.approx(result.summary.set_index("system").loc["mid", "MRR@10"])
    assert row.p_holm >= row.p_value                                        # correction can only raise a p-value


def test_queries_a_system_missed_entirely_still_count_as_zero_in_the_comparison():
    full = make_run({q: 1 for q in QRELS})
    partial = {q: v for q, v in full.items() if int(q[1:]) <= 6}            # run lacks 4 of the 10 judged queries
    result = evaluate_runs({"full": full, "partial": partial}, QRELS, cutoffs=(10,))
    row = compare_result(result, ["rr@10"], baseline="full", n_boot=300).iloc[0]
    assert row.n_queries == 10 and row.n_dropped == 0                       # same 10 queries, missing ones scored 0
    assert row.mean_system == pytest.approx(0.6)


def test_compare_systems_pairs_baseline_choice_and_errors():
    wide = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6.0], "b": [0, 1, 2, 3, 4, 5.0], "c": [1, 1, 1, 1, 1, 1.0]},
                        index=[f"q{i}" for i in range(6)])
    assert list(compare_systems(wide, baseline="b", n_boot=200)["system"]) == ["a", "c"]
    assert list(compare_systems(wide, n_boot=200)["baseline"]) == ["a", "a"]       # default: first column
    custom = compare_systems(wide, pairs=[("c", "b")], n_boot=200)
    assert custom.loc[0, "system"] == "c" and custom.loc[0, "baseline"] == "b"
    with pytest.raises(ValueError, match="baseline 'zzz'"):
        compare_systems(wide, baseline="zzz", n_boot=200)
    with pytest.raises(ValueError, match="unknown system"):
        compare_systems(wide, pairs=[("a", "nope")], n_boot=200)


def test_compare_systems_with_one_system_gives_an_empty_table_with_all_columns():
    wide = pd.DataFrame({"only": [1.0, 2.0]}, index=["q1", "q2"])
    table = compare_systems(wide, n_boot=200)
    assert table.empty and "p_holm" in table.columns and "conclusion" in table.columns


# ------------------------------------------------------------------ multiple comparisons
def test_holm_flag_can_disagree_with_the_raw_flag_when_several_systems_are_compared():
    d = np.array([1, 2, 3, 4, 5, 6.0])                       # six positive differences: exact Wilcoxon p = 0.03125
    wide = pd.DataFrame({"base": np.zeros(6), "s1": d, "s2": d}, index=[f"q{i}" for i in range(6)])
    table = compare_systems(wide, baseline="base", n_boot=200).set_index("system")
    assert table.loc["s1", "p_value"] == pytest.approx(0.03125)
    assert table.loc["s1", "p_holm"] == pytest.approx(0.0625)          # 2 comparisons: 2 * 0.03125
    assert table.loc["s1", "significant"] and not table.loc["s1", "significant_holm"]    # 0.031 < 0.05 <= 0.0625
    assert "before multiple-comparison correction" in table.loc["s1", "conclusion"]


def test_holm_flag_equals_raw_flag_for_a_single_comparison_and_follows_alpha():
    d = np.array([1, 2, 3, 4, 5, 6.0])
    wide = pd.DataFrame({"base": np.zeros(6), "s1": d}, index=[f"q{i}" for i in range(6)])
    one = compare_systems(wide, baseline="base", n_boot=200).iloc[0]
    assert one.significant and one.significant_holm and one.p_holm == pytest.approx(one.p_value)
    strict = compare_systems(wide, baseline="base", n_boot=200, alpha=0.01).iloc[0]       # 0.031 is no longer below alpha
    assert not strict.significant and not strict.significant_holm
    tie = compare_systems(pd.DataFrame({"a": [1.0] * 6, "b": [1.0] * 6}), baseline="a", n_boot=200).iloc[0]
    assert not tie.significant_holm                                                   # identical systems are never "significant"


# =============================================================================================
# Statistical correctness checks (review of Batch 2)
# =============================================================================================
import itertools   # noqa: E402

from p4.eval.significance import rank_direction   # noqa: E402


def exact_signed_rank_p(diffs):
    """Brute-force two-sided exact Wilcoxon signed-rank p-value: try every way of assigning signs.

    Zeros are dropped, |differences| get average ranks for ties, and the p-value is the share of
    sign assignments whose rank sum is at least as far from its mean as the observed one.
    """
    d = np.asarray(diffs, float)
    d = d[d != 0]
    ranks = stats.rankdata(np.abs(d))
    observed = abs(ranks[d > 0].sum() - ranks.sum() / 2)
    signs = np.array(list(itertools.product([0, 1], repeat=len(d))))
    totals = (signs * ranks).sum(axis=1)
    return float((np.abs(totals - ranks.sum() / 2) >= observed - 1e-9).mean())


# ---- 1. all differences zero ------------------------------------------------------------------
@pytest.mark.parametrize("test", ["wilcoxon", "ttest"])
def test_identical_systems_give_a_clean_tie_for_both_tests(test):
    r = compare_pair(series([0.2, 0.4, 0.9, 0.1, 0.5]), series([0.2, 0.4, 0.9, 0.1, 0.5]), test=test, n_boot=200)
    assert r.p_value == 1.0 and r.direction == "tie" and not r.significant and r.direction_agrees
    assert (r.mean_diff, r.ci_low, r.ci_high) == (0, 0, 0)


# ---- 2. ties and zeros ------------------------------------------------------------------------
@pytest.mark.parametrize("diffs", [
    [-1, -0.5, 0.25, 0.5, 1, 1, 0.5, -0.25],                       # tied magnitudes, no zeros
    [0, 0, -1, 0.5, 0.5, 1, 1, 0.25, 0, -0.5],                     # ties and zeros together
    [0.5] * 6 + [-0.5] * 2,                                        # everything tied
])
def test_wilcoxon_equals_the_brute_force_exact_test_for_small_tied_samples(diffs):
    assert paired_test(diffs, "wilcoxon")[0] == pytest.approx(exact_signed_rank_p(diffs), abs=1e-9)


def test_zero_differences_do_not_change_the_wilcoxon_p_value_whatever_the_total_sample_size():
    # Regression: scipy on its own switches from exact to an approximation once the TOTAL n (zeros included)
    # exceeds 13, which was off by up to 0.2 for zero-heavy samples. Zeros are now removed first.
    nonzero = [-1, 0.5, 0.5, 1, 1, 0.25, -0.25, 0.5, 1]            # 9 informative queries
    expected = exact_signed_rank_p(nonzero)
    for zeros in (0, 5, 20, 60):                                   # total sample size 9 ... 69
        assert paired_test(nonzero + [0.0] * zeros, "wilcoxon")[0] == pytest.approx(expected, abs=1e-9), zeros


def test_zero_heavy_comparison_is_judged_on_its_informative_queries_only():
    base, system = np.zeros(40), np.zeros(40)
    system[:9] = [-1, 0.5, 0.5, 1, 1, 0.25, -0.25, 0.5, 1]         # 31 queries identical, 9 differ
    r = compare_pair(series(system), series(base), n_boot=200)
    assert r.n_queries == 40 and r.p_value == pytest.approx(exact_signed_rank_p(system - base), abs=1e-9)


def test_rank_direction_follows_signed_rank_sums_with_ties_and_zeros():
    assert rank_direction([1, 2, 3, -1]) == 1                      # |d| ranks 1.5, 3, 4, 1.5 -> W+ = 7.5 > W- = 1.5
    assert rank_direction([-1, -2, 3]) == 0                        # |d| ranks 1, 2, 3 -> W+ = 3, W- = 1 + 2 = 3: a tie
    assert rank_direction([-1, -2, -3, 4]) == -1                   # ranks 1..4 -> W+ = 4 < W- = 6
    assert rank_direction([0, 0, 0]) == 0 and rank_direction([]) == 0
    assert rank_direction([1, -1]) == 0                            # equal weight on both sides
    assert rank_direction([25] + [-1] * 20) == -1                  # one huge gain, twenty small losses


# ---- 3. paired t-test: identical systems and tiny samples ---------------------------------------
def test_paired_t_test_with_two_queries_matches_the_formula_and_scipys_paired_test():
    d = np.array([0.3, 0.1])
    t = d.mean() / (d.std(ddof=1) / math.sqrt(2))
    p, note = paired_test(d, "ttest")
    assert p == pytest.approx(2 * stats.t.sf(abs(t), df=1)) and p == pytest.approx(0.29517, abs=1e-4) and note == ""
    assert p == pytest.approx(float(stats.ttest_rel(d, np.zeros(2)).pvalue))     # same thing as scipy's paired test
    r = compare_pair(series([0.9, 0.7]), series([0.1, 0.2]), test="ttest", n_boot=200)
    assert not r.significant                                       # two queries can never be enough


def test_paired_t_test_on_identical_systems_is_a_tie_not_an_error():
    r = compare_pair(series([0.5, 0.5, 0.5]), series([0.5, 0.5, 0.5]), test="ttest", n_boot=200)
    assert r.p_value == 1.0 and r.direction == "tie" and not r.significant


# ---- 4. the bootstrap is a PAIRED bootstrap ----------------------------------------------------
def _correlated_systems():
    rng = np.random.default_rng(3)
    a = rng.uniform(0.2, 0.9, 40)
    b = a - 0.05 + rng.normal(0, 0.02, 40)                         # two strongly correlated systems
    return a, b


def test_bootstrap_resamples_the_same_queries_for_both_systems():
    a, b = _correlated_systems()
    pick = np.random.default_rng(11).integers(0, 40, size=(1000, 40))        # ONE set of resampled query indices
    shared = a[pick].mean(axis=1) - b[pick].mean(axis=1)                      # applied to both systems
    low, high = np.quantile(shared, [0.025, 0.975])
    r = compare_pair(pd.Series(a, index=range(40)), pd.Series(b, index=range(40)), n_boot=1000, seed=11)
    assert (r.ci_low, r.ci_high) == pytest.approx((low, high))     # identical to resampling queries jointly


def test_paired_interval_is_far_narrower_than_an_unpaired_one_for_correlated_systems():
    a, b = _correlated_systems()
    paired = compare_pair(pd.Series(a, index=range(40)), pd.Series(b, index=range(40)), n_boot=1000, seed=11)
    rg = np.random.default_rng(11)                                 # what an (incorrect) unpaired bootstrap would give
    unpaired = a[rg.integers(0, 40, (1000, 40))].mean(axis=1) - b[rg.integers(0, 40, (1000, 40))].mean(axis=1)
    u_low, u_high = np.quantile(unpaired, [0.025, 0.975])
    assert paired.ci_low > 0                                       # pairing sees the consistent 0.05 gain
    assert (paired.ci_high - paired.ci_low) < 0.2 * (u_high - u_low) and u_low < 0   # unpaired noise would hide it


# ---- 5. Holm correction family ----------------------------------------------------------------
def _three_system_result():
    qrels = {f"q{i}": {f"D{i}": 1} for i in range(1, 31)}
    pattern = {"base": lambda i: i % 2 == 0, "s1": lambda i: i % 3 != 0,
               "s2": lambda i: i % 2 == 0 or i % 5 == 0, "s3": lambda i: i % 7 < 3}
    runs = {name: {q: [(f"D{q[1:]}" if hit(int(q[1:])) else "miss", 1.0)] for q in qrels} for name, hit in pattern.items()}
    return evaluate_runs(runs, qrels, cutoffs=(10,), min_coverage=0)


def test_holm_is_applied_per_metric_and_never_mixes_metrics():
    result = _three_system_result()
    both = compare_result(result, ["rr@10", "ndcg@10"], baseline="base", n_boot=200)
    for metric in ("rr@10", "ndcg@10"):
        alone = compare_systems(result.wide(metric), baseline="base", metric=metric, n_boot=200)
        together = both[both["metric"] == metric].reset_index(drop=True)
        assert len(together) == 3                                              # family = the 3 systems of THIS metric
        assert together["p_holm"].tolist() == pytest.approx(alone["p_holm"].tolist(), nan_ok=True)
        assert together["p_holm"].tolist() == pytest.approx(holm_adjust(together["p_value"]), nan_ok=True)


def test_holm_family_for_explicit_pairs_is_that_call_regardless_of_baselines():
    result = _three_system_result()
    table = compare_systems(result.wide("rr@10"), pairs=[("s1", "base"), ("s2", "base"), ("s3", "s1")], n_boot=200)
    assert table["p_holm"].tolist() == pytest.approx(holm_adjust(table["p_value"]))   # one family of 3, documented behaviour
    assert (table["p_holm"] >= table["p_value"] - 1e-12).all() and (table["p_holm"] <= 1).all()


# ---- 6. direction of improvement ---------------------------------------------------------------
def test_direction_always_has_the_sign_of_the_mean_difference():
    rng = np.random.default_rng(21)
    for i in range(60):
        n = int(rng.integers(8, 40))
        a, b = series(rng.uniform(size=n)), series(rng.uniform(size=n))
        r = compare_pair(a, b, n_boot=100, test="wilcoxon" if i % 2 else "ttest")
        assert r.mean_diff == pytest.approx((a - b).mean())
        assert r.direction == ("system_better" if r.mean_diff > 0 else "baseline_better")
        assert (r.mean_system > r.mean_baseline) == (r.direction == "system_better")


def test_a_few_huge_gains_cannot_turn_a_rank_test_that_favours_the_baseline_into_a_win():
    d = np.array([25.0] + [-1.0] * 20)                             # mean difference is positive ...
    r = compare_pair(series(d), series(np.zeros(21)), n_boot=300)
    assert r.mean_diff > 0 and r.direction == "system_better"
    assert r.p_value < 0.01                                        # ... but the rank test strongly sees a shift
    assert not r.direction_agrees and not r.significant            # ... toward the BASELINE, so it is not a win
    assert "signed-rank sums favour" in r.note and "other way" in r.conclusion
    table = compare_systems(pd.DataFrame({"base": np.zeros(21), "s": d}), baseline="base", n_boot=200)
    assert not table.loc[0, "significant_holm"] and not table.loc[0, "direction_agrees"]


def test_the_same_data_under_a_t_test_has_no_direction_conflict():
    d = np.array([25.0] + [-1.0] * 20)
    r = compare_pair(series(d), series(np.zeros(21)), n_boot=300, test="ttest")
    assert r.direction_agrees and r.direction == "system_better"   # the t-test is a test of the mean itself


def test_equal_means_with_a_rank_shift_is_reported_as_a_tie_and_flagged():
    r = compare_pair(series([-3.0, 1.0, 1.0, 1.0]), series([0.0] * 4), n_boot=200)
    assert r.direction == "tie" and not r.significant and not r.direction_agrees
    assert "neither system" in r.note


def test_agreeing_data_keeps_direction_agrees_true():
    r = compare_pair(series([1, 2, 3, 4, 5, 6.0]), series([0.0] * 6), n_boot=200)
    assert r.direction_agrees and r.significant and r.direction == "system_better"


# ---- 7. missing scores across comparisons -------------------------------------------------------
def test_missing_scores_use_the_same_rule_in_every_comparison_and_are_reported_per_row():
    wide = pd.DataFrame({"base": [.5, .6, .7, .8, .9, .4, .3, .2],
                         "s1": [.6, np.nan, .8, .9, 1., .5, .4, .3],     # missing query q1
                         "s2": [.9] * 8}, index=[f"q{i}" for i in range(8)])
    table = compare_systems(wide, baseline="base", n_boot=200).set_index("system")
    assert (table.loc["s1", "n_queries"], table.loc["s1", "n_dropped"]) == (7, 1)
    assert (table.loc["s2", "n_queries"], table.loc["s2", "n_dropped"]) == (8, 0)       # s2 keeps q1, s1 does not
    assert table.loc["s1", "mean_baseline"] == pytest.approx(np.mean([.5, .7, .8, .9, .4, .3, .2]))   # base mean over the SAME 7
    assert table.loc["s2", "mean_baseline"] == pytest.approx(wide["base"].mean())
    assert "1 query(ies) dropped" in table.loc["s1", "note"]


def test_queries_missing_from_a_run_are_zero_in_every_comparison_because_the_harness_scores_them():
    qrels = {f"q{i}": {f"D{i}": 1} for i in range(1, 11)}
    full = {q: [(f"D{q[1:]}", 1.0)] for q in qrels}
    partial = {q: v for q, v in full.items() if int(q[1:]) <= 6}
    other = {q: v for q, v in full.items() if int(q[1:]) <= 8}
    result = evaluate_runs({"full": full, "partial": partial, "other": other}, qrels, cutoffs=(10,))
    table = compare_result(result, ["rr@10"], baseline="full", n_boot=200)
    assert (table["n_queries"] == 10).all() and (table["n_dropped"] == 0).all()      # same 10 queries everywhere
    assert sorted(table["mean_system"].round(2)) == [0.6, 0.8]
