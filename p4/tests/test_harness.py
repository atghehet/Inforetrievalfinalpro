"""Tests for p4.eval.harness, including a regression test on the P2 demo data."""
import math
from pathlib import Path

import pytest

from p4.eval.harness import evaluate_run_files, evaluate_runs
from p4.eval.trec_io import read_qrels, read_run

HERE = Path(__file__).resolve().parent
DEMO_QRELS = HERE.parent.parent / "lexical" / "data" / "demo" / "validation.qrels"  # P2's demo data
FIXTURES = HERE / "fixtures"

# ---- hand-calculated example ------------------------------------------------
# Relevant documents:  q1 -> A,  q2 -> B,  q3 -> C
QRELS = {"q1": {"A": 1}, "q2": {"B": 1}, "q3": {"C": 1}}
# System "s1":  q1 finds A at rank 1, q2 finds B at rank 2, q3 returns nothing at all.
S1 = {"q1": [("A", 3.0)], "q2": [("X", 2.0), ("B", 1.0)]}
# System "s2":  q1 finds A at rank 3, q2 misses B, q3 finds C at rank 1. Also an extra query q9.
S2 = {"q1": [("X", 3.0), ("Y", 2.0), ("A", 1.0)], "q2": [("X", 2.0)], "q3": [("C", 5.0)],
      "q9": [("Z", 1.0)]}


def test_per_query_scores_match_hand_calculation():
    pq = evaluate_runs({"s1": S1, "s2": S2}, QRELS, cutoffs=(1, 2, 10)).per_query.set_index(["system", "query_id"])
    # s1, q1: rank 1 -> everything is 1
    assert pq.loc[("s1", "q1"), "rr@10"] == 1.0
    assert pq.loc[("s1", "q1"), "ndcg@10"] == 1.0
    assert pq.loc[("s1", "q1"), "ap"] == 1.0
    # s1, q2: B at rank 2
    assert pq.loc[("s1", "q2"), "recall@1"] == 0.0
    assert pq.loc[("s1", "q2"), "recall@2"] == 1.0
    assert pq.loc[("s1", "q2"), "rr@1"] == 0.0
    assert pq.loc[("s1", "q2"), "rr@10"] == 0.5
    assert pq.loc[("s1", "q2"), "ndcg@10"] == pytest.approx(1 / math.log2(3))
    assert pq.loc[("s1", "q2"), "ap"] == 0.5
    # s2, q1: A at rank 3
    assert pq.loc[("s2", "q1"), "recall@2"] == 0.0
    assert pq.loc[("s2", "q1"), "recall@10"] == 1.0
    assert pq.loc[("s2", "q1"), "ndcg@10"] == pytest.approx(0.5)


def test_query_with_no_results_scores_zero_and_is_counted():
    result = evaluate_runs({"s1": S1}, QRELS, cutoffs=(10,))
    row = result.per_query.set_index("query_id").loc["q3"]
    assert row["n_retrieved"] == 0
    assert row["recall@10"] == row["rr@10"] == row["ndcg@10"] == row["ap"] == 0.0
    assert result.summary.loc[0, "n_no_results"] == 1


def test_summary_is_the_mean_over_all_judged_queries():
    s = evaluate_runs({"s1": S1, "s2": S2}, QRELS, cutoffs=(1, 10)).summary.set_index("system")
    # s1: rr@10 = (1 + 0.5 + 0) / 3 = 0.5 ; recall@1 = (1 + 0 + 0) / 3 ; MAP = (1 + 0.5 + 0) / 3
    assert s.loc["s1", "MRR@10"] == pytest.approx(0.5)
    assert s.loc["s1", "Recall@1"] == pytest.approx(1 / 3)
    assert s.loc["s1", "MAP"] == pytest.approx(0.5)
    assert s.loc["s1", "nDCG@10"] == pytest.approx((1 + 1 / math.log2(3) + 0) / 3)
    # s2: rr@10 = (1/3 + 0 + 1) / 3 ; the extra query q9 is ignored
    assert s.loc["s2", "MRR@10"] == pytest.approx((1 / 3 + 0 + 1) / 3)
    assert s.loc["s2", "n_queries"] == 3
    assert list(s.index) == ["s1", "s2"]


def test_warnings_report_missing_and_extra_queries():
    w = evaluate_runs({"s1": S1, "s2": S2}, QRELS).warnings
    assert w["s1"] == {"missing": ["q3"], "extra": []}
    assert w["s2"] == {"missing": [], "extra": ["q9"]}


def test_multiple_cutoffs_produce_one_column_each():
    result = evaluate_runs({"s1": S1}, QRELS, cutoffs=[10, 1, 5, 5])   # unsorted, with a repeat
    assert result.metric_columns() == ["recall@1", "rr@1", "ndcg@1", "recall@5", "rr@5", "ndcg@5",
                                       "recall@10", "rr@10", "ndcg@10", "ap"]
    summary_cols = set(result.summary.columns)
    assert {"Recall@1", "Recall@5", "Recall@10", "MRR@5", "nDCG@1", "MAP"} <= summary_cols


@pytest.mark.parametrize("bad", [[], [0], [-5], [2.5]])
def test_invalid_cutoffs_rejected(bad):
    with pytest.raises(ValueError):
        evaluate_runs({"s1": S1}, QRELS, cutoffs=bad)


def test_wide_table_is_aligned_for_paired_tests():
    wide = evaluate_runs({"s1": S1, "s2": S2}, QRELS, cutoffs=(10,)).wide("rr@10")
    assert list(wide.index) == ["q1", "q2", "q3"]        # same queries for both systems, sorted
    assert list(wide.columns) == ["s1", "s2"]
    assert wide["s1"].tolist() == [1.0, 0.5, 0.0]
    assert wide["s2"].tolist() == pytest.approx([1 / 3, 0.0, 1.0])
    with pytest.raises(KeyError):
        evaluate_runs({"s1": S1}, QRELS).wide("nonsense")


def test_no_runs_or_no_qrels_rejected():
    with pytest.raises(ValueError):
        evaluate_runs({}, QRELS)
    with pytest.raises(ValueError):
        evaluate_runs({"s1": S1}, {})


# ---- regression test on P2's demo data --------------------------------------
# The three .run files in tests/fixtures are real outputs of P2's code
# (python -m p2.experiment --data data/demo --quick) on the demo validation
# queries v1 -> D1 and v2 -> D2. P2's own tuning.csv recorded these validation
# MRR@10 values, which our independent implementation must reproduce:
#   context bm25 = 1.0   (D1 and D2 both at rank 1)
#   context rm3  = 0.75  (v1: D1 at rank 2 -> 0.5 ; v2: D2 at rank 1 -> 1.0)
#   abstract bm25 = 0.0  (retrieved nothing)
def test_regression_reproduces_p2_demo_numbers():
    assert DEMO_QRELS.exists(), f"demo qrels missing: {DEMO_QRELS}"
    result = evaluate_run_files(
        {"context_bm25": FIXTURES / "p2_demo_context_bm25.run",
         "context_rm3": FIXTURES / "p2_demo_context_rm3.run",
         "abstract_bm25_empty": FIXTURES / "p2_demo_abstract_bm25_empty.run"},
        DEMO_QRELS)
    s = result.summary.set_index("system")
    assert s.loc["context_bm25", "MRR@10"] == pytest.approx(1.0)
    assert s.loc["context_rm3", "MRR@10"] == pytest.approx(0.75)
    assert s.loc["abstract_bm25_empty", "MRR@10"] == 0.0
    assert s.loc["abstract_bm25_empty", "n_no_results"] == 2
    assert s.loc["context_rm3", "Recall@1"] == pytest.approx(0.5)    # only v2 hits at rank 1
    assert s.loc["context_rm3", "Recall@10"] == pytest.approx(1.0)
    assert all(len(w["extra"]) == 0 for w in result.warnings.values())


def test_regression_matches_p3_style_calculation():
    """P3 computes MRR@10 and nDCG@10 as plain means of 1/rank and 1/log2(rank+1)."""
    qrels, run = read_qrels(DEMO_QRELS), read_run(FIXTURES / "p2_demo_context_rm3.run")
    ranks = {q: [d for d, _ in run[q]].index(next(iter(qrels[q]))) + 1 for q in qrels}   # v1 -> 2, v2 -> 1
    expected_mrr = sum(1 / r for r in ranks.values()) / len(ranks)
    expected_ndcg = sum(1 / math.log2(r + 1) for r in ranks.values()) / len(ranks)
    s = evaluate_runs({"x": run}, qrels).summary.iloc[0]
    assert s["MRR@10"] == pytest.approx(expected_mrr)
    assert s["nDCG@10"] == pytest.approx(expected_ndcg)


# ---- Batch 1: stable query order -------------------------------------------
from p4.eval.harness import RunQrelsMismatchError   # noqa: E402  (kept next to its tests)


def test_query_order_does_not_depend_on_qrels_order():
    shuffled = {"q3": QRELS["q3"], "q1": QRELS["q1"], "q2": QRELS["q2"]}      # NOT sorted
    a = evaluate_runs({"s1": S1, "s2": S2}, QRELS, cutoffs=(10,))
    b = evaluate_runs({"s1": S1, "s2": S2}, shuffled, cutoffs=(10,))
    assert b.per_query["query_id"].tolist() == ["q1", "q2", "q3", "q1", "q2", "q3"]   # sorted inside each system
    assert b.per_query.equals(a.per_query)            # identical table whatever the qrels order
    assert b.summary.equals(a.summary)
    assert list(b.wide("rr@10").index) == ["q1", "q2", "q3"]
    assert b.wide("rr@10").equals(a.wide("rr@10"))


def test_repeated_evaluation_is_deterministic():
    first = evaluate_runs({"s1": S1, "s2": S2}, QRELS).per_query
    second = evaluate_runs({"s2": S2, "s1": S1}, QRELS).per_query        # systems given in another order
    key = ["system", "query_id"]
    assert first.sort_values(key).reset_index(drop=True).equals(second.sort_values(key).reset_index(drop=True))


# ---- Batch 1: split mismatch must be loud -----------------------------------
VAL_QRELS = {"v1": {"A": 1}, "v2": {"B": 1}, "v3": {"C": 1}, "v4": {"D": 1}}
TEST_RUN = {"t1": [("A", 1.0)], "t2": [("B", 1.0)]}                        # a run for a DIFFERENT split


def test_run_from_another_split_raises_instead_of_reporting_zeros():
    with pytest.raises(RunQrelsMismatchError, match="wrong split") as err:
        evaluate_runs({"dense": TEST_RUN}, VAL_QRELS)
    msg = str(err.value)
    assert "'dense'" in msg and "0 of 4" in msg and "t1" in msg and "v1" in msg   # names system, counts, example IDs
    assert isinstance(err.value, ValueError)


def test_coverage_threshold_is_inclusive_at_the_boundary():
    one_of_four = {"v1": [("A", 1.0)]}                                       # 25 %
    two_of_four = {"v1": [("A", 1.0)], "v2": [("B", 1.0)]}                   # 50 % = default minimum
    with pytest.raises(RunQrelsMismatchError):
        evaluate_runs({"s": one_of_four}, VAL_QRELS)
    result = evaluate_runs({"s": two_of_four}, VAL_QRELS, cutoffs=(10,))      # exactly 50 %: allowed
    assert result.summary.loc[0, "n_no_results"] == 2                        # the other two score 0 and are counted


def test_custom_min_coverage_and_disabling_the_check():
    one_of_four = {"v1": [("A", 1.0)]}
    assert evaluate_runs({"s": one_of_four}, VAL_QRELS, min_coverage=0.25).summary.loc[0, "n_queries"] == 4
    result = evaluate_runs({"s": TEST_RUN}, VAL_QRELS, min_coverage=0, cutoffs=(10,))   # explicit opt-out
    assert result.summary.loc[0, "MRR@10"] == 0.0
    assert result.warnings["s"]["extra"] == ["t1", "t2"]


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_min_coverage_is_rejected(bad):
    with pytest.raises(ValueError, match="min_coverage"):
        evaluate_runs({"s": S1}, QRELS, min_coverage=bad)


def test_empty_run_is_still_allowed_and_scores_zero():
    result = evaluate_runs({"nothing": {}}, VAL_QRELS, cutoffs=(10,))
    assert result.summary.loc[0, "MRR@10"] == 0.0 and result.summary.loc[0, "n_no_results"] == 4


def test_run_covering_several_splits_is_fine_when_it_covers_the_judged_queries():
    big = {**TEST_RUN, "v1": [("A", 1.0)], "v2": [("B", 1.0)], "v3": [("C", 1.0)], "v4": [("D", 1.0)]}
    result = evaluate_runs({"all_splits": big}, VAL_QRELS, cutoffs=(10,))    # extra queries are ignored, not an error
    assert result.summary.loc[0, "MRR@10"] == 1.0
    assert result.warnings["all_splits"]["extra"] == ["t1", "t2"]


def test_all_systems_are_checked_before_any_scoring():
    good = {q: [(next(iter(d)), 1.0)] for q, d in VAL_QRELS.items()}
    with pytest.raises(RunQrelsMismatchError, match="'bad'"):
        evaluate_runs({"good": good, "bad": TEST_RUN}, VAL_QRELS)


def test_mismatch_error_also_raised_through_the_file_wrapper(tmp_path):
    run_file = tmp_path / "test.run"
    run_file.write_text("t1 Q0 A 1 1.0 x\nt2 Q0 B 1 1.0 x\n")
    qrels_file = tmp_path / "val.qrels"
    qrels_file.write_text("v1 0 A 1\nv2 0 B 1\n")
    with pytest.raises(RunQrelsMismatchError):
        evaluate_run_files({"x": run_file}, qrels_file)
    assert evaluate_run_files({"x": run_file}, qrels_file, min_coverage=0).summary.loc[0, "MRR@10"] == 0.0
