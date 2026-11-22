"""Tests for p4.eval.metrics. Every expected value is worked out by hand in a comment."""
import math

import pytest

from p4.eval.metrics import average_precision, ndcg_at_k, recall_at_k, reciprocal_rank

# One relevant document, found at rank 3.
RANKED = ["a", "b", "c", "d"]
REL = {"c"}


def test_recall_single_relevant_at_several_cutoffs():
    assert recall_at_k(RANKED, REL, 1) == 0.0   # c not in [a]
    assert recall_at_k(RANKED, REL, 2) == 0.0   # c not in [a, b]
    assert recall_at_k(RANKED, REL, 3) == 1.0   # c in [a, b, c]
    assert recall_at_k(RANKED, REL, 100) == 1.0  # cutoff longer than the list is fine


def test_reciprocal_rank_single_relevant():
    assert reciprocal_rank(RANKED, REL) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, REL, k=3) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, REL, k=2) == 0.0  # found only after the cutoff


def test_ndcg_single_relevant_is_one_over_log2_rank_plus_one():
    assert ndcg_at_k(RANKED, REL, 3) == pytest.approx(0.5)        # 1 / log2(3 + 1) = 1/2
    assert ndcg_at_k(["c"], REL, 5) == pytest.approx(1.0)         # rank 1: 1 / log2(2) = 1
    assert ndcg_at_k(["x", "c"], REL, 5) == pytest.approx(1 / math.log2(3))  # rank 2 = 0.63093
    assert ndcg_at_k(RANKED, REL, 2) == 0.0


def test_average_precision_single_relevant_equals_one_over_rank():
    assert average_precision(RANKED, REL) == pytest.approx(1 / 3)


def test_nothing_relevant_retrieved_scores_zero_everywhere():
    assert recall_at_k(["x", "y"], REL, 10) == 0.0
    assert reciprocal_rank(["x", "y"], REL) == 0.0
    assert ndcg_at_k(["x", "y"], REL, 10) == 0.0
    assert average_precision(["x", "y"], REL) == 0.0
    assert average_precision([], REL) == 0.0   # empty ranking


# Several relevant documents: relevant = {r1, r2, r3}, ranking = [r1, x, r2, y]
#   (r3 is never retrieved)
RANKED_MULTI = ["r1", "x", "r2", "y"]
REL_MULTI = {"r1", "r2", "r3"}


def test_recall_with_several_relevant():
    assert recall_at_k(RANKED_MULTI, REL_MULTI, 1) == pytest.approx(1 / 3)  # r1
    assert recall_at_k(RANKED_MULTI, REL_MULTI, 3) == pytest.approx(2 / 3)  # r1, r2
    assert recall_at_k(RANKED_MULTI, REL_MULTI, 4) == pytest.approx(2 / 3)  # r3 never found


def test_average_precision_with_several_relevant():
    # precision at rank 1 = 1/1, at rank 3 = 2/3; divide by 3 relevant documents
    assert average_precision(RANKED_MULTI, REL_MULTI) == pytest.approx((1 + 2 / 3) / 3)  # 5/9


def test_ndcg_with_several_relevant():
    dcg = 1 / math.log2(2) + 1 / math.log2(4)                            # ranks 1 and 3 = 1.5
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)       # perfect: ranks 1, 2, 3
    assert ndcg_at_k(RANKED_MULTI, REL_MULTI, 4) == pytest.approx(dcg / ideal)  # about 0.7039
    # With k = 2 only two ideal positions count: ideal = 1 + 1/log2(3); dcg = 1 (only r1 in top 2)
    assert ndcg_at_k(RANKED_MULTI, REL_MULTI, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_relevant_given_as_dict_keys_works():
    assert recall_at_k(RANKED, {"c": 1}, 3) == 1.0


@pytest.mark.parametrize("bad_k", [0, -1, 2.5, "10"])
def test_invalid_cutoff_is_rejected(bad_k):
    for fn in (recall_at_k, ndcg_at_k):
        with pytest.raises(ValueError, match="cutoff"):
            fn(RANKED, REL, bad_k)
    with pytest.raises(ValueError, match="cutoff"):
        reciprocal_rank(RANKED, REL, bad_k)


def test_empty_relevant_set_is_rejected():
    for fn in (lambda: recall_at_k(RANKED, set(), 3), lambda: reciprocal_rank(RANKED, set()),
               lambda: ndcg_at_k(RANKED, set(), 3), lambda: average_precision(RANKED, set())):
        with pytest.raises(ValueError, match="at least one relevant"):
            fn()


# ---- Batch 1: duplicate document IDs -----------------------------------------
# Without the guard, ranking ["r", "r"] with relevant {"r"} would give
# AP = (1/1 + 2/2) / 1 = 2.0 and nDCG = (1 + 1/log2(3)) / 1 = 1.63, both impossible (> 1).
DUP = ["r", "x", "r"]


@pytest.mark.parametrize("call", [
    lambda: average_precision(DUP, {"r"}),
    lambda: ndcg_at_k(DUP, {"r"}, 3),
    lambda: recall_at_k(DUP, {"r"}, 3),
    lambda: reciprocal_rank(DUP, {"r"}),
    lambda: reciprocal_rank(DUP, {"r"}, k=3),
])
def test_duplicate_document_in_ranking_is_rejected(call):
    with pytest.raises(ValueError, match="'r' more than once"):
        call()


def test_duplicate_beyond_the_examined_cutoff_is_not_an_error():
    # The duplicate "r" is at ranks 1 and 3, but a cutoff of 2 only examines ["r", "x"].
    assert recall_at_k(DUP, {"r"}, 2) == 1.0
    assert reciprocal_rank(DUP, {"r"}, k=2) == 1.0
    assert ndcg_at_k(DUP, {"r"}, 2) == pytest.approx(1.0)
    with pytest.raises(ValueError):          # average precision always examines the whole list
        average_precision(DUP, {"r"})


def test_ranking_without_duplicates_is_unaffected_by_the_guard():
    assert average_precision(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert average_precision([], {"c"}) == 0.0
