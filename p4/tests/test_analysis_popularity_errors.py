"""Tests for p4.analysis.popularity (retrieval-side popularity) and p4.analysis.errors (failure analysis)."""
import numpy as np
import pandas as pd
import pytest

from p4.analysis import errors as E
from p4.analysis import popularity as P
from p4.analysis.data import build_query_table
from p4.tests import analysis_fixtures as F

RUNS, QRELS, PAPERS = F.make_runs(), F.make_qrels(), F.make_papers()


# ======================================================================== popularity (retrieval side)
def test_corpus_reference_by_hand():
    ref = P.corpus_reference(PAPERS)
    # 50 papers: 5 with 0 citations, 5 with 1, 20 with 3..7, 20 with 61..80. The 90th percentile is 75.1, so the top
    # decile is the five papers with 76..80 citations.
    assert ref["n_papers"] == 50 and ref["n_zero_citation"] == 5 and ref["zero_citation_share"] == pytest.approx(0.1)
    assert ref["threshold"] == pytest.approx(75.1) and ref["top_decile_share"] == pytest.approx(0.1)


def test_retrieval_popularity_for_a_run_whose_top_k_is_known_by_hand():
    summary, per_query, ref = P.retrieval_popularity({"good": RUNS["good"]}, QRELS, PAPERS, k=10, n_boot=200)
    s = summary.iloc[0]
    # every top-10 is: the relevant paper, D06..D10 (1 citation) and D01..D04 (0 citations)
    assert s["n_queries"] == 40 and s["n_docs_topk"] == 400 and s["n_docs_unknown_citation_count"] == 0
    assert s["zero_citation_share_topk"] == pytest.approx(160 / 400)                 # 4 of 10 documents, always
    assert s["top_decile_share_topk"] == pytest.approx(5 / 400)                      # only q36-q40's relevant papers (76..80)
    assert s["top_decile_lift"] == pytest.approx((5 / 400) / 0.1) and s["corpus_zero_citation_share"] == pytest.approx(0.1)
    expected_mean = np.mean([(np.log10(1 + F.citation_count(i)) + 5 * np.log10(2) + 4 * 0.0) / 10 for i in range(1, 41)])
    assert s["mean_log10_citations_topk"] == pytest.approx(expected_mean)
    assert s["n_wrong_top1"] == 0 and np.isnan(s["wrong_top1_more_cited_rate"])      # the top-1 is always right
    assert per_query["top1_is_relevant"].all()


def test_wrong_top_one_papers_are_compared_with_the_relevant_paper():
    summary, per_query, _ = P.retrieval_popularity({"weak": RUNS["weak"]}, QRELS, PAPERS, k=10, n_boot=200)
    s = summary.iloc[0]
    assert s["n_wrong_top1"] == 20                                                    # q06-q20 and q36-q40 are wrong at rank 1
    assert s["wrong_top1_more_cited_rate"] == 0.0 and s["wrong_top1_less_cited_rate"] == 1.0      # D06 (1 citation) < every target
    assert s["wrong_top1_more_cited_ci_low"] == 0.0 and s["wrong_top1_more_cited_ci_high"] == 0.0


def test_a_system_that_returns_a_more_cited_paper_is_detected():
    runs = {"pop": {q: [("T40", 9.0)] + [(f"X{n}", 8.0 - n) for n in range(9)] for q in QRELS}}    # always the most cited paper
    s = P.retrieval_popularity(runs, QRELS, PAPERS, k=10, n_boot=200)[0].iloc[0]
    # T40 (80 citations) is the most cited paper, so it beats every OTHER relevant paper; for q40 it is the right answer
    assert s["n_wrong_top1"] == 39 and s["wrong_top1_more_cited_rate"] == 1.0 and s["wrong_top1_less_cited_rate"] == 0.0
    # the nine X documents per query are in no table: counted as unknown and left out, so the only KNOWN document is T40
    assert s["n_docs_unknown_citation_count"] == 360 and s["median_citations_topk"] == 80
    assert s["top_decile_share_topk"] == 1.0


def test_unknown_documents_are_counted_not_dropped_silently():
    # drop D06..D10 from the papers table: they are then unknown. 20 hit queries have 5 of them in the top 10 (100 documents);
    # the 20 missed queries have D06 plus X00..X08, none of them in the table (200 documents).
    s = P.retrieval_popularity({"w": RUNS["weak"]}, QRELS, PAPERS.iloc[:45], k=10, n_boot=200)[0].iloc[0]
    assert s["n_docs_topk"] == 400 and s["n_docs_unknown_citation_count"] == 300


def test_popularity_input_checks():
    with pytest.raises(ValueError, match="no citation_count column"):
        P.retrieval_popularity(RUNS, QRELS, PAPERS[["paper_id", "title"]])
    with pytest.raises(ValueError, match="k must be"):
        P.retrieval_popularity(RUNS, QRELS, PAPERS, k=0)


# ======================================================================== errors
def test_categories_at_every_boundary():
    assert [E.categorise(r, 10) for r in (1, 2, 10, 11, 100)] == ["rank 1", "rank 2-10", "rank 2-10", "rank 11 or deeper", "rank 11 or deeper"]
    assert E.categorise(float("nan"), 10) == "not retrieved" and E.category_order(5)[1] == "rank 2-5"


def test_relevant_ranks_by_hand():
    ranks = E.relevant_ranks(RUNS, QRELS, k=10)
    weak = ranks[ranks.system == "weak"].set_index("query_id")
    assert weak.loc["q01", "rank"] == 1 and weak.loc["q06", "rank"] == 13 and np.isnan(weak.loc["q18", "rank"])
    assert weak.loc["q06", "category"] == "rank 11 or deeper" and weak.loc["q18", "category"] == "not retrieved"
    assert weak.loc["q06", "top1_id"] == "D06" and not weak.loc["q06", "top1_is_relevant"] and weak.loc["q01", "top1_is_relevant"]
    assert bool(weak.loc["q01", "hit"]) and not weak.loc["q06", "hit"]
    assert (ranks[ranks.system == "good"]["rank"] == 1).all() and (ranks[ranks.system == "good"]["run_depth"] == 10).all()


def test_missing_results_count_as_not_retrieved():
    ranks = E.relevant_ranks({"s": {"q01": [("T01", 1.0)]}}, {"q01": {"T01": 1}, "q02": {"T02": 1}}, k=10)
    q2 = ranks[ranks.query_id == "q02"].iloc[0]
    assert q2["category"] == "not retrieved" and q2["run_depth"] == 0 and pd.isna(q2["top1_id"]) and not q2["hit"]


def test_error_summary_by_hand():
    s = E.error_summary(E.relevant_ranks(RUNS, QRELS, 10), 10).set_index("system")
    assert s.loc["good", "n_failures"] == 0 and s.loc["good", "failure_rate"] == 0.0 and s.loc["good", "n_rank 1"] == 40
    # weak: hits q01-05 and q21-35 = 20; the other 20 miss: q06-17 and q36-40 at rank 13 (17 queries), q18-20 absent (3)
    assert s.loc["weak", "n_hits"] == 20 and s.loc["weak", "n_failures"] == 20 and s.loc["weak", "failure_rate"] == 0.5
    assert s.loc["weak", "n_rank 11 or deeper"] == 17 and s.loc["weak", "n_not retrieved"] == 3 and s.loc["weak", "n_rank 1"] == 20
    assert s.loc["weak", "median_rank_when_found"] == 1.0          # 37 found: twenty at rank 1, seventeen at rank 13


def test_pairwise_outcomes_with_an_exact_binomial_test_by_hand():
    ranks = E.relevant_ranks(RUNS, QRELS, 10)
    p = E.pairwise_outcomes(ranks, baseline="weak").iloc[0]
    # good hits all 40, weak hits 20: 20 queries only `good` hits, none only `weak`
    assert (p["both_hit"], p["only_system_hits"], p["only_baseline_hits"], p["neither_hits"]) == (20, 20, 0, 0)
    assert p["n_discordant"] == 20 and p["p_value"] == pytest.approx(2 * 0.5 ** 20) and p["significant_holm"]
    flipped = E.pairwise_outcomes(ranks, baseline="good").iloc[0]
    assert flipped["only_system_hits"] == 0 and flipped["only_baseline_hits"] == 20 and flipped["p_value"] == pytest.approx(p["p_value"])


def test_pairwise_with_no_discordant_queries_has_no_p_value():
    ranks = E.relevant_ranks({"a": RUNS["good"], "b": RUNS["good"]}, QRELS, 10)
    p = E.pairwise_outcomes(ranks, "a").iloc[0]
    assert p["n_discordant"] == 0 and np.isnan(p["p_value"]) and not p["significant_holm"]
    with pytest.raises(ValueError, match="baseline 'zzz'"):
        E.pairwise_outcomes(ranks, "zzz")


def test_hard_queries_hubs_and_missed_papers():
    ranks = E.relevant_ranks(RUNS, QRELS, 10)
    table = build_query_table(QRELS, F.make_queries(), PAPERS, F.make_functions())
    hard = E.hard_queries(ranks, table, top_n=50)
    assert len(hard) == 40 and hard.iloc[0]["n_systems_failing"] == 1 and hard["failed_by_all"].sum() == 0     # `good` never fails
    assert hard.iloc[0]["query_id"] == "q06" and hard.iloc[0]["systems_failing"] == "weak"                    # ties: lowest query id first
    assert hard.iloc[0]["title"] == "Paper T06" and hard.iloc[0]["text"] == "query sentence 6"
    assert hard["n_systems_failing"].is_monotonic_decreasing
    hubs = E.hub_papers(ranks, PAPERS)
    assert hubs.iloc[0]["paper_id"] == "D06" and hubs.iloc[0]["system"] == "weak"
    assert hubs.iloc[0]["n_queries_wrong_top1"] == 20 and hubs.iloc[0]["citation_count"] == 1 and "good" not in set(hubs["system"])
    missed = E.missed_papers(ranks, table)
    assert len(missed) == 20 and missed["n_pairs_missed"].eq(1).all() and not missed["missed_in_every_pair"].any()
    assert missed.iloc[0]["target_id"] == "T06" and missed.iloc[0]["systems_that_missed"] == "weak" and missed.iloc[0]["n_pairs"] == 2


def test_a_paper_missed_by_every_system_is_flagged():
    runs = {"a": {"q1": [("X", 1.0)]}, "b": {"q1": [("Y", 1.0)]}}
    ranks = E.relevant_ranks(runs, {"q1": {"T": 1}}, 10)
    table = build_query_table({"q1": {"T": 1}}, None, None)
    m = E.missed_papers(ranks, table).iloc[0]
    assert m["missed_in_every_pair"] and m["systems_that_missed"] == "a; b"
    assert E.hard_queries(ranks, table).iloc[0]["failed_by_all"]


def test_top_decile_never_includes_papers_with_zero_citations():
    # 19 papers have 0 citations and one has 7: the 90th percentile is 0, but zero-citation papers are not "most cited"
    papers = pd.DataFrame({"paper_id": [f"p{i}" for i in range(20)], "citation_count": [0] * 19 + [7]})
    ref = P.corpus_reference(papers)
    assert ref["threshold"] == 0 and ref["top_decile_share"] == pytest.approx(1 / 20) and ref["zero_citation_share"] == pytest.approx(19 / 20)


def test_a_hit_at_exactly_rank_k_counts_and_rank_k_plus_one_does_not():
    run = {"q1": [(f"X{n}", 1.0) for n in range(9)] + [("T", 0.5)],                   # relevant paper at rank 10
           "q2": [(f"X{n}", 1.0) for n in range(10)] + [("T", 0.5)]}                  # relevant paper at rank 11
    ranks = E.relevant_ranks({"s": run}, {"q1": {"T": 1}, "q2": {"T": 1}}, k=10).set_index("query_id")
    assert ranks.loc["q1", "rank"] == 10 and bool(ranks.loc["q1", "hit"]) and ranks.loc["q1", "category"] == "rank 2-10"
    assert ranks.loc["q2", "rank"] == 11 and not ranks.loc["q2", "hit"] and ranks.loc["q2", "category"] == "rank 11 or deeper"
