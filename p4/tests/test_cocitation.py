"""Tests for p4.graph.cocitation. Counts are worked out by hand or by brute force over pairs."""
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from p4.graph.cocitation import CoCitationIndex, UnknownPaperError


def edges(pairs):
    return pd.DataFrame(pairs, columns=["citing_paper_id", "cited_paper_id"])


# X1 cites A, B ; X2 cites A, B, C ; X3 cites A, C   ->  co(A,B) = 2 (X1, X2), co(A,C) = 2 (X2, X3), co(B,C) = 1 (X2)
EXAMPLE = [("X1", "A"), ("X1", "B"), ("X2", "A"), ("X2", "B"), ("X2", "C"), ("X3", "A"), ("X3", "C")]


def test_counts_are_the_number_of_citing_papers_that_cite_both():
    index = CoCitationIndex.from_edges(edges(EXAMPLE))
    assert index.count("A", "B") == 2 and index.count("A", "C") == 2 and index.count("B", "C") == 1
    assert index.stats.n_pairs == 3 and index.stats.n_citing_papers == 3
    assert index.stats.n_papers == 6                   # A, B, C (cited) and X1, X2, X3 (citing): every paper the edges mention


def test_counts_are_symmetric_and_a_paper_is_never_co_cited_with_itself():
    index = CoCitationIndex.from_edges(edges(EXAMPLE))
    for a, b in itertools.combinations("ABC", 2):
        assert index.count(a, b) == index.count(b, a)
    assert index.count("A", "A") == 0                  # A is cited three times, but not "together with itself"


def test_recommendations_are_ordered_by_count_then_paper_id():
    index = CoCitationIndex.from_edges(edges(EXAMPLE))
    a = index.recommend("A")
    assert list(zip(a["paper_id"], a["cocitation_count"], a["rank"])) == [("B", 2, 1), ("C", 2, 2)]   # tie 2/2 -> B before C
    b = index.recommend("B")
    assert list(zip(b["paper_id"], b["cocitation_count"])) == [("A", 2), ("C", 1)]
    c = index.recommend("C")
    assert list(zip(c["paper_id"], c["cocitation_count"])) == [("A", 2), ("B", 1)]


def test_the_input_paper_is_never_recommended():
    index = CoCitationIndex.from_edges(edges(EXAMPLE))
    for paper in "ABC":
        assert paper not in index.recommend(paper)["paper_id"].tolist()


def test_top_k_limits_the_list_and_large_k_returns_everything():
    index = CoCitationIndex.from_edges(edges(EXAMPLE))
    assert index.recommend("A", top_k=1)["paper_id"].tolist() == ["B"]
    assert len(index.recommend("A", top_k=100)) == 2
    for bad in (0, -3, 1.5, "2"):
        with pytest.raises(ValueError, match="top_k"):
            index.recommend("A", top_k=bad)


def test_ordering_does_not_depend_on_edge_order():
    reference = CoCitationIndex.from_edges(edges(EXAMPLE)).recommend("A")
    for seed in range(5):
        shuffled = [EXAMPLE[i] for i in np.random.default_rng(seed).permutation(len(EXAMPLE))]
        pd.testing.assert_frame_equal(CoCitationIndex.from_edges(edges(shuffled)).recommend("A"), reference)


def test_many_way_ties_are_broken_alphabetically():
    pairs = [("X", p) for p in ("M", "C", "Z", "B", "Q")]                 # X cites five papers: every pair co-cited once
    rec = CoCitationIndex.from_edges(edges(pairs)).recommend("M")
    assert rec["paper_id"].tolist() == ["B", "C", "Q", "Z"] and set(rec["cocitation_count"]) == {1}


# ---- missing papers and papers with no neighbours ------------------------------------------------------
def test_unknown_paper_raises_a_clear_error():
    index = CoCitationIndex.from_edges(edges(EXAMPLE))
    with pytest.raises(UnknownPaperError, match="'NOPE' is not in the co-citation index"):
        index.recommend("NOPE")
    with pytest.raises(UnknownPaperError):
        index.count("A", "NOPE")
    assert isinstance(UnknownPaperError("x"), ValueError) and "A" in index and "NOPE" not in index


def test_a_known_paper_with_no_co_cited_neighbours_gets_an_empty_table():
    # D is cited by Y, but Y cites only D: nothing is cited together with D. X1 cites A and B only.
    index = CoCitationIndex.from_edges(edges([("X1", "A"), ("X1", "B"), ("Y", "D")]))
    rec = index.recommend("D")
    assert rec.empty and list(rec.columns) == ["paper_id", "cocitation_count", "rank"]
    assert index.count("A", "D") == 0


def test_papers_without_any_edge_are_known_when_a_node_list_is_given():
    everyone = ["A", "B", "C", "X1", "X2", "X3"]                          # citing papers are papers too
    index = CoCitationIndex.from_edges(edges(EXAMPLE), nodes=everyone + ["ISOLATED"])
    assert index.recommend("ISOLATED").empty and index.stats.n_papers == 7
    assert index.recommend("X1").empty                                       # a citing paper nobody cites has no co-cited papers
    with pytest.raises(UnknownPaperError):
        index.recommend("STILL_UNKNOWN")
    with pytest.raises(ValueError, match="not in `nodes`"):
        CoCitationIndex.from_edges(edges(EXAMPLE), nodes=["A"])


def test_empty_edge_table_gives_an_index_that_knows_nothing():
    index = CoCitationIndex.from_edges(edges([]))
    assert index.stats.n_papers == 0 and index.stats.n_pairs == 0
    with pytest.raises(UnknownPaperError):
        index.recommend("A")
    assert CoCitationIndex.from_edges(edges([]), nodes=["A"]).recommend("A").empty


# ---- duplicates and self-citations --------------------------------------------------------------------
def test_duplicate_edges_do_not_inflate_counts():
    index = CoCitationIndex.from_edges(edges(EXAMPLE + [("X1", "A"), ("X1", "A"), ("X1", "B"), ("X2", "C")]))
    assert index.count("A", "B") == 2 and index.count("B", "C") == 1       # same as without the duplicates
    assert index.stats.edge_stats.n_duplicates == 4


def test_self_citations_are_ignored():
    # P cites P and Q; R cites P and Q. Counting the self-citation would make co(P, Q) = 2; it is 1 (only R).
    index = CoCitationIndex.from_edges(edges([("P", "P"), ("P", "Q"), ("R", "P"), ("R", "Q")]))
    assert index.count("P", "Q") == 1
    assert index.stats.edge_stats.n_self_loops == 1
    only_self = CoCitationIndex.from_edges(edges([("P", "P"), ("P", "Q")]))     # after dropping P->P, P cites only Q
    assert only_self.recommend("Q").empty and only_self.stats.n_pairs == 0


# ---- limits ---------------------------------------------------------------------------------------
def test_max_references_skips_citing_papers_with_too_many_references():
    pairs = EXAMPLE + [("BIG", p) for p in ("A", "B", "C", "D")]            # BIG cites four papers
    full = CoCitationIndex.from_edges(edges(pairs))
    assert full.count("A", "B") == 3 and full.count("A", "D") == 1
    capped = CoCitationIndex.from_edges(edges(pairs), max_references=3)
    assert capped.count("A", "B") == 2 and capped.count("A", "D") == 0       # BIG ignored
    assert (capped.stats.n_citing_skipped, capped.stats.n_citing_papers) == (1, 3)
    assert "D" in capped                                                    # D is still a known paper
    for bad in (1, 0, 2.5):
        with pytest.raises(ValueError, match="max_references"):
            CoCitationIndex.from_edges(edges(pairs), max_references=bad)


def test_matches_a_brute_force_count_on_a_random_graph():
    rng = np.random.default_rng(12)
    papers = [f"p{i:02d}" for i in range(30)]
    pairs = sorted({(papers[a], papers[b]) for a, b in rng.integers(0, 30, size=(250, 2))})
    references = {}
    for citing, cited in pairs:
        if citing != cited:
            references.setdefault(citing, set()).add(cited)
    expected = Counter()
    for cited_set in references.values():
        for x, y in itertools.combinations(sorted(cited_set), 2):          # every pair this paper cites together
            expected[(x, y)] += 1
    index = CoCitationIndex.from_edges(edges(pairs))
    assert index.stats.n_pairs == len(expected)
    for x, y in itertools.combinations(papers, 2):
        assert index.count(x, y) == expected.get((x, y), 0), (x, y)
    for paper in papers[:5]:                                               # recommendations agree with the brute-force table
        neighbours = sorted(((-expected[k], k[1] if k[0] == paper else k[0]) for k in expected if paper in k))
        if paper in index:
            rec = index.recommend(paper, top_k=1000)
            assert list(zip(-rec["cocitation_count"], rec["paper_id"])) == neighbours


# ---- duplicates and self-citations together, and consistency with PageRank's cleaning ---------------------------
CLEAN_GRAPH = EXAMPLE + [("P", "Q"), ("P", "R")]
DIRTY_GRAPH = CLEAN_GRAPH + [("X1", "A"), ("X1", "A"), ("X2", "C"),                  # 3 duplicate rows
                             ("A", "A"), ("P", "P"), ("P", "P"), ("Q", "Q"), ("X3", "X3")]   # 5 self-citations


def test_duplicates_and_self_citations_together_leave_every_count_unchanged():
    clean, dirty = CoCitationIndex.from_edges(edges(CLEAN_GRAPH)), CoCitationIndex.from_edges(edges(DIRTY_GRAPH))
    papers = sorted({p for pair in CLEAN_GRAPH for p in pair})
    for x, y in itertools.permutations(papers, 2):
        assert dirty.count(x, y) == clean.count(x, y), (x, y)
    assert (dirty.stats.n_pairs, dirty.stats.n_papers) == (clean.stats.n_pairs, clean.stats.n_papers)
    assert (dirty.stats.edge_stats.n_self_loops, dirty.stats.edge_stats.n_duplicates) == (5, 3)
    # P cites P, Q and R: the self-citation is dropped, so P still co-cites Q and R once, and never pairs with itself
    assert dirty.count("Q", "R") == 1 and dirty.count("P", "Q") == 0 and dirty.count("P", "R") == 0


def test_cocitation_and_pagerank_apply_exactly_the_same_cleaning():
    from p4.graph.pagerank import pagerank
    index = CoCitationIndex.from_edges(edges(DIRTY_GRAPH))
    assert index.stats.edge_stats == pagerank(edges(DIRTY_GRAPH)).edge_stats          # same counts of dropped rows
    assert index.stats.edge_stats.n_edges == len(CLEAN_GRAPH)


def test_cocitation_counts_citing_papers_so_edge_weights_never_matter():
    weighted = edges(DIRTY_GRAPH)
    weighted["num_contexts"] = np.arange(1, len(weighted) + 1)                         # a different weight on every row
    plain, heavy = CoCitationIndex.from_edges(edges(CLEAN_GRAPH)), CoCitationIndex.from_edges(weighted)
    for x, y in itertools.permutations(sorted({p for pair in CLEAN_GRAPH for p in pair}), 2):
        assert heavy.count(x, y) == plain.count(x, y)


def test_max_references_is_judged_after_cleaning():
    # P writes P->P, P->Q, P->R: after dropping the self-citation it has 2 real references, so max_references=2 keeps it.
    index = CoCitationIndex.from_edges(edges(DIRTY_GRAPH), max_references=2)
    assert index.count("Q", "R") == 1 and index.stats.n_citing_skipped == 1          # only X2 (3 references) is skipped
