"""Tests for p4.graph.pagerank. Reference values come from an independent dense linear solve (and hand calculation)."""
import numpy as np
import pandas as pd
import pytest

from p4.graph.edges import clean_edges
from p4.graph.pagerank import pagerank


def edges(pairs, weights=None):
    df = pd.DataFrame(pairs, columns=["citing_paper_id", "cited_paper_id"])
    if weights is not None:
        df["num_contexts"] = weights
    return df


def exact_pagerank(ids, pairs, damping=0.85):
    """PageRank by solving the linear system r = d P r + (1-d)/N directly (no iteration), as an independent check."""
    n, position = len(ids), {k: i for i, k in enumerate(ids)}
    P, out = np.zeros((n, n)), np.zeros(n)
    for source, target in pairs:
        P[position[target], position[source]] += 1
        out[position[source]] += 1
    for j in range(n):
        P[:, j] = P[:, j] / out[j] if out[j] else 1.0 / n          # a paper citing nothing spreads its score evenly
    return dict(zip(ids, np.linalg.solve(np.eye(n) - damping * P, np.full(n, (1 - damping) / n))))


# ---- known graphs ---------------------------------------------------------------------------------
def test_three_cycle_gives_equal_scores_ranked_alphabetically():
    r = pagerank(edges([("A", "B"), ("B", "C"), ("C", "A")]))
    assert r.scores.tolist() == pytest.approx([1 / 3] * 3)
    assert list(r.ranking["paper_id"]) == ["A", "B", "C"]            # a perfect tie is broken by paper_id


def test_chain_with_a_dangling_end_matches_the_hand_calculation():
    # A -> B -> C, C cites nothing. With d = 0.85 and N = 3 the equations are
    #   r_C = 0.85 (r_B + r_C/3) + 0.05  ...  solving gives r_A = 0.18442, r_B = 0.34117, r_C = 0.47441
    r = pagerank(edges([("A", "B"), ("B", "C")]))
    assert r.scores.to_dict() == pytest.approx({"A": 0.184417, "B": 0.341171, "C": 0.474412}, abs=1e-6)
    assert list(r.ranking["paper_id"]) == ["C", "B", "A"]            # being cited by an important paper pays off
    assert r.n_dangling == 1 and r.n_nodes == 3 and r.n_edges == 2


def test_star_graph_hand_calculation_and_direction_of_propagation():
    # A, B, C each cite D. x = 0.85 (y/4) + 0.0375 for A, B, C and y = 1 - 3x for D: y = 0.541985, x = 0.152672
    r = pagerank(edges([("A", "D"), ("B", "D"), ("C", "D")]))
    assert r.scores.to_dict() == pytest.approx({"A": 0.152672, "B": 0.152672, "C": 0.152672, "D": 0.541985}, abs=1e-6)
    assert r.scores["D"] > r.scores["A"]                              # the CITED paper gains, not the citing ones


def test_matches_an_independent_exact_solution_on_a_random_graph():
    rng = np.random.default_rng(4)
    ids = [f"p{i:02d}" for i in range(40)]
    pairs = {(ids[a], ids[b]) for a, b in rng.integers(0, 40, size=(120, 2)) if a != b}
    r = pagerank(edges(sorted(pairs)), nodes=ids)
    expected = exact_pagerank(ids, sorted(pairs))
    assert r.scores.to_dict() == pytest.approx(expected, abs=1e-8)
    assert r.converged


def test_damping_changes_the_result_and_matches_the_exact_solution():
    pairs = [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("D", "A")]
    for d in (0.5, 0.85, 0.95):
        r = pagerank(edges(pairs), damping=d, tol=1e-13, max_iter=2000)
        assert r.scores.to_dict() == pytest.approx(exact_pagerank(list("ABCD"), pairs, d), abs=1e-9), d
    assert pagerank(edges(pairs), damping=0.5).scores["A"] != pytest.approx(pagerank(edges(pairs)).scores["A"])


def test_zero_damping_gives_the_uniform_distribution():
    r = pagerank(edges([("A", "B"), ("B", "C")]), damping=0)
    assert r.scores.tolist() == pytest.approx([1 / 3] * 3)


# ---- dangling and isolated nodes ----------------------------------------------------------------
def test_dangling_nodes_keep_the_total_score_at_one_and_spread_evenly():
    r = pagerank(edges([("A", "C"), ("B", "C")]))                      # C cites nothing (dangling)
    assert r.scores.sum() == pytest.approx(1.0, abs=1e-12)
    assert r.n_dangling == 1
    assert r.scores.to_dict() == pytest.approx(exact_pagerank(list("ABC"), [("A", "C"), ("B", "C")]), abs=1e-9)


def test_isolated_nodes_get_a_small_equal_share_and_are_ranked_last_or_tied():
    r = pagerank(edges([("A", "B"), ("B", "C")]), nodes=["A", "B", "C", "Y", "Z"])
    assert r.n_nodes == 5 and r.scores.sum() == pytest.approx(1.0)
    assert r.scores["Y"] == pytest.approx(r.scores["Z"])               # both isolated: identical score
    assert r.scores["Y"] > 0                                           # but never zero
    assert r.scores["Y"] == pytest.approx(r.scores["A"])               # an uncited paper that cites someone ends up equal
    assert r.scores.to_dict() == pytest.approx(exact_pagerank(list("ABCYZ"), [("A", "B"), ("B", "C")]), abs=1e-9)


def test_papers_without_any_edge_are_uniform():
    r = pagerank(edges([]), nodes=["x", "y", "z", "w"])
    assert r.scores.tolist() == pytest.approx([0.25] * 4) and r.n_edges == 0 and r.converged
    assert list(r.ranking["paper_id"]) == ["w", "x", "y", "z"]         # tie -> alphabetical


def test_empty_graph_gives_an_empty_result_not_an_error():
    r = pagerank(edges([]))
    assert r.ranking.empty and r.scores.empty and r.n_nodes == 0 and r.converged and r.iterations == 0
    assert list(r.ranking.columns) == ["paper_id", "score", "score_scaled", "rank"]


# ---- normalisation, convergence, determinism ----------------------------------------------------------
def test_scores_sum_to_one_and_scaled_scores_peak_at_one():
    r = pagerank(edges([("A", "B"), ("B", "C"), ("D", "C"), ("C", "A"), ("E", "B")]))
    assert r.scores.sum() == pytest.approx(1.0, abs=1e-12)
    assert r.ranking["score"].sum() == pytest.approx(1.0, abs=1e-12)
    assert r.ranking["score_scaled"].max() == 1.0 and (r.ranking["score_scaled"] > 0).all()
    assert r.ranking["score_scaled"].tolist() == pytest.approx((r.ranking["score"] / r.ranking["score"].max()).tolist())
    assert r.ranking["rank"].tolist() == list(range(1, 6))


def test_convergence_is_reported_and_depends_on_tolerance_and_iteration_limit():
    graph = edges([("A", "B"), ("B", "C")])
    r = pagerank(graph)
    assert r.converged and r.final_delta < r.tol and r.iterations < 100
    loose = pagerank(graph, tol=1e-3)
    assert loose.converged and loose.iterations < r.iterations         # a looser tolerance stops earlier
    capped = pagerank(graph, max_iter=2)
    assert not capped.converged and capped.iterations == 2 and capped.final_delta > capped.tol
    assert capped.scores.sum() == pytest.approx(1.0)                   # still a valid distribution when it stops early
    assert loose.scores.to_dict() == pytest.approx(r.scores.to_dict(), abs=1e-3)


def test_ranking_is_deterministic_whatever_the_edge_order():
    pairs = [("A", "D"), ("B", "D"), ("C", "D"), ("D", "E"), ("E", "A"), ("B", "C")]
    reference = pagerank(edges(pairs))
    for seed in range(5):
        shuffled = [pairs[i] for i in np.random.default_rng(seed).permutation(len(pairs))]
        again = pagerank(edges(shuffled))
        pd.testing.assert_frame_equal(again.ranking, reference.ranking)
        assert again.scores.equals(reference.scores)


def test_ties_are_broken_by_paper_id_even_when_floating_point_noise_differs():
    # B and C are symmetric: cited by the same paper X. Their scores agree to ~1e-17, order must still be B, C.
    r = pagerank(edges([("X", "B"), ("X", "C"), ("B", "Y"), ("C", "Y")]))
    assert abs(r.scores["B"] - r.scores["C"]) < 1e-12
    assert list(r.ranking["paper_id"]).index("B") < list(r.ranking["paper_id"]).index("C")


# ---- top-k ----------------------------------------------------------------------------------------
def test_top_k_limits_the_ranking_but_not_the_scores():
    r = pagerank(edges([("A", "B"), ("B", "C"), ("C", "D")]), top_k=2)
    assert list(r.ranking["paper_id"]) == ["D", "C"] and r.ranking["rank"].tolist() == [1, 2]
    assert len(r.scores) == 4
    assert len(pagerank(edges([("A", "B")]), top_k=50).ranking) == 2   # k larger than the graph returns everything


@pytest.mark.parametrize("kwargs", [dict(top_k=0), dict(top_k=-1), dict(top_k=1.5), dict(damping=1.0), dict(damping=-0.1),
                                    dict(tol=0), dict(tol=-1e-9), dict(max_iter=0), dict(max_iter=2.5)])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        pagerank(edges([("A", "B")]), **kwargs)


# ---- duplicates, self-citations, weights, node universe --------------------------------------------------
def test_duplicate_edges_do_not_change_unweighted_pagerank():
    clean = pagerank(edges([("A", "B"), ("A", "C"), ("B", "C")]))
    dup = pagerank(edges([("A", "B"), ("A", "B"), ("A", "B"), ("A", "C"), ("B", "C")]))
    assert dup.scores.to_dict() == pytest.approx(clean.scores.to_dict()) and dup.n_edges == 3
    assert dup.edge_stats.n_duplicates == 2


def test_self_citations_are_ignored_and_reported():
    base = pagerank(edges([("A", "B"), ("B", "C")]))
    with_loops = pagerank(edges([("A", "B"), ("B", "C"), ("A", "A"), ("C", "C"), ("C", "C")]))
    assert with_loops.scores.to_dict() == pytest.approx(base.scores.to_dict())
    assert with_loops.edge_stats.n_self_loops == 3 and with_loops.n_dangling == 1   # C still cites nothing


def test_a_paper_that_only_cites_itself_is_just_isolated():
    r = pagerank(edges([("A", "A")]), nodes=["A", "B"])
    assert r.scores.tolist() == pytest.approx([0.5, 0.5]) and r.n_edges == 0


def test_weights_change_the_scores_only_when_requested():
    pairs, weights = [("A", "B"), ("A", "C")], [9, 1]
    plain = pagerank(edges(pairs, weights))
    assert plain.scores["B"] == pytest.approx(plain.scores["C"])        # unweighted: equal citations
    heavy = pagerank(edges(pairs, weights), weighted=True)
    assert heavy.scores["B"] > heavy.scores["C"]
    # A splits 0.85 * r_A in the ratio 9 : 1, so B - C = 0.85 * r_A * 0.8
    assert heavy.scores["B"] - heavy.scores["C"] == pytest.approx(0.85 * heavy.scores["A"] * 0.8)
    assert heavy.scores.sum() == pytest.approx(1.0)


def test_weighted_pagerank_needs_the_weight_column():
    with pytest.raises(Exception, match="num_contexts"):
        pagerank(edges([("A", "B")]), weighted=True)


def test_edge_endpoints_outside_the_given_nodes_are_an_error():
    with pytest.raises(ValueError, match="not in `nodes`"):
        pagerank(edges([("A", "B")]), nodes=["A"])


def test_accepts_an_already_cleaned_edge_table():
    table = clean_edges(edges([("A", "B"), ("B", "C")]))
    assert pagerank(table).scores.to_dict() == pytest.approx(pagerank(edges([("A", "B"), ("B", "C")])).scores.to_dict())


def test_scores_that_differ_only_by_floating_point_noise_are_ordered_by_paper_id():
    from p4.graph.pagerank import _rank_table
    ids = np.array(["B", "C", "A"], dtype=object)
    noisy = np.array([0.2, 0.2 + 3e-16, 0.2 - 1e-16])                 # all equal up to rounding noise
    assert list(_rank_table(ids, noisy)["paper_id"]) == ["A", "B", "C"]
    assert list(_rank_table(ids, noisy[::-1].copy())["paper_id"]) == ["A", "B", "C"]    # whatever the noise looks like
    real_gap = np.array([0.2, 0.2 + 1e-6, 0.2])                       # a genuine difference of 1e-6 is NOT a tie
    assert list(_rank_table(ids, real_gap)["paper_id"]) == ["C", "A", "B"]


# ---- weighted PageRank: hand calculation and the weighted / unweighted difference ----------------------
def exact_weighted_pagerank(ids, weighted_pairs, damping=0.85):
    """Direct linear solve with edge weights: a paper splits its score in proportion to the weights of its citations."""
    n, position = len(ids), {k: i for i, k in enumerate(ids)}
    P = np.zeros((n, n))
    for source, target, weight in weighted_pairs:
        P[position[target], position[source]] += weight
    for j in range(n):
        total = P[:, j].sum()
        P[:, j] = P[:, j] / total if total else 1.0 / n
    return dict(zip(ids, np.linalg.solve(np.eye(n) - damping * P, np.full(n, (1 - damping) / n))))


def test_weighted_pagerank_matches_a_hand_calculation_where_edge_weights_differ():
    # A -> B (3 sentences), A -> C (1 sentence); B and C cite nothing; d = 0.85, N = 3.
    #   r_A = 0.05 + 0.85 (1 - r_A) / 3                        -> r_A = 0.259740
    #   weighted:   A sends 3/4 to B and 1/4 to C              -> r_B = 0.05 + 0.85 (0.75 r_A + (1 - r_A)/3) = 0.425325
    #                                                             r_C = 0.05 + 0.85 (0.25 r_A + (1 - r_A)/3) = 0.314935
    #   unweighted: A sends 1/2 to each                        -> r_B = r_C = (1 - r_A) / 2 = 0.370130
    graph = edges([("A", "B"), ("A", "C")], [3, 1])
    weighted = pagerank(graph, weighted=True)
    assert weighted.scores.to_dict() == pytest.approx({"A": 0.259740, "B": 0.425325, "C": 0.314935}, abs=1e-6)
    assert list(weighted.ranking["paper_id"]) == ["B", "C", "A"]
    unweighted = pagerank(graph)                                    # the num_contexts column is present but must be ignored
    assert unweighted.scores.to_dict() == pytest.approx({"A": 0.259740, "B": 0.370130, "C": 0.370130}, abs=1e-6)
    assert unweighted.scores["B"] == pytest.approx(unweighted.scores["C"])
    assert weighted.scores["B"] != pytest.approx(unweighted.scores["B"], abs=1e-3)      # the two modes really differ
    exact = exact_weighted_pagerank(["A", "B", "C"], [("A", "B", 3), ("A", "C", 1)])
    assert weighted.scores.to_dict() == pytest.approx(exact, abs=1e-9)


def test_weighted_pagerank_matches_an_exact_solution_on_a_random_weighted_graph():
    rng = np.random.default_rng(31)
    ids = [f"p{i:02d}" for i in range(25)]
    pairs = sorted({(ids[a], ids[b]) for a, b in rng.integers(0, 25, size=(90, 2)) if a != b})
    weights = rng.integers(1, 9, size=len(pairs))
    graph = edges(pairs, weights.tolist())
    result = pagerank(graph, nodes=ids, weighted=True)
    exact = exact_weighted_pagerank(ids, [(s, t, w) for (s, t), w in zip(pairs, weights)])
    assert result.scores.to_dict() == pytest.approx(exact, abs=1e-8)
    assert result.scores.sum() == pytest.approx(1.0)


def test_weighted_duplicate_rows_are_summed_into_one_edge():
    merged = pagerank(edges([("A", "B"), ("A", "C")], [3, 1]), weighted=True)
    split = pagerank(edges([("A", "B"), ("A", "B"), ("A", "C")], [1, 2, 1]), weighted=True)     # B-edge: 1 + 2 = 3
    assert split.scores.to_dict() == pytest.approx(merged.scores.to_dict())
    assert split.edge_stats.n_duplicates == 1


def test_weighted_is_refused_for_a_table_read_without_weights_instead_of_silently_ignoring_it():
    plain = clean_edges(edges([("A", "B"), ("A", "C")], [3, 1]))                     # no weight_col -> placeholder weights
    assert not plain.has_weights
    with pytest.raises(ValueError, match="without a weight column"):
        pagerank(plain, weighted=True)
    weighted_table = clean_edges(edges([("A", "B"), ("A", "C")], [3, 1]), weight_col="num_contexts")
    assert weighted_table.has_weights
    assert pagerank(weighted_table, weighted=True).scores["B"] == pytest.approx(0.425325, abs=1e-6)
    assert pagerank(weighted_table).scores["B"] == pytest.approx(0.370130, abs=1e-6)  # unweighted on a weighted table: ignores them
    assert pagerank(plain).scores["B"] == pytest.approx(0.370130, abs=1e-6)


def test_weighted_pagerank_is_deterministic_for_any_row_order():
    pairs = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A"), ("D", "A"), ("D", "C")]
    weights = [3, 1, 2, 5, 1, 4]
    reference = pagerank(edges(pairs, weights), weighted=True)
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(len(pairs))
        again = pagerank(edges([pairs[i] for i in order], [weights[i] for i in order]), weighted=True)
        pd.testing.assert_frame_equal(again.ranking, reference.ranking)
