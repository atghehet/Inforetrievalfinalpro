"""PageRank on the citation graph.

Idea: a paper is important if important papers cite it. Imagine a reader who starts on a random paper and
keeps doing one of two things: with probability d (the damping factor, 0.85) follow a random citation
of the current paper; with probability 1 - d jump to a completely random paper.
The PageRank of a paper is the long-run fraction of time the reader spends on it.

Direction of propagation
    An edge points from the CITING paper to the CITED paper (P1's schema: citing_paper_id -> cited_paper_id).
    Score flows ALONG the edge: a paper passes its score on to the papers it cites. So a paper gains rank by
    being cited, especially by papers that are themselves highly ranked. (A paper that cites many others
    splits its score between them.)

Formula (N papers, r the vector of scores, out(j) the total weight of j's outgoing edges)
    r_i = (1 - d) / N  +  d * [ sum over papers j citing i of  w(j->i) / out(j) * r_j   +   D / N ]
    where D = sum of r_j over "dangling" papers (those that cite nothing, out(j) = 0).
    Dangling papers spread their score evenly over all N papers (otherwise their score would vanish).
    Isolated papers (no edges at all) are simply dangling papers nobody cites: they receive the
    (1 - d)/N teleport share plus an equal share of the dangling mass.

Computation: power iteration, r <- formula(r), starting from the uniform vector, until the L1 change
    sum_i |r_new_i - r_i| < tol or max_iter iterations. Each iteration is one sparse matrix-vector product.
    Time O(iterations * (N + E)), memory O(N + E). Scores sum to 1.

Ranking: by score (highest first), then by paper_id (ascending). Scores closer than 1e-12 count as tied, so
    floating-point noise cannot change the order from one machine to another.

Edge weights: unweighted by default (every citation counts once). With weights (P1's num_contexts), a paper
    splits its score in proportion to the weights of the citations it makes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .edges import WEIGHT_COLUMN, EdgeStats, EdgeTable, as_edge_table

DEFAULT_DAMPING = 0.85
DEFAULT_TOL = 1e-10
DEFAULT_MAX_ITER = 100
TIE_DECIMALS = 12          # scores equal up to 1e-12 are ties and are ordered by paper_id

COLUMNS = ["paper_id", "score", "score_scaled", "rank"]


@dataclass
class PageRankResult:
    ranking: pd.DataFrame       # top_k rows (or all), columns: paper_id, score, score_scaled, rank
    scores: pd.Series           # every paper (index = paper_id, sorted by id): score summing to 1
    n_nodes: int
    n_edges: int                # after cleaning
    n_dangling: int             # papers that cite nothing (including isolated papers)
    iterations: int
    converged: bool
    final_delta: float          # L1 change in the last iteration
    damping: float
    tol: float
    edge_stats: EdgeStats | None = None


def _check_parameters(damping: float, tol: float, max_iter: int, top_k: int | None) -> None:
    if not 0 <= damping < 1:
        raise ValueError(f"damping must be in [0, 1), got {damping!r}")
    if not tol > 0:
        raise ValueError(f"tol must be positive, got {tol!r}")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a whole number >= 1, got {max_iter!r}")
    if top_k is not None and (not isinstance(top_k, (int, np.integer)) or top_k < 1):
        raise ValueError(f"top_k must be a whole number >= 1 (or None for all), got {top_k!r}")


def _rank_table(ids: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Order by score (high first), ties by paper_id; add the scaled score and the 1-based rank."""
    order = np.lexsort((ids, -np.round(scores, TIE_DECIMALS)))     # last key is the primary one
    ordered = scores[order]
    top = ordered.max() if len(ordered) else 1.0
    return pd.DataFrame({"paper_id": ids[order], "score": ordered,
                         "score_scaled": ordered / top if top > 0 else ordered,
                         "rank": np.arange(1, len(ordered) + 1)}, columns=COLUMNS)


def pagerank(edges, *, nodes=None, damping: float = DEFAULT_DAMPING, tol: float = DEFAULT_TOL,
             max_iter: int = DEFAULT_MAX_ITER, weighted: bool = False, top_k: int | None = None,
             **column_options) -> PageRankResult:
    """Compute PageRank.

    edges    an EdgeTable (p4.graph.edges.read_edges), an edge DataFrame (citing_paper_id, cited_paper_id,
             and num_contexts if weighted) or a path to edges.parquet
    nodes    optional iterable of ALL paper IDs (e.g. papers.parquet), so papers without any edge are
             included. Every edge endpoint must be among them, else ValueError. Default: papers seen in edges
    damping  probability of following a citation instead of jumping to a random paper (default 0.85)
    tol      stop when the L1 change between iterations is below this (default 1e-10)
    max_iter iteration limit (default 100); check `converged` in the result
    weighted use the edge weights (num_contexts) instead of counting each citation once
    top_k    keep only the best k rows in `ranking` (`scores` always holds every paper)

    An empty graph (no papers at all) returns an empty result. Papers but no edges: every score is 1/N.
    """
    _check_parameters(damping, tol, max_iter, top_k)
    if isinstance(edges, EdgeTable):
        table = edges            # already cleaned
        if weighted and not table.has_weights:
            raise ValueError("weighted=True needs edge weights, but this EdgeTable was read without a weight column; "
                             "use read_edges(path, weight_col='num_contexts') or pass the path / DataFrame instead")
    else:
        options = dict(column_options)
        if weighted:
            options.setdefault("weight_col", WEIGHT_COLUMN)      # P1's num_contexts
        table = as_edge_table(edges, **options)
    edge_df = table.edges
    endpoints = set(edge_df["source"]) | set(edge_df["target"])
    if nodes is not None:
        universe = {str(n) for n in nodes}
        unknown = sorted(endpoints - universe)
        if unknown:
            raise ValueError(f"{len(unknown)} edge endpoint(s) are not in `nodes`, e.g. {unknown[:3]}")
    else:
        universe = endpoints
    ids = np.array(sorted(universe), dtype=object)
    n = len(ids)
    if n == 0:
        empty = pd.DataFrame({c: [] for c in COLUMNS})
        return PageRankResult(empty, pd.Series(dtype=float, name="score"), 0, 0, 0, 0, True, 0.0, damping, tol, table.stats)

    index = pd.Index(ids)
    src = index.get_indexer(edge_df["source"])
    dst = index.get_indexer(edge_df["target"])
    weight = edge_df["weight"].to_numpy(float) if weighted else np.ones(len(edge_df))
    out_weight = np.bincount(src, weights=weight, minlength=n)
    dangling = out_weight == 0
    # transition matrix: column j holds the share of j's score that goes to each paper it cites
    transition = sparse.csr_matrix((weight / out_weight[src], (dst, src)), shape=(n, n))

    rank = np.full(n, 1.0 / n)
    delta, converged, iterations = float("inf"), False, 0
    for iterations in range(1, max_iter + 1):
        dangling_mass = rank[dangling].sum()
        new = damping * (transition @ rank + dangling_mass / n) + (1.0 - damping) / n
        delta = float(np.abs(new - rank).sum())
        rank = new
        if delta < tol:
            converged = True
            break
    rank = rank / rank.sum()                                  # remove accumulated rounding drift: scores sum to 1

    table_all = _rank_table(ids, rank)
    shown = table_all if top_k is None else table_all.head(int(top_k))
    return PageRankResult(
        ranking=shown.reset_index(drop=True), scores=pd.Series(rank, index=index, name="score"),
        n_nodes=n, n_edges=len(edge_df), n_dangling=int(dangling.sum()), iterations=iterations,
        converged=converged, final_delta=delta, damping=damping, tol=tol, edge_stats=table.stats)
