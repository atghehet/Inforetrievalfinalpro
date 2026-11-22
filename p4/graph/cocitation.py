"""Co-citation recommender.

Definition: two papers are CO-CITED when one citing paper cites both of them. The co-citation count of a
pair (x, y) is the number of distinct citing papers that cite both. Papers that are often cited together
tend to be about related topics, so for a given paper the recommendations are the papers with the highest
co-citation count with it. (This uses only the citation structure, not any text.)

Computation
    Let A be the 0/1 matrix with one row per citing paper c and one column per cited paper p, A[c, p] = 1 if
    c cites p. Then   Co = A^T A   has Co[x, y] = number of citing papers that cite both x and y.
    The diagonal Co[x, x] is just x's citation count and is set to zero (a paper is not co-cited with itself).
    Self-citations and duplicate citations are removed first (see edges.py), so a paper never counts itself and a
    citing paper counts once per pair.

Complexity
    Building:   O(sum over citing papers c of r_c^2) time, where r_c is the number of references of c, because
                a paper citing r references creates r(r-1)/2 pairs. Memory O(number of co-cited pairs).
                A few papers with hundreds of references can dominate this cost: `max_references` skips them.
    Recommend:  O(k_x log k_x) for a paper with k_x co-cited neighbours (a lookup in the stored matrix row).

Ordering: highest count first; ties broken by paper_id (ascending), so results are deterministic.

Errors: asking about a paper the index has never seen raises UnknownPaperError. A paper that is known but has
    no co-cited neighbours (for example it is cited only by papers that cite nothing else in the corpus)
    returns an empty table.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .edges import EdgeStats, EdgeTable, as_edge_table

COLUMNS = ["paper_id", "cocitation_count", "rank"]


class UnknownPaperError(ValueError):
    """The paper is not in the co-citation index (it appears in no edge and was not given in `nodes`)."""


@dataclass(frozen=True)
class CoCitationStats:
    n_papers: int                 # papers in the index: every paper an edge mentions (citing or cited) plus `nodes`
    n_citing_papers: int          # citing papers that contributed (after any max_references skip)
    n_citing_skipped: int         # citing papers skipped because they have more than max_references references
    n_pairs: int                  # distinct co-cited pairs (each unordered pair counted once)
    edge_stats: EdgeStats | None = None


class CoCitationIndex:
    """Co-citation counts for every pair of papers, built once from an edge table."""

    def __init__(self, ids: np.ndarray, matrix: sparse.csr_matrix, stats: CoCitationStats):
        self.ids = ids                                         # sorted paper IDs (object array)
        self._index = pd.Index(ids)
        self._matrix = matrix                                  # symmetric, zero diagonal, integer counts
        self.stats = stats

    @classmethod
    def from_edges(cls, edges, *, nodes=None, max_references: int | None = None, **column_options) -> "CoCitationIndex":
        """Build the index.

        edges           an EdgeTable, an edge DataFrame (citing_paper_id, cited_paper_id) or a path to edges.parquet
        nodes           optional iterable of ALL paper IDs, so papers without edges are known (they get an empty
                        recommendation list instead of UnknownPaperError). Edge endpoints must all be among them
        max_references  ignore citing papers that cite more than this many papers (None = use all)
        """
        if max_references is not None and (not isinstance(max_references, (int, np.integer)) or max_references < 2):
            raise ValueError(f"max_references must be a whole number >= 2 (or None), got {max_references!r}")
        table: EdgeTable = as_edge_table(edges, **column_options)
        df = table.edges
        endpoints = set(df["source"]) | set(df["target"])
        if nodes is not None:
            universe = {str(n) for n in nodes}
            unknown = sorted(endpoints - universe)
            if unknown:
                raise ValueError(f"{len(unknown)} edge endpoint(s) are not in `nodes`, e.g. {unknown[:3]}")
        else:
            universe = endpoints
        ids = np.array(sorted(universe), dtype=object)
        n = len(ids)
        index = pd.Index(ids)

        citing = pd.Index(sorted(set(df["source"])))
        rows = citing.get_indexer(df["source"])
        cols = index.get_indexer(df["target"])
        references = np.bincount(rows, minlength=len(citing))             # references per citing paper
        keep_row = np.ones(len(citing), dtype=bool) if max_references is None else references <= max_references
        keep = keep_row[rows]
        incidence = sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.int64), (rows[keep], cols[keep])),
                                      shape=(len(citing), n))
        co = (incidence.T @ incidence).tocsr()                            # co[x, y] = citing papers citing both
        co.setdiag(0)
        co.eliminate_zeros()
        stats = CoCitationStats(n_papers=n, n_citing_papers=int(keep_row.sum()),
                                n_citing_skipped=int((~keep_row).sum()), n_pairs=int(co.nnz // 2),
                                edge_stats=table.stats)
        return cls(ids, co, stats)

    def __contains__(self, paper_id) -> bool:
        return str(paper_id) in self._index

    def _position(self, paper_id) -> int:
        key = str(paper_id)
        if key not in self._index:
            raise UnknownPaperError(f"paper {key!r} is not in the co-citation index "
                                    f"({len(self.ids)} papers); check the ID")
        return int(self._index.get_loc(key))

    def count(self, paper_a, paper_b) -> int:
        """Number of citing papers that cite both. 0 for a paper paired with itself (never co-cited with itself)."""
        a, b = self._position(paper_a), self._position(paper_b)
        return int(self._matrix[a, b])

    def recommend(self, paper_id, top_k: int = 10) -> pd.DataFrame:
        """The papers most often cited together with `paper_id`, as a table paper_id, cocitation_count, rank.

        The paper itself is never included. Empty table (same columns) if it has no co-cited neighbours.
        Raises UnknownPaperError for a paper the index does not know.
        """
        if not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError(f"top_k must be a whole number >= 1, got {top_k!r}")
        pos = self._position(paper_id)
        start, end = self._matrix.indptr[pos], self._matrix.indptr[pos + 1]
        neighbours, counts = self._matrix.indices[start:end], self._matrix.data[start:end]
        order = np.lexsort((self.ids[neighbours], -counts))               # count descending, then paper_id ascending
        order = order[:top_k]
        return pd.DataFrame({"paper_id": self.ids[neighbours[order]], "cocitation_count": counts[order].astype(int),
                             "rank": np.arange(1, len(order) + 1)}, columns=COLUMNS)
