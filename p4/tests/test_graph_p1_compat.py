"""The graph code must work on edge tables exactly as P1's pipeline produces them.

Fixture: P1's own SYNTHETIC sample (scripts/make_sample.py, random text, fixed seed) pushed through P1's build_tables,
in memory and read-only. This checks the schema and the algorithms on a realistically shaped graph (hundreds of
papers, Zipf-like popularity). The numbers it produces are NOT results about real citations.
"""
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from p4.graph.cocitation import CoCitationIndex
from p4.graph.edges import read_edges
from p4.graph.pagerank import pagerank

REPO = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def p1_tables():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import make_sample
        from citeds import schema
        from citeds.tables import build_tables
        from citeds.unarxive import parse_record
    except ImportError as err:                                          # pragma: no cover
        pytest.skip(f"P1 code not importable: {err}")
    records = make_sample.make(400, seed=0)
    parsed = [parse_record(r) for r in records if "cs.CL" in r["metadata"]["categories"].split()]
    papers, _contexts, edges, _log = build_tables(parsed)
    return papers, edges, schema


def test_p1_edge_columns_are_what_the_loader_expects(p1_tables):
    _papers, edges, schema = p1_tables
    assert list(edges.columns) == schema.EDGES.names == ["citing_paper_id", "cited_paper_id", "num_contexts"]
    assert len(edges) > 200


def test_parquet_written_with_p1s_schema_round_trips_through_the_loader(p1_tables, tmp_path):
    _papers, edges, schema = p1_tables
    path = tmp_path / "edges.parquet"
    pq.write_table(schema.conform(edges, schema.EDGES), path)           # exactly how citeds.build writes it
    table = read_edges(path, weight_col="num_contexts")
    assert len(table.edges) == len(edges)                                # P1 already removed duplicates and self-citations
    assert table.stats.n_self_loops == 0 and table.stats.n_duplicates == 0
    assert table.edges["weight"].sum() == edges["num_contexts"].sum()


def test_pagerank_on_a_p1_shaped_graph_matches_an_exact_solution(p1_tables):
    papers, edges, _ = p1_tables
    ids = papers["paper_id"].tolist()                                    # includes papers that neither cite nor are cited
    result = pagerank(edges, nodes=ids)
    n = len(ids)
    position = {k: i for i, k in enumerate(ids)}
    P, out = np.zeros((n, n)), np.zeros(n)
    for s, t in zip(edges["citing_paper_id"], edges["cited_paper_id"]):
        P[position[t], position[s]] += 1
        out[position[s]] += 1
    for j in range(n):
        P[:, j] = P[:, j] / out[j] if out[j] else 1.0 / n
    exact = np.linalg.solve(np.eye(n) - 0.85 * P, np.full(n, 0.15 / n))
    assert result.n_nodes == n and result.converged
    assert result.scores.loc[ids].to_numpy() == pytest.approx(exact, abs=1e-8)
    assert result.scores.sum() == pytest.approx(1.0)
    # more citations should mean (on average) a higher score: the cited paper gains rank from every citation it receives
    cited_count = edges.groupby("cited_paper_id").size().reindex(ids).fillna(0).to_numpy()
    assert np.corrcoef(cited_count, result.scores.loc[ids].to_numpy())[0, 1] > 0.5


def test_weighted_and_unweighted_pagerank_both_run_on_p1_edges(p1_tables):
    papers, edges, _ = p1_tables
    plain = pagerank(edges, nodes=papers["paper_id"])
    weighted = pagerank(edges, nodes=papers["paper_id"], weighted=True)
    assert weighted.converged and weighted.scores.sum() == pytest.approx(1.0)
    assert plain.n_edges == weighted.n_edges


def test_cocitation_on_a_p1_shaped_graph_matches_brute_force(p1_tables):
    papers, edges, _ = p1_tables
    index = CoCitationIndex.from_edges(edges, nodes=papers["paper_id"])
    references = edges.groupby("citing_paper_id")["cited_paper_id"].apply(sorted)
    expected = Counter()
    for cited in references:
        expected.update(itertools.combinations(cited, 2))
    assert index.stats.n_pairs == len(expected)
    most_cited = edges["cited_paper_id"].value_counts().index[0]
    rec = index.recommend(most_cited, top_k=5)
    neighbours = sorted(((-n, (a if b == most_cited else b)) for (a, b), n in expected.items() if most_cited in (a, b)))[:5]
    assert list(zip(-rec["cocitation_count"], rec["paper_id"])) == neighbours
    assert most_cited not in rec["paper_id"].tolist()
