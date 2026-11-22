"""Tests for p4.graph.leakage_audit (leakage-safe graph and popularity signals)."""
import json

import pandas as pd
import pytest

from p4.graph import leakage_audit as L

# queries: C1->T1 (train), C2->T1 (val), C3->T2 and C3->T3 (both test: one citing paper, two queries), C4->T2 (train)
QUERIES = pd.DataFrame({
    "query_id": ["q1", "q2", "q3", "q4", "q5"], "split": ["train", "val", "test", "test", "train"],
    "citing_paper_id": ["C1", "C2", "C3", "C3", "C4"], "cited_paper_id": ["T1", "T1", "T2", "T3", "T2"],
    "context_id": ["c1", "c2", "c3", "c4", "c5"]})
# the edges contain every query's edge, plus C5->T1 (no query) and C2->T4 (a second edge of a validation citing paper)
EDGES = pd.DataFrame({"citing_paper_id": ["C1", "C2", "C3", "C3", "C4", "C5", "C2"],
                      "cited_paper_id": ["T1", "T1", "T2", "T3", "T2", "T1", "T4"]})
PAPERS = pd.DataFrame({"paper_id": ["T1", "T2", "T3", "T4", "T5"]})


def test_splits_are_disjoint_by_citing_paper():
    a = L.split_audit(QUERIES)
    assert a["disjoint_by_citing_paper"] and a["citing_papers_in_more_than_one_split"] == 0
    assert a["queries_per_split"] == {"test": 2, "train": 2, "val": 1} and a["citing_papers_per_split"] == {"test": 1, "train": 2, "val": 1}
    leaky = pd.concat([QUERIES, pd.DataFrame({"query_id": ["q9"], "split": ["test"], "citing_paper_id": ["C2"],
                                              "cited_paper_id": ["T9"], "context_id": ["c9"]})])
    bad = L.split_audit(leaky)
    assert not bad["disjoint_by_citing_paper"] and bad["shared_examples"] == ["C2"]          # C2 now in val and test


def test_safe_edges_drop_every_edge_of_a_held_out_citing_paper():
    safe, stats = L.leakage_safe_edges(EDGES, QUERIES)
    assert sorted(zip(safe["citing_paper_id"], safe["cited_paper_id"])) == [("C1", "T1"), ("C4", "T2"), ("C5", "T1")]
    assert stats["held_out_citing_papers"] == 2 and stats["edges_total"] == 7
    assert (stats["edges_removed"], stats["edges_kept"]) == (4, 3) and stats["share_removed"] == pytest.approx(4 / 7)
    # C2's OTHER edge (C2->T4) goes too: all of a held-out citing paper's edges are removed, not only the query's own
    assert ("C2", "T4") not in set(zip(safe["citing_paper_id"], safe["cited_paper_id"]))


def test_hold_out_choice_and_no_held_out_queries():
    only_test, stats = L.leakage_safe_edges(EDGES, QUERIES, hold_out=("test",))
    assert stats["edges_removed"] == 2 and ("C2", "T1") in set(zip(only_test["citing_paper_id"], only_test["cited_paper_id"]))
    same, stats = L.leakage_safe_edges(EDGES, QUERIES[QUERIES.split == "train"])
    assert len(same) == len(EDGES) and stats["edges_removed"] == 0
    empty, stats = L.leakage_safe_edges(EDGES.iloc[0:0], QUERIES)
    assert empty.empty and stats["share_removed"] == 0.0


def test_citation_counts_count_distinct_citing_papers():
    counts = L.citation_counts(pd.concat([EDGES, EDGES.iloc[[0]]]))                              # a duplicate row changes nothing
    assert counts.to_dict() == {"T1": 3, "T2": 2, "T3": 1, "T4": 1}
    safe, _ = L.leakage_safe_edges(EDGES, QUERIES)
    assert L.citation_counts(safe).to_dict() == {"T1": 2, "T2": 1}                                # T3 and T4 lose all their citations


def test_the_target_gains_citations_from_its_own_query_in_the_full_graph():
    safe, _ = L.leakage_safe_edges(EDGES, QUERIES)
    full, clean = L.citation_counts(EDGES), L.citation_counts(safe)
    for _, q in QUERIES[QUERIES.split.isin(L.HELD_OUT)].iterrows():
        assert full.get(q["cited_paper_id"], 0) > clean.get(q["cited_paper_id"], 0)             # every held-out target loses its own citation


def test_expected_ranks_share_the_position_among_ties():
    scores = pd.Series({"a": 3.0, "b": 2.0, "c": 1.0, "d": 1.0, "e": 0.0})
    assert L._expected_ranks(scores, ["a", "b", "c", "d", "e", "missing"]).tolist() == [1, 2, 3.5, 3.5, 5, 5]


def test_popularity_prior_diagnostic_by_hand():
    safe, _ = L.leakage_safe_edges(EDGES, QUERIES)
    d = L.popularity_prior_diagnostic(QUERIES, PAPERS, EDGES, safe, "test", k=2)
    # test targets are T2 and T3. Full counts T1=3 T2=2 T3=1 T4=1 T5=0 -> ranks 2 and 3.5; safe counts T1=2 T2=1, rest 0 -> ranks 2 and 4
    assert d["n_queries"] == 2 and d["n_papers"] == 5
    assert d["full_graph"]["mrr"] == pytest.approx((1 / 2 + 1 / 3.5) / 2) and d["safe_graph"]["mrr"] == pytest.approx((1 / 2 + 1 / 4) / 2)
    assert d["full_graph"]["recall_at_k"] == 0.5 and d["safe_graph"]["recall_at_k"] == 0.5
    assert d["full_graph"]["targets_with_zero_citations"] == 0.0 and d["safe_graph"]["targets_with_zero_citations"] == 0.5
    assert d["full_graph"]["mrr"] > d["safe_graph"]["mrr"]                                         # the leak helps the prior


def test_same_citing_paper_index_contexts():
    index = pd.DataFrame({"context_id": ["x1", "x2"], "citing_paper_id": ["C3", "C9"], "cited_paper_id": ["T2", "T3"]})
    r = L.same_citer_index_contexts(QUERIES, index, "test")
    assert r["n_queries"] == 2 and r["queries_with_other_sentences_by_same_citing_paper"] == 1 and r["share"] == 0.5   # q3 yes, q4 no


def test_audit_on_a_processed_folder_and_cli(tmp_path, capsys):
    folder = tmp_path / "processed"
    folder.mkdir()
    papers = PAPERS.assign(citation_count=PAPERS["paper_id"].map(L.citation_counts(EDGES)).fillna(0).astype(int))
    papers.to_parquet(folder / "papers.parquet", index=False)
    QUERIES.to_parquet(folder / "queries.parquet", index=False)
    EDGES.assign(num_contexts=1).to_parquet(folder / "edges.parquet", index=False)
    pd.DataFrame({"context_id": ["x"], "citing_paper_id": ["C3"], "cited_paper_id": ["T2"]}).to_parquet(folder / "index_contexts.parquet", index=False)
    report = L.audit(folder, k=2)
    assert report["split_audit"]["disjoint_by_citing_paper"]
    assert report["held_out_queries_whose_own_edge_is_in_the_full_graph"] == 1.0                    # every held-out query's edge is in the graph
    assert report["papers_citation_count_equals_full_graph_in_degree"]
    assert report["edge_removal_for_the_evaluation_graph"]["edges_removed"] == 4
    assert report["held_out_targets"]["mean_citation_count_full"] > report["held_out_targets"]["mean_citation_count_safe"]
    out = tmp_path / "report.json"
    assert L.main(["--processed", str(folder), "--out", str(out)]) == 0 and json.loads(out.read_text())["split_audit"]["queries"] == 5
    assert L.main(["--processed", str(folder), "--out", str(out)]) == 2 and "already exists" in capsys.readouterr().err
    assert L.main(["--processed", str(tmp_path / "nope"), "--out", str(tmp_path / "o2.json")]) == 2


# ---- gate for the context files given to P2/P3 -------------------------------------------------------------------
def write_contexts(path, rows):
    path.write_text("".join(json.dumps(r) + chr(10) for r in rows))
    return path


INDEX = pd.DataFrame({"context_id": ["x1", "x2", "x3"], "citing_paper_id": ["C3", "C9", "C1"], "cited_paper_id": ["T2", "T3", "T1"]})
QUERIES_WITH_TEXT = QUERIES.assign(text=["sentence one", "sentence two", "sentence three", "sentence four", "sentence five"])


def test_a_clean_context_file_passes(tmp_path):
    rows = [{"context_id": "x1", "cited_paper_id": "T2", "sentence": "another description"},
            {"context_id": "x2", "cited_paper_id": "T3", "sentence": "something else"}]
    r = L.audit_context_file(write_contexts(tmp_path / "c.jsonl", rows), INDEX, QUERIES_WITH_TEXT)
    assert r["ok"] and r["rows"] == 2 and r["index_contexts_missing_from_file"] == 1            # x3 is in the index but not in this file
    assert (r["context_ids_not_in_index"], r["query_context_ids"], r["sentences_equal_to_a_query_text"]) == (0, 0, 0)


def test_each_kind_of_leak_in_a_context_file_is_detected(tmp_path):
    rows = [{"context_id": "x1", "cited_paper_id": "T2", "sentence": "fine"},
            {"context_id": "c3", "cited_paper_id": "T2", "sentence": "this is a query context id"},          # a query's own context
            {"context_id": "zzz", "cited_paper_id": "T2", "sentence": "not in the index at all"},
            {"context_id": "x2", "cited_paper_id": "T3", "sentence": "Sentence THREE!"}]                       # equals a query text after normalising
    r = L.audit_context_file(write_contexts(tmp_path / "c.jsonl", rows), INDEX, QUERIES_WITH_TEXT)
    assert not r["ok"] and r["query_context_ids"] == 1 and r["context_ids_not_in_index"] == 2          # c3 and zzz
    assert r["sentences_equal_to_a_query_text"] == 1


def test_duplicate_context_ids_and_blank_lines_are_handled(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"context_id": "x1", "sentence": "a"}) + chr(10) + chr(10) + json.dumps({"context_id": "x1", "sentence": "b"}) + chr(10))
    r = L.audit_context_file(path, INDEX, QUERIES_WITH_TEXT)
    assert r["rows"] == 2 and r["duplicated_context_ids"] == 1 and r["ok"]


def test_audit_runs_the_context_file_gate_through_the_processed_folder_and_cli(tmp_path):
    from p4.tests import graph_fixtures as GF
    folder = GF.small_processed(tmp_path / "processed")
    pd.DataFrame({"paper_id": ["T1", "T2", "T3", "T4", "T5"], "citation_count": [3, 2, 1, 1, 0]}).to_parquet(folder / "papers.parquet", index=False)
    pd.read_parquet(folder / "index_contexts.parquet").assign(sentence="").to_parquet(folder / "index_contexts.parquet", index=False)
    good = write_contexts(tmp_path / "good.jsonl", [{"context_id": "x1", "cited_paper_id": "T2", "sentence": "ok"}])
    bad = write_contexts(tmp_path / "bad.jsonl", [{"context_id": "c3", "cited_paper_id": "T2", "sentence": "ok"}])
    report = L.audit(folder, context_files=[good, bad])
    assert [f["ok"] for f in report["context_files"]] == [True, False]
    out = tmp_path / "report.json"
    assert L.main(["--processed", str(folder), "--out", str(out), "--context-file", str(good)]) == 0
    assert json.loads(out.read_text())["context_files"][0]["ok"] is True
