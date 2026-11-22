"""Tests for p4.analysis.data: reading P1's tables and joining them per query."""
import numpy as np
import pandas as pd
import pytest

from p4.analysis import data as D
from p4.tests import analysis_fixtures as F


def test_data_labels_are_checked():
    for label in ("real", "synthetic", "demo"):
        assert D.check_data_label(label) == label
    for bad in ("Real", "fake", "", None):
        with pytest.raises(D.AnalysisDataError, match="data label must be one of"):
            D.check_data_label(bad)


def test_loaders_read_p1_parquet_files(tmp_path):
    paths = F.write_files(tmp_path)
    assert len(D.load_queries(paths["queries"])) == 40 and "cited_citation_count" in D.load_queries(paths["queries"]).columns
    assert D.load_papers(paths["papers"]).shape[0] == 50
    assert set(D.load_functions(paths["functions"])["function"]) == {"background", "method", "result"}
    counts = D.load_index_context_counts(paths["index_contexts"])
    assert counts["T01"] == 1 and counts["T15"] == 8 and counts["T40"] == 30


def test_loaders_read_csv_and_keep_ids_as_text(tmp_path):
    (tmp_path / "p.csv").write_text("paper_id,title,year,citation_count\n1801.10000,t,2018,4\n")
    (tmp_path / "q.csv").write_text("query_id,cited_paper_id\nq000001,1801.10000\n")
    assert D.load_papers(tmp_path / "p.csv").loc[0, "paper_id"] == "1801.10000"          # not 1801.1
    assert D.load_queries(tmp_path / "q.csv").loc[0, "cited_paper_id"] == "1801.10000"


@pytest.mark.parametrize("loader, content, message", [
    (D.load_queries, "query_id\nq1\n", "missing column"),
    (D.load_papers, "title\nx\n", "missing column"),
    (D.load_functions, "context_id,function\nc1,opinion\n", "unknown function label"),
    (D.load_papers, "paper_id\na\na\n", "duplicate paper_id"),
])
def test_bad_tables_are_rejected_with_clear_messages(tmp_path, loader, content, message):
    path = tmp_path / "t.csv"
    path.write_text(content)
    with pytest.raises(D.AnalysisDataError, match=message):
        loader(path)


def test_file_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="queries file not found"):
        D.load_queries(tmp_path / "nope.parquet")
    (tmp_path / "x.txt").write_text("a")
    with pytest.raises(D.AnalysisDataError, match="unsupported papers file type"):
        D.load_papers(tmp_path / "x.txt")


def test_query_table_joins_everything_by_hand():
    table = D.build_query_table(F.make_qrels(), F.make_queries(), F.make_papers(), F.make_functions(), F.make_index_counts())
    assert len(table) == 40 and table.index.is_unique
    row = table.loc["q07"]
    assert row["target_id"] == "T07" and row["citation_count"] == 3 + 6 % 5 and row["year"] == 2019
    assert row["text"] == "query sentence 7" and row["title"] == "Paper T07"
    assert row["function"] == "method" and row["function_confidence"] == 0.9 and row["n_index_contexts"] == 1
    assert table.loc["q30", "citation_count"] == 70 and table.loc["q30", "n_index_contexts"] == 30
    assert table.loc["q01", "cited_citation_count"] == table.loc["q01", "citation_count"]


def test_missing_metadata_is_nan_not_dropped():
    qrels = {**F.make_qrels(), "q99": {"UNKNOWN": 1}}                         # a judged query that no table knows
    papers = F.make_papers().assign(year=lambda d: d["year"].astype(float))
    papers.loc[papers.paper_id == "T01", "year"] = np.nan                     # a paper with unknown year
    table = D.build_query_table(qrels, F.make_queries(), papers)
    assert len(table) == 41
    assert pd.isna(table.loc["q99", "text"]) and pd.isna(table.loc["q99", "citation_count"]) and not table.loc["q99", "target_in_papers"]
    assert pd.isna(table.loc["q01", "year"]) and table.loc["q01", "citation_count"] == 3
    assert table["function"].isna().all() and table["n_index_contexts"].isna().all()   # not given, so empty (not invented)


def test_citation_age_is_citing_year_minus_target_year():
    papers = pd.DataFrame({"paper_id": ["T", "C"], "year": [2018, 2022], "citation_count": [5, 0]})
    queries = pd.DataFrame({"query_id": ["q1"], "cited_paper_id": ["T"], "citing_paper_id": ["C"]})
    table = D.build_query_table({"q1": {"T": 1}}, queries, papers)
    assert table.loc["q1", "citing_year"] == 2022 and table.loc["q1", "citation_age"] == 4


def test_dataset_stats_counts_problems_explicitly():
    papers = F.make_papers()
    queries = F.make_queries()
    queries.loc[queries.query_id == "q02", "cited_paper_id"] = "T99"          # disagrees with the qrels
    queries.loc[queries.query_id == "q03", "cited_citation_count"] = 99       # disagrees with papers.citation_count
    qrels = {**F.make_qrels(), "q99": {"UNKNOWN": 1}}
    table = D.build_query_table(qrels, queries, papers, F.make_functions(), F.make_index_counts())
    stats = D.dataset_stats(table, papers, have_queries=True, have_functions=True, have_index_counts=True)
    assert stats["judged queries"] == 41 and stats["papers table rows"] == 50
    assert stats["papers with citation_count 0"] == 5
    assert stats["queries without a row in the queries file"] == 1 and stats["targets not found in papers"] == 1
    assert stats["queries whose target differs between qrels and queries file"] == 1
    assert stats["queries where cited_citation_count != papers.citation_count"] == 1
    assert stats["queries without a citation-function label"] == 1             # q99 has no context id
    assert stats["queries whose target has 0 indexable contexts"] == 1         # the unknown paper


def test_dataset_stats_says_when_an_input_was_not_given():
    table = D.build_query_table(F.make_qrels(), None, None)
    stats = D.dataset_stats(table, None, have_queries=False, have_functions=False, have_index_counts=False)
    assert stats["queries without a row in the queries file"] == "queries file not given"
    assert stats["targets not found in papers"] == "papers file not given"
    assert stats["queries without a citation-function label"] == "context_functions file not given"
    assert "queries whose target has 0 indexable contexts" not in stats
