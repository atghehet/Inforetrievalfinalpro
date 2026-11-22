"""Tests for p4.graph.edges: reading and cleaning the citation edge table (P1's edges.parquet schema)."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from p4.graph.edges import EdgeFormatError, EdgeTable, as_edge_table, clean_edges, read_edges, read_paper_ids

P1_EDGES_SCHEMA = pa.schema([("citing_paper_id", pa.string()), ("cited_paper_id", pa.string()),
                             ("num_contexts", pa.int32())])            # copied from citeds/schema.py EDGES


def frame(rows, columns=("citing_paper_id", "cited_paper_id", "num_contexts")):
    return pd.DataFrame(rows, columns=list(columns))


def test_clean_edges_keeps_valid_edges_sorted_with_unit_weights():
    table = clean_edges(frame([("B", "C", 1), ("A", "B", 1), ("A", "C", 2)]))
    assert list(zip(table.edges["source"], table.edges["target"])) == [("A", "B"), ("A", "C"), ("B", "C")]
    assert table.edges["weight"].tolist() == [1.0, 1.0, 1.0]                 # unweighted unless asked
    assert (table.stats.n_input, table.stats.n_edges) == (3, 3)


def test_self_citations_are_removed_and_counted():
    table = clean_edges(frame([("A", "A", 1), ("A", "B", 1), ("B", "B", 4)]))
    assert list(zip(table.edges["source"], table.edges["target"])) == [("A", "B")]
    assert table.stats.n_self_loops == 2


def test_duplicate_edges_are_merged_and_their_weights_added():
    rows = [("A", "B", 1), ("A", "B", 2), ("A", "B", 3), ("A", "C", 1)]
    unweighted = clean_edges(frame(rows))
    assert len(unweighted.edges) == 2 and unweighted.stats.n_duplicates == 2
    assert unweighted.edges["weight"].tolist() == [3.0, 1.0]                  # three rows of A->B counted as weight 3
    weighted = clean_edges(frame(rows), weight_col="num_contexts")
    assert weighted.edges["weight"].tolist() == [6.0, 1.0]                    # 1 + 2 + 3


def test_missing_and_blank_endpoints_are_dropped_and_counted():
    table = clean_edges(frame([("A", "B", 1), (None, "B", 1), ("A", None, 1), ("", "B", 1), ("A", "  ", 1)]))
    assert len(table.edges) == 1 and table.stats.n_missing == 4 and table.stats.n_input == 5


def test_non_string_ids_are_converted_to_strings():
    table = clean_edges(frame([(1, 2, 1), (2, 3, 1)]))
    assert table.edges["source"].tolist() == ["1", "2"] and table.edges["target"].tolist() == ["2", "3"]


def test_empty_table_is_fine():
    table = clean_edges(frame([]))
    assert table.edges.empty and list(table.edges.columns) == ["source", "target", "weight"] and table.stats.n_edges == 0


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), "x"])
def test_invalid_weights_are_rejected(bad):
    with pytest.raises(EdgeFormatError, match="positive finite"):
        clean_edges(frame([("A", "B", 1), ("A", "C", bad)]), weight_col="num_contexts")


def test_missing_columns_are_reported_with_the_columns_found():
    with pytest.raises(EdgeFormatError, match="no source column 'citing_paper_id'.*\\['a', 'b'\\]"):
        clean_edges(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(EdgeFormatError, match="no weight column 'num_contexts'"):
        clean_edges(frame([("A", "B", 1)], columns=("citing_paper_id", "cited_paper_id", "other")), weight_col="num_contexts")


def test_custom_column_names_are_supported():
    df = pd.DataFrame({"from": ["A"], "to": ["B"], "n": [5]})
    table = clean_edges(df, source_col="from", target_col="to", weight_col="n")
    assert table.edges.to_dict("records") == [{"source": "A", "target": "B", "weight": 5.0}]


# ---- files --------------------------------------------------------------------------------------
def write_p1_parquet(path, rows):
    cols = list(zip(*rows)) if rows else [[], [], []]
    pq.write_table(pa.table({"citing_paper_id": pa.array(cols[0], pa.string()),
                             "cited_paper_id": pa.array(cols[1], pa.string()),
                             "num_contexts": pa.array(cols[2], pa.int32())}, schema=P1_EDGES_SCHEMA), path)
    return path


def test_reads_a_parquet_file_with_the_p1_schema(tmp_path):
    path = write_p1_parquet(tmp_path / "edges.parquet", [("2106.09685", "1902.00751", 2), ("2106.09685", "1810.04805", 1)])
    table = read_edges(path)
    assert table.edges["source"].tolist() == ["2106.09685", "2106.09685"]
    assert table.edges["target"].tolist() == ["1810.04805", "1902.00751"]
    assert read_edges(path, weight_col="num_contexts").edges["weight"].tolist() == [1.0, 2.0]


def test_a_p1_edge_file_with_zero_rows_gives_an_empty_table(tmp_path):
    # P1's real sample build (49 papers, 0 in-corpus citations) wrote exactly such a file.
    table = read_edges(write_p1_parquet(tmp_path / "edges.parquet", []))
    assert table.edges.empty and table.stats.n_edges == 0


def test_extra_parquet_columns_are_ignored(tmp_path):
    pq.write_table(pa.table({"citing_paper_id": ["A"], "cited_paper_id": ["B"], "junk": [1]}), tmp_path / "e.parquet")
    assert len(read_edges(tmp_path / "e.parquet").edges) == 1


def test_reads_csv_and_tsv_keeping_ids_as_text(tmp_path):
    (tmp_path / "e.csv").write_text("citing_paper_id,cited_paper_id\n0001.00001,2106.09685\n")
    (tmp_path / "e.tsv").write_text("citing_paper_id\tcited_paper_id\n0001.00001\t2106.09685\n")
    for name in ("e.csv", "e.tsv"):
        assert read_edges(tmp_path / name).edges.loc[0, "source"] == "0001.00001"     # not 1.00001


def test_file_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="edge file not found"):
        read_edges(tmp_path / "nope.parquet")
    (tmp_path / "e.txt").write_text("a b")
    with pytest.raises(EdgeFormatError, match="unsupported edge file type"):
        read_edges(tmp_path / "e.txt")
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(EdgeFormatError, match="cannot read edges"):
        read_edges(tmp_path / "empty.csv")
    pq.write_table(pa.table({"a": [1], "b": [2]}), tmp_path / "wrong.parquet")
    with pytest.raises(EdgeFormatError, match="no source column"):
        read_edges(tmp_path / "wrong.parquet")


def test_as_edge_table_accepts_tables_frames_and_paths(tmp_path):
    table = clean_edges(frame([("A", "B", 1)]))
    assert as_edge_table(table) is table
    assert as_edge_table(frame([("A", "B", 1)])).stats.n_edges == 1
    path = write_p1_parquet(tmp_path / "e.parquet", [("A", "B", 1)])
    assert isinstance(as_edge_table(path), EdgeTable) and isinstance(as_edge_table(str(path)), EdgeTable)


def test_read_paper_ids_reads_ids_and_optional_titles(tmp_path):
    pq.write_table(pa.table({"paper_id": ["B", "A", "A", None], "title": ["tb", "ta", "dup", "x"], "year": [1, 2, 3, 4]}),
                   tmp_path / "papers.parquet")
    papers = read_paper_ids(tmp_path / "papers.parquet")
    assert papers["paper_id"].tolist() == ["B", "A"] and papers["title"].tolist() == ["tb", "ta"]   # null and duplicate dropped
    (tmp_path / "p.csv").write_text("paper_id\n0001.00001\n")
    assert read_paper_ids(tmp_path / "p.csv")["paper_id"].tolist() == ["0001.00001"]
    pq.write_table(pa.table({"x": [1]}), tmp_path / "bad.parquet")
    with pytest.raises(EdgeFormatError, match="no column 'paper_id'"):
        read_paper_ids(tmp_path / "bad.parquet")
    with pytest.raises(FileNotFoundError):
        read_paper_ids(tmp_path / "missing.parquet")


# ---- missing, zero and invalid weights are explicit errors (weighted mode only) ----------------------------
def test_invalid_weight_error_says_how_many_and_where():
    df = frame([("A", "B", 3), ("A", "C", 0), ("B", "C", -1), ("C", "A", float("nan"))])
    with pytest.raises(EdgeFormatError) as err:
        clean_edges(df, weight_col="num_contexts")
    message = str(err.value)
    assert "positive finite numbers" in message and "3 invalid value(s)" in message
    assert "first at row 1: 0" in message                                    # row index 1 holds the first bad weight (0)


@pytest.mark.parametrize("bad", [0, -2, None, float("nan"), "n/a", float("inf")])
def test_every_kind_of_bad_weight_is_rejected_in_weighted_mode(bad):
    df = frame([("A", "B", 1), ("A", "C", 1)])
    df["num_contexts"] = pd.Series([1, bad], dtype=object)
    with pytest.raises(EdgeFormatError, match="1 invalid value"):
        clean_edges(df, weight_col="num_contexts")


def test_unweighted_mode_never_looks_at_the_weight_column():
    df = frame([("A", "B", 1), ("A", "C", 1)])
    df["num_contexts"] = pd.Series([1, None], dtype=object)                  # unusable weight, but weights were not requested
    table = clean_edges(df)
    assert len(table.edges) == 2 and table.edges["weight"].tolist() == [1.0, 1.0] and not table.has_weights


def test_has_weights_flag_follows_the_weight_column_argument():
    assert not clean_edges(frame([("A", "B", 2)])).has_weights
    assert clean_edges(frame([("A", "B", 2)]), weight_col="num_contexts").has_weights


def test_a_null_weight_in_a_parquet_file_is_rejected_when_weights_are_requested(tmp_path):
    path = tmp_path / "e.parquet"
    pq.write_table(pa.table({"citing_paper_id": ["A", "A"], "cited_paper_id": ["B", "C"],
                             "num_contexts": pa.array([3, None], pa.int32())}), path)
    with pytest.raises(EdgeFormatError, match="1 invalid value"):
        read_edges(path, weight_col="num_contexts")
    assert len(read_edges(path).edges) == 2                                  # fine when weights are not used
