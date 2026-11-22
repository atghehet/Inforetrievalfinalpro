"""Tests for p4.eval.trec_io: reading and validating run and qrels files."""
import pytest

from p4.eval.trec_io import TrecFormatError, check_run_queries, read_qrels, read_run


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_read_run_orders_by_rank_not_by_line_order(tmp_path):
    p = write(tmp_path, "a.run", "q1 Q0 B 2 1.5 sys\nq1 Q0 A 1 9.0 sys\n\nq2 Q0 C 1 3.0 sys\n")
    assert read_run(p) == {"q1": [("A", 9.0), ("B", 1.5)], "q2": [("C", 3.0)]}


def test_read_empty_run_gives_no_queries(tmp_path):
    assert read_run(write(tmp_path, "empty.run", "")) == {}


@pytest.mark.parametrize("line, message", [
    ("q1 Q0 A 1 2.0", "expected 6 fields"),            # too few fields
    ("q1 Q0 A 1 2.0 sys extra", "expected 6 fields"),  # too many fields
    ("q1 Q0 A one 2.0 sys", "not an integer"),         # rank is text
    ("q1 Q0 A 1.5 2.0 sys", "not an integer"),         # rank is a decimal
    ("q1 Q0 A 0 2.0 sys", "rank must be >= 1"),        # rank zero
    ("q1 Q0 A -3 2.0 sys", "rank must be >= 1"),       # negative rank
    ("q1 Q0 A 1 high sys", "not a number"),            # score is text
    ("q1 Q0 A 1 nan sys", "must be finite"),           # score is NaN
])
def test_read_run_rejects_malformed_rows(tmp_path, line, message):
    with pytest.raises(TrecFormatError, match=message):
        read_run(write(tmp_path, "bad.run", line + "\n"))


def test_error_message_names_file_and_line(tmp_path):
    p = write(tmp_path, "bad.run", "q1 Q0 A 1 2.0 sys\nq1 Q0 B oops 1.0 sys\n")
    with pytest.raises(TrecFormatError, match=r"bad\.run, line 2"):
        read_run(p)


def test_read_run_rejects_duplicate_document_in_one_query(tmp_path):
    p = write(tmp_path, "dup.run", "q1 Q0 A 1 2.0 sys\nq1 Q0 A 2 1.0 sys\n")
    with pytest.raises(TrecFormatError, match="appears twice"):
        read_run(p)


def test_same_document_in_different_queries_is_fine(tmp_path):
    p = write(tmp_path, "ok.run", "q1 Q0 A 1 2.0 sys\nq2 Q0 A 1 2.0 sys\n")
    assert set(read_run(p)) == {"q1", "q2"}


def test_read_run_rejects_duplicate_rank_in_one_query(tmp_path):
    p = write(tmp_path, "dup_rank.run", "q1 Q0 A 1 2.0 sys\nq1 Q0 B 1 1.0 sys\n")
    with pytest.raises(TrecFormatError, match="rank 1 used twice"):
        read_run(p)


def test_read_qrels_keeps_only_positive_grades(tmp_path):
    p = write(tmp_path, "q.txt", "q1 0 A 1\nq1 0 B 0\nq2 0 C 2\n")
    assert read_qrels(p) == {"q1": {"A": 1}, "q2": {"C": 2}}


@pytest.mark.parametrize("text, message", [
    ("q1 0 A\n", "expected 4 fields"),
    ("q1 0 A yes\n", "not an integer"),
    ("q1 0 A 1\nq1 0 A 1\n", "judged twice"),
    ("q1 0 A 0\n", "no relevant document"),
])
def test_read_qrels_rejects_bad_input(tmp_path, text, message):
    with pytest.raises(TrecFormatError, match=message):
        read_qrels(write(tmp_path, "q.txt", text))


def test_check_run_queries_reports_missing_and_extra():
    qrels = {"q1": {"A": 1}, "q2": {"B": 1}}
    run = {"q2": [("B", 1.0)], "q9": [("Z", 1.0)]}
    assert check_run_queries(run, qrels) == {"missing": ["q1"], "extra": ["q9"]}


# ---- Batch 1: text encodings --------------------------------------------------
import codecs   # noqa: E402

RUN_TEXT = "q1 Q0 A 1 2.5 sys\nq1 Q0 B 2 1.5 sys\nq2 Q0 C 1 3.0 sys\n"
QRELS_TEXT = "q1 0 A 1\nq2 0 C 1\n"
EXPECTED_RUN = {"q1": [("A", 2.5), ("B", 1.5)], "q2": [("C", 3.0)]}
EXPECTED_QRELS = {"q1": {"A": 1}, "q2": {"C": 1}}


def write_bytes(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("label, encode", [
    ("plain utf-8", lambda s: s.encode("utf-8")),
    ("utf-8 with BOM", lambda s: codecs.BOM_UTF8 + s.encode("utf-8")),
    ("utf-16 LE with BOM (PowerShell >)", lambda s: s.encode("utf-16")),          # Python adds an LE BOM
    ("utf-16 BE with BOM", lambda s: codecs.BOM_UTF16_BE + s.encode("utf-16-be")),
    ("Windows CRLF line endings", lambda s: s.replace("\n", "\r\n").encode("utf-8")),
    ("BOM + CRLF", lambda s: codecs.BOM_UTF8 + s.replace("\n", "\r\n").encode("utf-8")),
])
def test_supported_encodings_give_identical_results(tmp_path, label, encode):
    assert read_run(write_bytes(tmp_path, "a.run", encode(RUN_TEXT))) == EXPECTED_RUN, label
    assert read_qrels(write_bytes(tmp_path, "a.qrels", encode(QRELS_TEXT))) == EXPECTED_QRELS, label


def test_utf8_bom_does_not_corrupt_the_first_query_id(tmp_path):
    # Before the fix the first key was "﻿q1", silently splitting query q1 in two.
    run = read_run(write_bytes(tmp_path, "bom.run", codecs.BOM_UTF8 + RUN_TEXT.encode()))
    assert sorted(run) == ["q1", "q2"]
    assert all("﻿" not in q for q in run)


@pytest.mark.parametrize("label, encode, message", [
    ("utf-16 without BOM", lambda s: s.encode("utf-16-le"), "UTF-16 without a byte-order mark"),
    ("utf-32 with BOM", lambda s: s.encode("utf-32"), "UTF-32 is not supported"),
    ("latin-1 accent", lambda s: s.replace("A", "\xe9").encode("latin-1"), "not valid utf-8"),
    ("broken utf-8 after a BOM", lambda s: codecs.BOM_UTF8 + s.replace("A", "\xe9").encode("latin-1"),
     "not valid utf-8-sig"),
    ("truncated utf-16", lambda s: s.encode("utf-16")[:-1], "not valid utf-16"),
])
def test_unsupported_or_broken_encodings_give_a_clear_error(tmp_path, label, encode, message):
    # Each reader gets a file in ITS OWN format (the qrels reader streams line by line, so a
    # run-shaped file would fail on field count before reaching the broken bytes).
    for reader, text, name in ((read_run, RUN_TEXT, "bad.run"), (read_qrels, QRELS_TEXT, "bad.qrels")):
        path = write_bytes(tmp_path, name, encode(text))
        with pytest.raises(TrecFormatError, match=message) as err:
            reader(path)
        assert name in str(err.value) and "UTF-8" in str(err.value)        # names the file, tells the fix


def test_bom_only_or_empty_file_reads_as_no_rows(tmp_path):
    assert read_run(write_bytes(tmp_path, "bom_only.run", codecs.BOM_UTF8)) == {}
    assert read_run(write_bytes(tmp_path, "empty.run", b"")) == {}
