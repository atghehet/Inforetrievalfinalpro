"""Read and validate TREC run files and qrels files.

Run file, one line per retrieved document, six whitespace-separated fields:

    qid Q0 docid rank score run_name        e.g.  q000123 Q0 1902.00751 1 17.43 bm25

Qrels file, one line per judgment, four fields:

    qid 0 docid relevance                   e.g.  q000123 0 1902.00751 1

Design decisions (explain these in the viva):
  * The ORDER of a ranking is given by the `rank` column (as in P2/P3 code),
    not by sorting scores. We sort by rank, so line order in the file is irrelevant.
  * Bad input is an error, never silently skipped. A silently skipped line
    could change a result without anyone noticing.
  * Text encoding: UTF-8 (BOM optional) and UTF-16 with a BOM are read; other
    encodings are rejected with a clear message (see `_detect_encoding`).
  * A query that has judgments but NO lines in the run is NOT an error: it is
    a system that retrieved nothing, and it later scores 0. (This matches P3.)
    `check_run_queries` reports which queries are missing or extra.
"""
from __future__ import annotations

import codecs
import math
from pathlib import Path

# A run is  {query_id: [(doc_id, score), ...]}  with the list in rank order.
Run = dict[str, list[tuple[str, float]]]
# Qrels are {query_id: {doc_id: relevance_grade}}  holding only grades > 0.
Qrels = dict[str, dict[str, int]]


class TrecFormatError(ValueError):
    """A run or qrels file is malformed. The message names the file and line."""


def _detect_encoding(path: Path) -> str:
    """Choose how to decode the file by looking at its first bytes.

    UTF-8 (with or without a BOM) and UTF-16 with a BOM are accepted. Windows
    PowerShell's `>` writes UTF-16 with a BOM, and Notepad can add a UTF-8 BOM.
    A BOM left in the text would silently turn the first query ID "q1" into
    "\\ufeffq1", so BOMs are always consumed here. Anything else (UTF-32,
    UTF-16 without a BOM) is refused with a clear message, never guessed.
    """
    with path.open("rb") as fh:
        head = fh.read(4)
    if head.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):   # check before UTF-16: FF FE 00 00
        raise TrecFormatError(f"{path.name}: UTF-32 is not supported; re-save the file as UTF-8")
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"                       # strips the BOM
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"                          # reads the BOM, picks the byte order, strips it
    if b"\x00" in head:                          # plain text never contains NUL bytes
        raise TrecFormatError(f"{path.name}: looks like UTF-16 without a byte-order mark "
                              "(contains NUL bytes); re-save the file as UTF-8")
    return "utf-8"


def _lines(path):
    """Yield (line_number, fields) for every non-blank line.

    The file is streamed line by line (run files can be large). Line endings
    \\n, \\r\\n and \\r are all accepted. Undecodable bytes raise TrecFormatError.
    """
    path = Path(path)
    encoding = _detect_encoding(path)
    try:
        with path.open(encoding=encoding) as fh:
            for number, line in enumerate(fh, start=1):
                if line.strip():
                    yield number, line.split()
    except UnicodeDecodeError as err:
        raise TrecFormatError(f"{path.name}: not valid {encoding} text ({err.reason} at byte {err.start}); "
                              "re-save the file as UTF-8") from None


def read_run(path) -> Run:
    """Read a TREC run file into {query_id: [(doc_id, score), ...]} in rank order.

    Raises TrecFormatError for: a line without exactly 6 fields, a rank that is
    not a whole number >= 1, a score that is not a finite number, the same
    document twice in one query, or the same rank twice in one query.
    """
    path = Path(path)
    by_query: dict[str, dict[int, tuple[str, float]]] = {}
    seen_docs: dict[str, set[str]] = {}
    for number, fields in _lines(path):
        where = f"{path.name}, line {number}"
        if len(fields) != 6:
            raise TrecFormatError(f"{where}: expected 6 fields, found {len(fields)}: {' '.join(fields)!r}")
        qid, _q0, doc_id, rank_text, score_text, _tag = fields
        try:
            rank = int(rank_text)
        except ValueError:
            raise TrecFormatError(f"{where}: rank {rank_text!r} is not an integer") from None
        if rank < 1:
            raise TrecFormatError(f"{where}: rank must be >= 1, found {rank}")
        try:
            score = float(score_text)
        except ValueError:
            raise TrecFormatError(f"{where}: score {score_text!r} is not a number") from None
        if not math.isfinite(score):
            raise TrecFormatError(f"{where}: score must be finite, found {score_text!r}")
        if doc_id in seen_docs.setdefault(qid, set()):
            raise TrecFormatError(f"{where}: document {doc_id!r} appears twice for query {qid!r}")
        seen_docs[qid].add(doc_id)
        ranks = by_query.setdefault(qid, {})
        if rank in ranks:
            raise TrecFormatError(f"{where}: rank {rank} used twice for query {qid!r}")
        ranks[rank] = (doc_id, score)
    return {qid: [ranks[r] for r in sorted(ranks)] for qid, ranks in by_query.items()}


def read_qrels(path) -> Qrels:
    """Read a qrels file into {query_id: {doc_id: grade}}, keeping grades > 0 only.

    Raises TrecFormatError for: a line without exactly 4 fields, a grade that is
    not an integer, the same (query, document) judged twice, or a query that
    has no relevant document at all (it could never score above 0).
    """
    path = Path(path)
    judged: dict[str, dict[str, int]] = {}
    for number, fields in _lines(path):
        where = f"{path.name}, line {number}"
        if len(fields) != 4:
            raise TrecFormatError(f"{where}: expected 4 fields, found {len(fields)}: {' '.join(fields)!r}")
        qid, _zero, doc_id, grade_text = fields
        try:
            grade = int(grade_text)
        except ValueError:
            raise TrecFormatError(f"{where}: relevance {grade_text!r} is not an integer") from None
        docs = judged.setdefault(qid, {})
        if doc_id in docs:
            raise TrecFormatError(f"{where}: document {doc_id!r} judged twice for query {qid!r}")
        docs[doc_id] = grade
    qrels: Qrels = {}
    for qid, docs in judged.items():
        positive = {d: g for d, g in docs.items() if g > 0}
        if not positive:
            raise TrecFormatError(f"{path.name}: query {qid!r} has no relevant document")
        qrels[qid] = positive
    return qrels


def run_tag(path) -> str:
    """The run name (sixth column) of the first row of a run file. P2 and P3 write the winning configuration there."""
    for _number, fields in _lines(path):
        if len(fields) != 6:
            raise TrecFormatError(f"{Path(path).name}: cannot read the run tag, the first row does not have 6 fields")
        return fields[5]
    raise TrecFormatError(f"{Path(path).name}: the run file is empty, so it has no run tag")


def check_run_queries(run: Run, qrels: Qrels) -> dict[str, list[str]]:
    """Compare the queries in a run with the queries in the qrels.

    Returns {"missing": [...], "extra": [...]}:
      missing = judged queries the run returned nothing for (they score 0)
      extra   = queries in the run with no judgments (they are ignored)
    """
    return {"missing": sorted(set(qrels) - set(run)),
            "extra": sorted(set(run) - set(qrels))}
