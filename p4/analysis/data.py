"""Read P1's tables and join them into one row per judged query.

P1 schemas used (citeds/schema.py):
    queries.parquet           query_id, split, context_id, sentence_id, citing_paper_id, cited_paper_id, text,
                              cited_citation_count
    papers.parquet            paper_id, title, abstract, year (nullable), categories, citation_count, num_contexts
    context_functions.parquet context_id, function (background | method | result), confidence
    index_contexts.parquet    one row per leakage-controlled citing sentence; cited_paper_id says which paper it expands

Two popularity fields, with different meanings (both come from P1):
    papers.citation_count           number of citing papers inside the corpus. PRIMARY measure in every P4 analysis.
                                    It is defined for every paper, including papers that are never a query target.
    queries.cited_citation_count    the same number, copied onto the query row when P1 sampled the query.
                                    SUPPLEMENTARY: kept as metadata and cross-checked against papers.citation_count.
Both INCLUDE the query's own citing paper (the query sentence is one of the citations), so a paper that is the target of
a query always has citation_count >= 1. Papers with citation_count 0 can never be evaluated as targets; they only
appear in rankings as distractors.

Missing data is never dropped silently: a query without metadata keeps its row, the field is NaN, and the counts of
missing values are reported in `dataset_stats`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_LABELS = ("real", "synthetic", "demo")
FUNCTIONS = ("background", "method", "result")          # citeds.schema.CITATION_FUNCTIONS


class AnalysisDataError(ValueError):
    """An input table is missing, unreadable, or lacks a required column."""


def check_data_label(label) -> str:
    if label not in DATA_LABELS:
        raise AnalysisDataError(f"data label must be one of {DATA_LABELS}, got {label!r}")
    return label


def _read_table(path, kind: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{kind} file not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            import pyarrow.parquet as pq
            try:
                names = pq.ParquetFile(path).schema_arrow.names
            except Exception as err:                      # pyarrow raises several types (ArrowInvalid, OSError, ...)
                raise AnalysisDataError(f"{path.name}: cannot read {kind} as a parquet file ({err})") from None
            missing = [c for c in required if c not in names]
            if missing:
                raise AnalysisDataError(f"{path.name} ({kind}) is missing column(s) {missing}; columns found: {names}")
            df = pd.read_parquet(path, columns=[c for c in (*required, *optional) if c in names])
        elif suffix in (".csv", ".tsv"):
            df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", encoding="utf-8-sig",
                             dtype={c: str for c in ("query_id", "paper_id", "cited_paper_id", "citing_paper_id",
                                                     "context_id")})
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise AnalysisDataError(f"{path.name} ({kind}) is missing column(s) {missing}; columns found: {list(df.columns)}")
        else:
            raise AnalysisDataError(f"unsupported {kind} file type {suffix!r}: use .parquet, .csv or .tsv ({path.name})")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as err:
        raise AnalysisDataError(f"{path.name}: cannot read {kind} ({err})") from None
    return df


def load_queries(path) -> pd.DataFrame:
    return _read_table(path, "queries", ("query_id", "cited_paper_id"),
                       ("split", "context_id", "citing_paper_id", "text", "cited_citation_count"))


def load_papers(path) -> pd.DataFrame:
    df = _read_table(path, "papers", ("paper_id",), ("title", "year", "citation_count"))
    if df["paper_id"].duplicated().any():
        raise AnalysisDataError(f"{Path(path).name}: duplicate paper_id values")
    return df


def load_functions(path) -> pd.DataFrame:
    df = _read_table(path, "context functions", ("context_id", "function"), ("confidence",))
    bad = sorted(set(df["function"].dropna()) - set(FUNCTIONS))
    if bad:
        raise AnalysisDataError(f"{Path(path).name}: unknown function label(s) {bad}; expected {list(FUNCTIONS)}")
    return df.drop_duplicates("context_id")


def load_index_context_counts(path) -> pd.Series:
    """Number of indexable (leakage-controlled) contexts per cited paper, from index_contexts.parquet."""
    df = _read_table(path, "index contexts", ("cited_paper_id",))
    return df.groupby("cited_paper_id").size().rename("n_index_contexts")


def build_query_table(qrels: dict[str, dict[str, int]], queries: pd.DataFrame | None, papers: pd.DataFrame | None,
                      functions: pd.DataFrame | None = None, index_counts: pd.Series | None = None) -> pd.DataFrame:
    """One row per judged query (the qrels define which), indexed by query_id.

    Columns: target_id (from the qrels: the ground truth), text, split, citing_paper_id, context_id, title (of the
    target), year, citation_count (papers.citation_count of the target), cited_citation_count (from queries),
    citing_year, citation_age (citing year - target year), function, function_confidence, n_index_contexts.
    Anything that cannot be looked up is NaN; `dataset_stats` counts these.
    """
    table = pd.DataFrame({"query_id": sorted(qrels)})
    table["target_id"] = [sorted(qrels[q])[0] for q in table["query_id"]]       # first relevant paper (one per query in P1)
    if queries is not None:
        q = queries.drop_duplicates("query_id").set_index("query_id")
        for col in ("text", "split", "citing_paper_id", "context_id", "cited_citation_count"):
            table[col] = table["query_id"].map(q[col]) if col in q.columns else np.nan
        table["queries_target_id"] = table["query_id"].map(q["cited_paper_id"])
    else:
        for col in ("text", "split", "citing_paper_id", "context_id", "cited_citation_count", "queries_target_id"):
            table[col] = np.nan
    if papers is not None:
        p = papers.set_index("paper_id")
        for src, dst in (("title", "title"), ("year", "year"), ("citation_count", "citation_count")):
            table[dst] = table["target_id"].map(p[src]) if src in p.columns else np.nan
        table["citing_year"] = table["citing_paper_id"].map(p["year"]) if "year" in p.columns else np.nan
        table["target_in_papers"] = table["target_id"].isin(p.index)
    else:
        table["target_in_papers"] = np.nan
        for col in ("title", "year", "citation_count", "citing_year"):
            table[col] = np.nan
    for col in ("year", "citation_count", "cited_citation_count", "citing_year"):
        table[col] = pd.to_numeric(table[col], errors="coerce").astype(float)
    table["citation_age"] = table["citing_year"] - table["year"]
    if functions is not None:
        f = functions.set_index("context_id")
        table["function"] = table["context_id"].map(f["function"])
        table["function_confidence"] = (pd.to_numeric(table["context_id"].map(f["confidence"]), errors="coerce")
                                        if "confidence" in f.columns else np.nan)
    else:
        table["function"], table["function_confidence"] = np.nan, np.nan
    table["n_index_contexts"] = (pd.to_numeric(table["target_id"].map(index_counts), errors="coerce").fillna(0)
                                 if index_counts is not None else np.nan)
    return table.set_index("query_id")


def dataset_stats(table: pd.DataFrame, papers: pd.DataFrame | None, *, have_queries: bool, have_functions: bool,
                  have_index_counts: bool) -> dict:
    """Counts that describe the data behind a report (shown in every report's metadata)."""
    n = len(table)
    stats = {
        "judged queries": n,
        "papers table rows": None if papers is None else len(papers),
        "papers with citation_count 0": None if papers is None or "citation_count" not in papers.columns
        else int((papers["citation_count"] == 0).sum()),
        "queries without a row in the queries file": int(table["text"].isna().sum()) if have_queries else "queries file not given",
        "targets not found in papers": int((~table["target_in_papers"].astype(bool)).sum()) if papers is not None else "papers file not given",
        "queries with unknown target year": int(table["year"].isna().sum()),
        "queries with unknown target citation_count": int(table["citation_count"].isna().sum()),
    }
    if have_queries:
        both = table[["queries_target_id"]].dropna()
        stats["queries whose target differs between qrels and queries file"] = int(
            (both["queries_target_id"] != table.loc[both.index, "target_id"]).sum())
        both = table[["citation_count", "cited_citation_count"]].dropna()
        stats["queries where cited_citation_count != papers.citation_count"] = int(
            (both["citation_count"] != both["cited_citation_count"]).sum())
    stats["queries without a citation-function label"] = (int(table["function"].isna().sum()) if have_functions
                                                         else "context_functions file not given")
    if have_index_counts:
        stats["queries whose target has 0 indexable contexts"] = int((table["n_index_contexts"] == 0).sum())
    return stats
