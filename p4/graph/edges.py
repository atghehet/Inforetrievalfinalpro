"""Read and clean a citation edge table.

P1's real schema (citeds/schema.py EDGES, docs/SCHEMA.md), one row per (citing paper, cited paper):

    citing_paper_id   string   source node: the paper that contains the citation
    cited_paper_id    string   target node: the paper being cited
    num_contexts      int32    number of citing sentences (optional edge weight)

so an edge points from the CITING paper to the CITED paper. P1 already removes self-citations and
duplicate pairs, but this module does not rely on that: it cleans any edge table the same way, and
reports what it removed, so results are the same whatever produced the table.

Cleaning rules (applied in this order, each counted in `EdgeStats`):
  1. rows with a missing (null / empty) endpoint are dropped
  2. self-citations (source == target) are dropped: a paper citing itself says nothing about its importance
  3. duplicate (source, target) pairs are merged into one edge; their weights are added up
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# P1's column names (the defaults everywhere in p4.graph)
SOURCE_COLUMN = "citing_paper_id"
TARGET_COLUMN = "cited_paper_id"
WEIGHT_COLUMN = "num_contexts"
PAPER_ID_COLUMN = "paper_id"          # papers.parquet


class EdgeFormatError(ValueError):
    """The edge file is missing, has the wrong columns, or has unusable values."""


@dataclass(frozen=True)
class EdgeStats:
    n_input: int            # rows read
    n_missing: int          # rows dropped for a null or empty endpoint
    n_self_loops: int       # self-citations dropped
    n_duplicates: int       # extra rows merged into an existing edge
    n_edges: int            # edges left


@dataclass(frozen=True)
class EdgeTable:
    """Cleaned edges. `edges` has the columns source, target, weight (sorted by source, target).

    `has_weights` is True only if the weights come from a real weight column (e.g. num_contexts). Otherwise every
    weight is a placeholder 1.0, and weighted PageRank refuses to use the table instead of silently ignoring weights.
    """
    edges: pd.DataFrame
    stats: EdgeStats
    has_weights: bool = False


def clean_edges(df: pd.DataFrame, *, source_col: str = SOURCE_COLUMN, target_col: str = TARGET_COLUMN,
                weight_col: str | None = None) -> EdgeTable:
    """Clean an edge DataFrame (see the module docstring). `weight_col=None` gives every edge weight 1."""
    for name, col in (("source", source_col), ("target", target_col)):
        if col not in df.columns:
            raise EdgeFormatError(f"edge table has no {name} column {col!r}; columns found: {list(df.columns)}")
    if weight_col is not None and weight_col not in df.columns:
        raise EdgeFormatError(f"edge table has no weight column {weight_col!r}; columns found: {list(df.columns)}")
    work = pd.DataFrame({"source": df[source_col], "target": df[target_col]})
    if weight_col is None:
        work["weight"] = 1.0
    else:
        values = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        bad = ~np.isfinite(values) | (values <= 0)               # NaN, missing, text, infinite, zero and negative
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            example = df[weight_col].iloc[first]
            example = example.item() if hasattr(example, "item") else example      # plain Python value, not np.int64(0)
            raise EdgeFormatError(f"weight column {weight_col!r} must contain positive finite numbers; "
                                  f"{int(bad.sum())} invalid value(s), first at row {df.index[first]!r}: {example!r}")
        work["weight"] = values
    n_input = len(work)

    missing = work["source"].isna() | work["target"].isna()
    work = work[~missing].astype({"source": str, "target": str})
    blank = (work["source"].str.strip() == "") | (work["target"].str.strip() == "")
    work = work[~blank]
    n_missing = n_input - len(work)

    loops = work["source"] == work["target"]
    n_self = int(loops.sum())
    work = work[~loops]

    n_before = len(work)
    work = work.groupby(["source", "target"], as_index=False, sort=True)["weight"].sum()
    n_dup = n_before - len(work)
    return EdgeTable(work.reset_index(drop=True), EdgeStats(n_input, n_missing, n_self, n_dup, len(work)),
                     has_weights=weight_col is not None)


def read_edges(path, *, source_col: str = SOURCE_COLUMN, target_col: str = TARGET_COLUMN,
               weight_col: str | None = None) -> EdgeTable:
    """Read P1's edges.parquet (or a .csv / .tsv with the same columns) and clean it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"edge file not found: {path}")
    wanted = [c for c in (source_col, target_col, weight_col) if c]
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            df = pd.read_parquet(path, columns=[c for c in wanted if c in _parquet_columns(path)])
        elif suffix in (".csv", ".tsv"):
            df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype={source_col: str, target_col: str},
                             encoding="utf-8-sig")
        else:
            raise EdgeFormatError(f"unsupported edge file type {suffix!r}: use .parquet, .csv or .tsv ({path.name})")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as err:
        if isinstance(err, FileNotFoundError):
            raise
        raise EdgeFormatError(f"{path.name}: cannot read edges ({err})") from None
    return clean_edges(df, source_col=source_col, target_col=target_col, weight_col=weight_col)


def _parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq
    return list(pq.ParquetFile(path).schema_arrow.names)


def read_paper_ids(path, *, id_col: str = PAPER_ID_COLUMN) -> pd.DataFrame:
    """Read papers.parquet / .csv and return its `paper_id` column (plus `title` if present).

    Used to tell the algorithms about papers that have no edges at all, which would otherwise be invisible.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"papers file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        names = _parquet_columns(path)
        if id_col not in names:
            raise EdgeFormatError(f"{path.name} has no column {id_col!r}; columns found: {names}")
        df = pd.read_parquet(path, columns=[c for c in (id_col, "title") if c in names])
    elif suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=str, encoding="utf-8-sig")
        if id_col not in df.columns:
            raise EdgeFormatError(f"{path.name} has no column {id_col!r}; columns found: {list(df.columns)}")
        df = df[[c for c in (id_col, "title") if c in df.columns]]
    else:
        raise EdgeFormatError(f"unsupported papers file type {suffix!r}: use .parquet, .csv or .tsv")
    df = df.dropna(subset=[id_col]).astype({id_col: str}).drop_duplicates(id_col)
    return df.rename(columns={id_col: "paper_id"}).reset_index(drop=True)


def as_edge_table(edges, **column_options) -> EdgeTable:
    """Accept an EdgeTable, a DataFrame (cleaned with `column_options`), or a path to an edge file."""
    if isinstance(edges, EdgeTable):
        return edges
    if isinstance(edges, pd.DataFrame):
        return clean_edges(edges, **column_options)
    return read_edges(edges, **column_options)
