"""Data loading module for all team members.

    from citeds import io
    papers   = io.load_papers()                  # documents (abstract-only index)
    ctx      = io.load_index_contexts()          # leakage-controlled contexts (expansion)
    queries  = io.load_queries("val")            # query_id, text, cited_paper_id, ...
    edges    = io.load_edges()                   # citation graph for PageRank
    funcs    = io.load_context_functions()       # citation-function labels (Member 2)
    docs     = io.expanded_documents()           # paper_id -> abstract + contexts text

NEVER build an index from `load_all_contexts()`: it still contains the query
sentences. It exists only for statistics and the citation-function classifier.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def _read(path: Path, columns=None) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - run `python -m citeds.build` first")
    return pd.read_parquet(path, columns=columns)


def load_papers(columns=None) -> pd.DataFrame:
    return _read(config.PAPERS_PATH, columns)


def load_index_contexts(columns=None) -> pd.DataFrame:
    return _read(config.INDEX_CONTEXTS_PATH, columns)


def load_all_contexts(columns=None) -> pd.DataFrame:
    """UNSAFE for indexing (contains query sentences)."""
    return _read(config.CONTEXTS_PATH, columns)


def load_edges() -> pd.DataFrame:
    return _read(config.EDGES_PATH)


def load_queries(split: str | None = None) -> pd.DataFrame:
    q = _read(config.QUERIES_PATH)
    if split is not None:
        if split not in config.SPLIT_SIZES:
            raise ValueError(f"split must be one of {list(config.SPLIT_SIZES)}")
        q = q[q["split"] == split].reset_index(drop=True)
    return q


def load_context_functions() -> pd.DataFrame:
    return _read(config.CONTEXT_FUNCTIONS_PATH)


def qrels_path(split: str) -> Path:
    return config.QRELS_DIR / f"qrels.{split}.txt"


def topics_path(split: str) -> Path:
    return config.TOPICS_DIR / f"topics.{split}.tsv"


def expanded_documents(include_abstract: bool = True, include_contexts: bool = True,
                       sep: str = " ") -> pd.DataFrame:
    """One row per paper: paper_id, title, abstract, contexts (joined), text.

    `text` is title + abstract + all leakage-controlled citing sentences, the
    simplest "document expansion" variant. Members 2/3 can build fancier
    field-weighted variants from load_index_contexts() directly.
    """
    papers = load_papers(["paper_id", "title", "abstract"])
    ctx = load_index_contexts(["cited_paper_id", "text"])
    joined = ctx.groupby("cited_paper_id")["text"].agg(sep.join)
    papers["contexts"] = papers["paper_id"].map(joined).fillna("")
    parts = [papers["title"]]
    if include_abstract:
        parts.append(papers["abstract"])
    if include_contexts:
        parts.append(papers["contexts"])
    papers["text"] = parts[0].str.cat(parts[1:], sep=sep).str.strip()
    return papers
