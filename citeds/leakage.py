"""Leakage control: keep query sentences out of the citation-context index.

A query is a citing sentence and its relevant document is the paper it cites.
If that same sentence (or a copy of it) is also in the cited paper's expanded
representation, retrieval becomes trivial string matching and scores are
inflated. `build_index_contexts` removes, for EVERY query in EVERY split:

  1. the query's own sentence (all rows sharing its sentence_id),
  2. exact copies: any context whose normalised text equals a query text
     (e.g. the same sentence re-used in another version or by co-authors),
  3. near duplicates: contexts citing the SAME paper whose word 3-gram
     Jaccard similarity with the query is >= config.NEAR_DUP_JACCARD,
  4. query text appearing as a neighbour: prev_text/next_text equal to a
     query sentence are blanked, so "surrounding sentence" features can't leak.

Use `assert_no_leakage` (Member 4) before reporting any results.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from . import config
from .clean import normalise_for_match


def shingles(text: str, k: int = config.SHINGLE_SIZE) -> frozenset:
    toks = normalise_for_match(text).split()
    if len(toks) < k:
        return frozenset([" ".join(toks)]) if toks else frozenset()
    return frozenset(" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_leaking_rows(contexts: pd.DataFrame, queries: pd.DataFrame,
                      threshold: float = config.NEAR_DUP_JACCARD) -> pd.Series:
    """Boolean mask over `contexts`: True where the row leaks a query."""
    q_sentences = set(queries["sentence_id"])
    q_norm = set(queries["text"].map(normalise_for_match))

    leak = contexts["sentence_id"].isin(q_sentences)
    leak |= contexts["text"].map(normalise_for_match).isin(q_norm)

    # Near-duplicates: only compare against contexts of the same cited paper.
    q_by_cited: dict[str, list[frozenset]] = defaultdict(list)
    for cited, text in zip(queries["cited_paper_id"], queries["text"]):
        q_by_cited[cited].append(shingles(text))
    candidates = contexts.index[~leak & contexts["cited_paper_id"].isin(q_by_cited.keys())]
    near = []
    for idx in candidates:
        row_sh = shingles(contexts.at[idx, "text"])
        if any(jaccard(row_sh, qs) >= threshold for qs in q_by_cited[contexts.at[idx, "cited_paper_id"]]):
            near.append(idx)
    leak.loc[near] = True
    return leak


def build_index_contexts(contexts: pd.DataFrame, queries: pd.DataFrame,
                         threshold: float = config.NEAR_DUP_JACCARD) -> pd.DataFrame:
    """Contexts that are safe to put in the expanded index."""
    leak = find_leaking_rows(contexts, queries, threshold)
    out = contexts.loc[~leak].copy()
    q_norm = set(queries["text"].map(normalise_for_match))
    for col in ("prev_text", "next_text"):
        hit = out[col].map(normalise_for_match).isin(q_norm) & (out[col] != "")
        out.loc[hit, col] = ""
    return out.reset_index(drop=True)


def assert_no_leakage(index_contexts: pd.DataFrame, queries: pd.DataFrame,
                      threshold: float = config.NEAR_DUP_JACCARD) -> None:
    """Raise AssertionError if any query sentence or copy is still in the index."""
    leak = find_leaking_rows(index_contexts, queries, threshold)
    assert not leak.any(), f"{int(leak.sum())} index contexts leak query text, e.g. {index_contexts.loc[leak, 'context_id'].head(3).tolist()}"
    q_norm = set(queries["text"].map(normalise_for_match))
    for col in ("prev_text", "next_text"):
        bad = (index_contexts[col] != "") & index_contexts[col].map(normalise_for_match).isin(q_norm)
        assert not bad.any(), f"{int(bad.sum())} index contexts leak query text via {col}"
