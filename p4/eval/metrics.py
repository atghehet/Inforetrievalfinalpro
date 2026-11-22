"""Retrieval metrics for ONE query.

Every function takes
    ranked   : list of document IDs, best first (e.g. [doc_id for doc_id, _ in run[qid]])
    relevant : the relevant document IDs (a set, or the keys of a qrels dict)
and returns a number between 0 and 1.

Relevance is BINARY here (a document is relevant or not). Our queries have ONE
relevant paper (the paper the citing sentence cites), but the code also works
when there are several, so it is correct for any qrels file.

A query where nothing relevant is retrieved scores 0 on every metric.

ASSUMPTION: `ranked` contains no document ID twice. A ranking lists each
document once. `trec_io.read_run` already rejects duplicates, and every metric
here also checks the part of `ranked` it looks at and raises ValueError, because
a repeated relevant document would be counted twice (AP and nDCG could exceed 1).
"""
from __future__ import annotations

import math
from typing import Collection, Sequence


def _check_no_duplicates(ranked: Sequence[str]) -> None:
    """Raise ValueError if a document ID appears more than once in `ranked`."""
    if len(set(ranked)) != len(ranked):
        seen, dup = set(), None
        for doc in ranked:
            if doc in seen:
                dup = doc
                break
            seen.add(doc)
        raise ValueError(f"ranking lists document {dup!r} more than once; a ranking must not contain duplicates")


def _check_k(k: int) -> None:
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"cutoff k must be a whole number >= 1, got {k!r}")


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of the relevant documents found in the top k.

    With one relevant document this is 1 if it is in the top k, else 0
    (also called hit rate / success@k). Assumes no duplicate IDs in the top k.
    """
    _check_k(k)
    if not relevant:
        raise ValueError("a query needs at least one relevant document")
    _check_no_duplicates(ranked[:k])
    found = len(set(ranked[:k]) & set(relevant))
    return found / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str], k: int | None = None) -> float:
    """1 / (rank of the first relevant document); 0 if none within the top k.

    k=None looks at the whole ranking. The mean over queries is MRR (MRR@10
    when k=10, which is what P2 and P3 report). Assumes no duplicate IDs in the
    part of `ranked` that is examined.
    """
    if k is not None:
        _check_k(k)
    if not relevant:
        raise ValueError("a query needs at least one relevant document")
    _check_no_duplicates(ranked[:k])
    for rank, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary gains.

    DCG  = sum over relevant documents in the top k of 1 / log2(rank + 1)
    IDCG = DCG of the perfect ranking (all relevant documents first)
    nDCG = DCG / IDCG

    With one relevant document at rank r: nDCG = 1 / log2(r + 1), the same
    formula P3 uses.

    Assumes no duplicate IDs in the top k: a repeated relevant document would be
    counted twice and could push nDCG above 1, so it raises ValueError instead.
    """
    _check_k(k)
    if not relevant:
        raise ValueError("a query needs at least one relevant document")
    _check_no_duplicates(ranked[:k])
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, doc in enumerate(ranked[:k], start=1) if doc in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def average_precision(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """Average precision over the whole ranking you give it.

    AP = (sum of precision@r at every rank r holding a relevant document)
         / (number of relevant documents)

    Relevant documents that were never retrieved add nothing to the sum but
    still count in the denominator. The mean over queries is MAP. With one
    relevant document at rank r, AP = 1 / r, the same as reciprocal rank
    without a cutoff.

    Assumes `ranked` has no duplicate IDs: a repeated relevant document would be
    counted as an extra hit and AP could exceed 1, so it raises ValueError instead.
    """
    if not relevant:
        raise ValueError("a query needs at least one relevant document")
    _check_no_duplicates(ranked)
    hits = 0
    total = 0.0
    for rank, doc in enumerate(ranked, start=1):
        if doc in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)
