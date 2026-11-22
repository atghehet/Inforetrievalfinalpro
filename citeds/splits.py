"""Train/val/test query sets with a fixed seed, plus TREC qrels and topics files.

Query = a citing sentence; relevant document = the paper it cites.

Rules
-----
* Only sentences that cite exactly ONE corpus paper become queries, so every
  query has exactly one relevant document (clean ground truth).
* The split is assigned per CITING paper (hash of its ID + seed), so sentences
  from one citing paper never appear in two splits.
* A cited paper is only used as a query target while it still keeps
  >= MIN_REMAINING_CONTEXTS other citing sentences for the index.
* At most MAX_QUERIES_PER_CITED_PAPER queries per cited paper, so very popular
  papers don't dominate the evaluation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _split_of(citing_id: str, seed: int, sizes: dict[str, float]) -> str:
    h = int(hashlib.sha1(f"{seed}:{citing_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in sizes.items():
        acc += frac
        if h <= acc:
            return name
    return list(sizes)[-1]


def make_queries(contexts: pd.DataFrame, papers: pd.DataFrame,
                 seed: int = config.SEED,
                 max_queries: int = config.MAX_QUERIES,
                 per_cited_cap: int = config.MAX_QUERIES_PER_CITED_PAPER,
                 min_remaining: int = config.MIN_REMAINING_CONTEXTS,
                 sizes: dict[str, float] = config.SPLIT_SIZES) -> pd.DataFrame:
    # Number of distinct citing sentences available per cited paper.
    sent_per_cited = contexts.groupby("cited_paper_id")["sentence_id"].nunique().to_dict()
    cand = contexts[contexts["num_cited_in_sentence"] == 1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cand))
    cand = cand.iloc[order]

    used: dict[str, int] = {}
    rows = []
    for r in cand.itertuples(index=False):
        k = used.get(r.cited_paper_id, 0)
        if k >= per_cited_cap:
            continue
        if sent_per_cited[r.cited_paper_id] - (k + 1) < min_remaining:
            continue
        used[r.cited_paper_id] = k + 1
        rows.append(r)
        if len(rows) >= max_queries:
            break
    q = pd.DataFrame(rows, columns=cand.columns)
    q["split"] = q["citing_paper_id"].map(lambda c: _split_of(c, seed, sizes))
    cc = papers.set_index("paper_id")["citation_count"]
    q["cited_citation_count"] = q["cited_paper_id"].map(cc).astype("int32")
    return assign_query_ids(q, sizes)


def assign_query_ids(q: pd.DataFrame, sizes: dict[str, float] = config.SPLIT_SIZES) -> pd.DataFrame:
    """Sort by split then context_id and number queries q000000, q000001, ..."""
    rank = {s: i for i, s in enumerate(sizes)}
    q = (q.assign(_r=q["split"].map(rank)).sort_values(["_r", "context_id"])
         .drop(columns="_r").reset_index(drop=True))
    q["query_id"] = [f"q{i:06d}" for i in range(len(q))]
    return q


def write_trec_files(queries: pd.DataFrame, qrels_dir: Path | None = None,
                     topics_dir: Path | None = None) -> None:
    """qrels.<split>.txt: 'qid 0 docid 1' ; topics.<split>.tsv: 'qid<TAB>text'."""
    qrels_dir = qrels_dir or config.QRELS_DIR
    topics_dir = topics_dir or config.TOPICS_DIR
    qrels_dir.mkdir(parents=True, exist_ok=True)
    topics_dir.mkdir(parents=True, exist_ok=True)
    for split, g in queries.groupby("split"):
        with open(qrels_dir / f"qrels.{split}.txt", "w") as fh:
            for qid, did in zip(g["query_id"], g["cited_paper_id"]):
                fh.write(f"{qid} 0 {did} 1\n")
        with open(topics_dir / f"topics.{split}.tsv", "w") as fh:
            for qid, text in zip(g["query_id"], g["text"]):
                fh.write(f"{qid}\t{text.replace(chr(9), ' ')}\n")
