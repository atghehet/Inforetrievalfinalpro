"""Retrieval-side popularity bias: how popular are the papers a system returns?

Query-side effects (does a system score better when the RELEVANT paper is popular?) are handled by the group and
trend tools in groups.py. This module looks at the OTHER side: the papers the system puts at the top, whatever the query.

Popularity = papers.citation_count (citing papers inside the corpus), known for every paper in papers.parquet.
Measures, per system, over the top-k results of every judged query (documents missing from papers.parquet are counted
and left out):

  * mean_log10_citations_topk   mean of log10(1 + citation_count) over the retrieved documents, then over queries
  * median_citations_topk       median citation_count over all retrieved documents
  * top_decile_share_topk       share of retrieved documents that belong to the most-cited 10 % of the corpus, and
    corpus_top_decile_share     the share of the corpus that group occupies (about 0.10; ties at the cutoff can make it
                                larger). A system with no popularity preference would show a share close to the corpus
                                share; top_decile_lift = share / corpus share.
  * zero_citation_share_topk    share of retrieved documents with citation_count 0, against the corpus share of such
                                papers. These papers are never query targets, so they can be analysed only as distractors.
  * wrong top-1 popularity      among queries whose top-1 result is wrong: how often that wrong paper is strictly more /
                                strictly less cited than the relevant paper (ties are in neither).

The relevant papers are themselves not a random sample of the corpus (every one has at least one citation), so the corpus
share is a reference point, not a neutral expectation. Differences between systems are the more informative comparison.
These are descriptions of what systems return, not explanations of why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from p4.eval.significance import DEFAULT_CONFIDENCE, DEFAULT_N_BOOT, DEFAULT_SEED, bootstrap_ci

TOP_FRACTION = 0.10


def corpus_reference(papers: pd.DataFrame) -> dict:
    """Corpus-level reference numbers from papers.parquet (needs a citation_count column)."""
    counts = pd.to_numeric(papers["citation_count"], errors="coerce").dropna()
    threshold = float(np.quantile(counts, 1 - TOP_FRACTION)) if len(counts) else float("nan")
    in_top = (counts >= threshold) & (counts > 0)
    return {"n_papers": int(len(counts)), "threshold": threshold,
            "top_decile_share": float(in_top.mean()) if len(counts) else float("nan"),
            "zero_citation_share": float((counts == 0).mean()) if len(counts) else float("nan"),
            "n_zero_citation": int((counts == 0).sum()),
            "median_citations": float(counts.median()) if len(counts) else float("nan")}


def retrieval_popularity(runs: dict, qrels: dict, papers: pd.DataFrame, *, k: int = 10,
                         n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                         confidence: float = DEFAULT_CONFIDENCE) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (system summary, per-query table, corpus reference)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if "citation_count" not in papers.columns:
        raise ValueError("papers table has no citation_count column; popularity analysis needs it")
    reference = corpus_reference(papers)
    citations = pd.to_numeric(papers.set_index("paper_id")["citation_count"], errors="coerce").to_dict()
    threshold = reference["threshold"]
    rows = []
    for system, run in runs.items():
        for qid in sorted(qrels):
            ranked = [doc for doc, _ in run.get(qid, [])]
            top = ranked[:k]
            known = [citations[d] for d in top if d in citations and citations[d] == citations[d]]
            target = sorted(qrels[qid])[0]
            top1 = ranked[0] if ranked else None
            rows.append({
                "system": system, "query_id": qid, "target_id": target, "target_citation_count": citations.get(target, np.nan),
                "n_topk": len(top), "n_topk_known": len(known),
                "topk_mean_log10_citations": float(np.mean(np.log10(1 + np.array(known)))) if known else np.nan,
                "topk_top_decile_hits": int(sum(1 for c in known if c >= threshold and c > 0)),
                "topk_zero_citation_hits": int(sum(1 for c in known if c == 0)),
                "top1_id": top1, "top1_citation_count": citations.get(top1, np.nan) if top1 is not None else np.nan,
                "top1_is_relevant": top1 in qrels[qid] if top1 is not None else False,
                "_known": known})
    per_query = pd.DataFrame(rows)
    summary_rows = []
    for system, part in per_query.groupby("system", sort=False):
        pooled = [c for known in part["_known"] for c in known]
        n_docs = int(part["n_topk"].sum())
        hits = int(part["topk_top_decile_hits"].sum())
        zeros = int(part["topk_zero_citation_hits"].sum())
        share = hits / len(pooled) if pooled else float("nan")
        zero_share = zeros / len(pooled) if pooled else float("nan")
        per_q_mean = part["topk_mean_log10_citations"].dropna().to_numpy(float)
        wrong = part[(part["n_topk"] > 0) & ~part["top1_is_relevant"] & part["top1_citation_count"].notna()
                     & part["target_citation_count"].notna()]
        more = (wrong["top1_citation_count"] > wrong["target_citation_count"]).to_numpy(float)
        less = (wrong["top1_citation_count"] < wrong["target_citation_count"]).to_numpy(float)
        more_mean, more_lo, more_hi = bootstrap_ci(more, n_boot, seed, confidence) if len(more) else (np.nan,) * 3
        summary_rows.append({
            "system": system, "n_queries": len(part), "k": k, "n_docs_topk": n_docs,
            "n_docs_unknown_citation_count": n_docs - len(pooled),
            "mean_log10_citations_topk": float(per_q_mean.mean()) if len(per_q_mean) else np.nan,
            "median_citations_topk": float(np.median(pooled)) if pooled else np.nan,
            "top_decile_share_topk": share, "corpus_top_decile_share": reference["top_decile_share"],
            "top_decile_lift": share / reference["top_decile_share"] if reference["top_decile_share"] else np.nan,
            "zero_citation_share_topk": zero_share, "corpus_zero_citation_share": reference["zero_citation_share"],
            "relevant_mean_log10_citations": float(np.mean(np.log10(1 + part["target_citation_count"].dropna()))) if
            part["target_citation_count"].notna().any() else np.nan,
            "n_wrong_top1": len(wrong), "wrong_top1_more_cited_rate": more_mean,
            "wrong_top1_more_cited_ci_low": more_lo, "wrong_top1_more_cited_ci_high": more_hi,
            "wrong_top1_less_cited_rate": float(less.mean()) if len(less) else np.nan})
    return pd.DataFrame(summary_rows), per_query.drop(columns="_known"), reference
