"""Leakage audit for graph-based and popularity-based signals (read-only on P1's tables).

The problem. P1 removes each query SENTENCE from the index (leakage control). But `edges.parquet` and
`papers.citation_count` are built from ALL cleaned contexts, including the query sentences. So the edge
"citing paper -> cited paper" that the query came from is in the graph, and it raises the target's in-degree,
PageRank and citation count. A prior computed from that graph rewards exactly the paper the query asks for.

How P1 splits (citeds/splits.py): the split is a hash of (seed, citing paper id), so ALL queries of one citing paper
are in the same split. That makes a clean rule possible.

Leakage-safe rule (the "evaluation graph"): drop every edge whose CITING paper is the citing paper of any validation
or test query. Then no edge that could have produced a held-out query is in the graph. The same rule gives
leakage-safe citation counts (`safe_citation_count`) for popularity buckets and priors.

What this module provides
  * split_audit               are the splits really disjoint by citing paper?
  * leakage_safe_edges        the evaluation graph (and what was removed)
  * citation_counts           in-corpus citing papers per paper, for any edge table
  * popularity_prior_diagnostic  how well a QUERY-INDEPENDENT "most cited first" prior finds the targets with the full
                              graph versus the safe graph. Any gap is leakage. This is a diagnostic, not a retrieval result.
  * same_citer_index_contexts residual risk: index contexts written by the query's own citing paper about the target
  * audit_context_file        gate for the context files given to P2/P3 (JSONL with context_id, cited_paper_id, sentence):
                              every row must be a leakage-controlled index context, never a query context or query sentence
  * audit                     all of the above for a P1 processed folder

Not covered (documented limitations): temporal leakage (the graph contains citations from papers published after the
query's citing paper) and near-duplicates of query sentences, which P1's own `assert_no_leakage` checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HELD_OUT = ("val", "test")
SRC, DST = "citing_paper_id", "cited_paper_id"


def split_audit(queries: pd.DataFrame) -> dict:
    """Are the splits disjoint by citing paper? (They are, if P1 assigns splits per citing paper.)"""
    per_paper = queries.groupby(SRC)["split"].nunique()
    shared = per_paper[per_paper > 1]
    return {"queries": int(len(queries)), "queries_per_split": queries["split"].value_counts().sort_index().to_dict(),
            "citing_papers": int(per_paper.size),
            "citing_papers_per_split": queries.groupby("split")[SRC].nunique().sort_index().to_dict(),
            "citing_papers_in_more_than_one_split": int(len(shared)), "disjoint_by_citing_paper": bool(shared.empty),
            "shared_examples": shared.index[:5].tolist()}


def leakage_safe_edges(edges: pd.DataFrame, queries: pd.DataFrame, hold_out=HELD_OUT) -> tuple[pd.DataFrame, dict]:
    """Edges minus every edge whose citing paper has a query in one of the `hold_out` splits."""
    held = set(queries.loc[queries["split"].isin(hold_out), SRC])
    removed = edges[SRC].isin(held)
    stats = {"hold_out_splits": list(hold_out), "held_out_citing_papers": len(held), "edges_total": int(len(edges)),
             "edges_removed": int(removed.sum()), "edges_kept": int((~removed).sum()),
             "share_removed": float(removed.mean()) if len(edges) else 0.0}
    return edges[~removed].reset_index(drop=True), stats


def citation_counts(edges: pd.DataFrame) -> pd.Series:
    """Number of distinct citing papers per cited paper (the definition of papers.citation_count)."""
    return edges.drop_duplicates([SRC, DST]).groupby(DST).size().rename("citation_count")


def _expected_ranks(scores: pd.Series, targets: list[str]) -> np.ndarray:
    """Rank of each target when all papers are sorted by score (high first); ties share their average position."""
    values = scores.to_numpy(float)
    ranks = []
    for target in targets:
        s = scores.get(target, 0.0)
        higher = int((values > s).sum())
        tied = int((values == s).sum())                               # includes the target itself
        ranks.append(higher + (tied + 1) / 2)
    return np.array(ranks)


def popularity_prior_diagnostic(queries: pd.DataFrame, papers: pd.DataFrame, edges_full: pd.DataFrame,
                                edges_safe: pd.DataFrame, split: str = "test", k: int = 10) -> dict:
    """Rank ALL papers by citation count (the same list for every query) and see where the targets land.

    With a leak-free prior this should be poor; if the full graph does much better, its counts contain the answer.
    """
    part = queries[queries["split"] == split]
    targets = part[DST].tolist()
    result = {"split": split, "n_queries": len(part), "n_papers": int(len(papers))}
    for name, edges in (("full_graph", edges_full), ("safe_graph", edges_safe)):
        counts = citation_counts(edges).reindex(papers["paper_id"]).fillna(0)
        ranks = _expected_ranks(counts, targets)
        result[name] = {"mrr": float((1 / ranks).mean()) if len(ranks) else None,
                        "recall_at_k": float((ranks <= k).mean()) if len(ranks) else None,
                        "median_rank": float(np.median(ranks)) if len(ranks) else None,
                        "targets_with_zero_citations": float((counts.reindex(targets).fillna(0) == 0).mean()) if targets else None}
    return result


def same_citer_index_contexts(queries: pd.DataFrame, index_contexts: pd.DataFrame, split: str = "test") -> dict:
    """Residual risk: other sentences by the query's own citing paper about the same target remain in the index."""
    part = queries[queries["split"] == split][["query_id", SRC, DST]]
    pairs = index_contexts[[SRC, DST]].drop_duplicates().assign(_hit=True)
    merged = part.merge(pairs, on=[SRC, DST], how="left")
    n = int(merged["_hit"].fillna(False).astype(bool).sum())
    return {"split": split, "n_queries": len(part), "queries_with_other_sentences_by_same_citing_paper": n,
            "share": n / len(part) if len(part) else None}


def audit_context_file(path, index_contexts: pd.DataFrame, queries: pd.DataFrame) -> dict:
    """Check a JSONL context file that P2/P3 index (one JSON object per line: context_id, cited_paper_id, sentence).

    SERIOUS problems (make `ok` False): a context_id that is not in the leakage-controlled index; a context_id that is a query's
    own context; a sentence equal to a query text (after the same normalisation P1 uses). Informational: index contexts missing
    from the file, and duplicated context_ids.
    """
    from citeds.clean import normalise_for_match                  # P1's normalisation, imported read-only
    index_ids = set(index_contexts["context_id"])
    query_context_ids = set(queries["context_id"]) if "context_id" in queries.columns else set()
    query_texts = {normalise_for_match(x) for x in queries["text"].dropna()} if "text" in queries.columns else set()
    n = not_in_index = query_ctx = same_text = 0
    seen: set[str] = set()
    duplicates = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            cid = str(row.get("context_id"))
            duplicates += cid in seen
            seen.add(cid)
            not_in_index += cid not in index_ids
            query_ctx += cid in query_context_ids
            same_text += normalise_for_match(str(row.get("sentence", ""))) in query_texts
    return {"file": str(path), "rows": n, "context_ids_not_in_index": not_in_index, "query_context_ids": query_ctx,
            "sentences_equal_to_a_query_text": same_text, "duplicated_context_ids": duplicates,
            "index_contexts_missing_from_file": len(index_ids - seen),
            "ok": not (not_in_index or query_ctx or same_text)}


def audit(processed_dir, k: int = 10, context_files=()) -> dict:
    """Run every check on a P1 processed folder (papers, queries, edges, index_contexts parquet files)."""
    d = Path(processed_dir)
    papers = pd.read_parquet(d / "papers.parquet", columns=["paper_id", "citation_count"])
    queries = pd.read_parquet(d / "queries.parquet", columns=["query_id", "split", SRC, DST, "context_id"])
    edges = pd.read_parquet(d / "edges.parquet", columns=[SRC, DST])
    index_contexts = pd.read_parquet(d / "index_contexts.parquet", columns=["context_id", SRC, DST])
    safe, removal = leakage_safe_edges(edges, queries)
    full_counts, safe_counts = citation_counts(edges), citation_counts(safe)
    held = queries[queries["split"].isin(HELD_OUT)]
    own_edge_present = held.merge(edges.drop_duplicates([SRC, DST]).assign(_e=True), on=[SRC, DST], how="left")["_e"].notna().mean()
    targets = held[DST]
    report = {
        "split_audit": split_audit(queries),
        "held_out_queries_whose_own_edge_is_in_the_full_graph": float(own_edge_present) if len(held) else None,
        "edge_removal_for_the_evaluation_graph": removal,
        "papers_citation_count_equals_full_graph_in_degree": bool(
            (papers.set_index("paper_id")["citation_count"].reindex(full_counts.index).fillna(0).astype(int) == full_counts).all()),
        "held_out_targets": {"n": int(len(targets)),
                             "mean_citation_count_full": float(full_counts.reindex(targets).fillna(0).mean()) if len(targets) else None,
                             "mean_citation_count_safe": float(safe_counts.reindex(targets).fillna(0).mean()) if len(targets) else None,
                             "share_with_zero_citations_safe": float((safe_counts.reindex(targets).fillna(0) == 0).mean()) if len(targets) else None},
        "popularity_prior_diagnostic": {s: popularity_prior_diagnostic(queries, papers, edges, safe, s, k) for s in HELD_OUT},
        "same_citing_paper_index_contexts": {s: same_citer_index_contexts(queries, index_contexts, s) for s in HELD_OUT},
    }
    if context_files:
        with_text = pd.read_parquet(d / "queries.parquet")
        report["context_files"] = [audit_context_file(f, index_contexts, with_text) for f in context_files]
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", required=True, type=Path, help="P1's data/processed folder")
    p.add_argument("--out", required=True, type=Path, help="JSON report to write (must not exist)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--context-file", action="append", default=[], type=Path,
                   help="a JSONL context file given to P2/P3 (repeatable): checked against the index and the queries")
    args = p.parse_args(argv)
    if args.out.exists():
        print(f"error: {args.out} already exists", file=sys.stderr)
        return 2
    try:
        report = audit(args.processed, args.k, args.context_file)
    except (FileNotFoundError, KeyError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
