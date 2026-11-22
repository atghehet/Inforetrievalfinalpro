"""Hand-made DEMO fixtures for the evaluation-graph and fusion tests (NOT a test file). Everything here is invented.

Two layouts
  small_processed(folder)   the 5-query example of test_leakage_audit written as a P1 `processed` folder
  leak_scenario(folder)     the controlled leak scenario below

Leak scenario. 20 validation and 20 test queries. Query i of a split has citing paper C<i> and target T<i>. The ONLY citation of every
target is the query's own edge C<i> -> T<i>. Distractors D01..D30 are cited by nobody.
  * Leaky graph (the queries' edges are present): every target has 1 citation, every distractor 0.
  * Safe graph (edges of held-out citing papers removed): every paper has 0 citations, the prior carries no information.
The base run lists 12 papers per query with strictly decreasing scores; the target is at rank 4. So with a safe (or no) prior the
reciprocal rank is exactly 1/4, and a prior that sees the leaked edge can lift every target to rank 1.
"""
import numpy as np
import pandas as pd

N_PER_SPLIT = 20
TARGET_RANK = 4
N_DOCS = 12


def small_processed(folder):
    folder.mkdir(parents=True, exist_ok=True)
    queries = pd.DataFrame({
        "query_id": ["q1", "q2", "q3", "q4", "q5"], "split": ["train", "val", "test", "test", "train"],
        "citing_paper_id": ["C1", "C2", "C3", "C3", "C4"], "cited_paper_id": ["T1", "T1", "T2", "T3", "T2"],
        "context_id": ["c1", "c2", "c3", "c4", "c5"], "text": ["sentence one", "sentence two", "sentence three", "sentence four", "sentence five"]})
    edges = pd.DataFrame({"citing_paper_id": ["C1", "C2", "C3", "C3", "C4", "C5", "C2"],
                          "cited_paper_id": ["T1", "T1", "T2", "T3", "T2", "T1", "T4"], "num_contexts": 1})
    papers = pd.DataFrame({"paper_id": ["T1", "T2", "T3", "T4", "T5", "C1", "C2", "C3", "C4", "C5"]})
    index_contexts = pd.DataFrame({"context_id": ["x1", "x2", "x3", "x4"], "citing_paper_id": ["C3", "C9", "C1", "C2"],
                                   "cited_paper_id": ["T2", "T3", "T1", "T1"], "text": ["a", "b", "c", "d"]})
    queries.to_parquet(folder / "queries.parquet", index=False)
    edges.to_parquet(folder / "edges.parquet", index=False)
    papers.to_parquet(folder / "papers.parquet", index=False)
    index_contexts.to_parquet(folder / "index_contexts.parquet", index=False)
    return folder


def leak_tables():
    rows, edges = [], []
    for split, prefix in (("val", "v"), ("test", "t")):
        for i in range(1, N_PER_SPLIT + 1):
            rows.append({"query_id": f"{prefix}{i:02d}", "split": split, "citing_paper_id": f"C{prefix}{i:02d}",
                         "cited_paper_id": f"T{prefix}{i:02d}", "context_id": f"ctx{prefix}{i:02d}", "text": f"query {prefix}{i}"})
            edges.append({"citing_paper_id": f"C{prefix}{i:02d}", "cited_paper_id": f"T{prefix}{i:02d}", "num_contexts": 1})
    queries, edges = pd.DataFrame(rows), pd.DataFrame(edges)
    paper_ids = ([f"T{p}{i:02d}" for p in "vt" for i in range(1, N_PER_SPLIT + 1)] + [f"D{j:02d}" for j in range(1, 31)]
                 + [f"C{p}{i:02d}" for p in "vt" for i in range(1, N_PER_SPLIT + 1)])
    papers = pd.DataFrame({"paper_id": paper_ids})
    papers["citation_count"] = papers["paper_id"].map(edges.groupby("cited_paper_id").size()).fillna(0).astype(int)   # P1: full-graph in-degree
    index_contexts = pd.DataFrame({"context_id": ["z1"], "citing_paper_id": ["Zother"], "cited_paper_id": ["D01"], "text": ["unrelated"]})
    return papers, queries, edges, index_contexts


def leak_scenario(folder):
    folder.mkdir(parents=True, exist_ok=True)
    papers, queries, edges, index_contexts = leak_tables()
    papers.to_parquet(folder / "papers.parquet", index=False)
    queries.to_parquet(folder / "queries.parquet", index=False)
    edges.to_parquet(folder / "edges.parquet", index=False)
    index_contexts.to_parquet(folder / "index_contexts.parquet", index=False)
    return folder


def leak_run_and_qrels(split):
    """(run, qrels) for the 'v' (val) or 't' (test) queries: target at rank 4, scores strictly decreasing."""
    prefix = "v" if split == "val" else "t"
    run, qrels = {}, {}
    for i in range(1, N_PER_SPLIT + 1):
        qid, target = f"{prefix}{i:02d}", f"T{prefix}{i:02d}"
        docs = [f"D{j:02d}" for j in range(1, N_DOCS)]
        docs.insert(TARGET_RANK - 1, target)
        run[qid] = [(d, 10.0 - 0.01 * r) for r, d in enumerate(docs, start=1)]
        qrels[qid] = {target: 1}
    return run, qrels


def write_run_file(path, run, tag="base"):
    with open(path, "w", encoding="utf-8", newline=chr(10)) as fh:
        for qid in sorted(run):
            for rank, (doc, score) in enumerate(run[qid], start=1):
                fh.write(f"{qid} Q0 {doc} {rank} {score} {tag}" + chr(10))


def write_qrels_file(path, qrels):
    path.write_text("".join(f"{q} 0 {next(iter(d))} 1" + chr(10) for q, d in qrels.items()))
