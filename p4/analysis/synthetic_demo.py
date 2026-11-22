"""Regenerate P1's SYNTHETIC sample tables so the analyses can be demonstrated and tested.

    python -m p4.analysis.synthetic_demo --out p4/outputs/synthetic_inputs

The data are random text produced by scripts/make_sample.py (P1) and processed by P1's own build functions, in memory,
with the recipe that reproduces P3's committed synthetic results (400 papers, seed 0, 200 queries). Writes
papers.parquet, queries.parquet, index_contexts.parquet, qrels.val.txt, qrels.test.txt and a SYNTHETIC_DATA.txt marker.
Nothing about real retrieval can be learned from these files. No citation-function labels are written: they would have to
be invented, because P1's classifier needs the SciCite download.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
MARKER = ("SYNTHETIC DATA: random text from scripts/make_sample.py (P1), processed by P1's build functions.\n"
          "These files demonstrate the P4 analysis code. They say nothing about real retrieval.\n")


def synthetic_all(n_papers: int = 400, seed: int = 0, max_queries: int = 200) -> dict:
    """All of P1's SYNTHETIC tables as DataFrames: papers, contexts, index_contexts, edges, queries (P1's code, read-only use)."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import make_sample
    from citeds import config
    from citeds.leakage import build_index_contexts
    from citeds.splits import assign_query_ids, make_queries
    from citeds.tables import build_tables
    from citeds.unarxive import parse_record

    records = make_sample.make(n_papers, seed=seed)
    parsed = [parse_record(r) for r in records if "cs.CL" in r["metadata"]["categories"].split()]
    papers, contexts, edges, _log = build_tables(parsed)
    queries = make_queries(contexts, papers, seed=config.SEED, max_queries=max_queries)
    index_contexts = build_index_contexts(contexts, queries)
    remaining = index_contexts.groupby("cited_paper_id").size()
    queries = assign_query_ids(queries[queries["cited_paper_id"].map(remaining).fillna(0) >= config.MIN_REMAINING_CONTEXTS])
    return {"papers": papers, "contexts": contexts, "index_contexts": index_contexts, "edges": edges, "queries": queries}


def synthetic_tables(n_papers: int = 400, seed: int = 0, max_queries: int = 200):
    """Return (papers, queries, index_contexts) as DataFrames."""
    tables = synthetic_all(n_papers, seed, max_queries)
    return tables["papers"], tables["queries"], tables["index_contexts"]


def write_synthetic_processed(out_dir, **kwargs) -> dict:
    """Write a SYNTHETIC folder in P1's `data/processed` layout (papers, contexts, index_contexts, edges, queries, qrels)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = synthetic_all(**kwargs)
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    queries = tables["queries"]
    (out / "qrels").mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        part = queries[queries["split"] == split]
        lines = [f"{q} 0 {d} 1" for q, d in zip(part["query_id"], part["cited_paper_id"])]
        (out / "qrels" / f"qrels.{split}.txt").write_text("".join(line + chr(10) for line in lines))
    (out / "SYNTHETIC_DATA.txt").write_text(MARKER)
    return {k: len(v) for k, v in tables.items()}


def write_synthetic_inputs(out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    papers, queries, index_contexts = synthetic_tables()
    papers.to_parquet(out / "papers.parquet", index=False)
    queries.to_parquet(out / "queries.parquet", index=False)
    index_contexts[["context_id", "cited_paper_id"]].to_parquet(out / "index_contexts.parquet", index=False)
    for split in ("val", "test"):
        part = queries[queries["split"] == split]
        (out / f"qrels.{split}.txt").write_text("".join(f"{q} 0 {d} 1\n" for q, d in zip(part["query_id"], part["cited_paper_id"])))
    (out / "SYNTHETIC_DATA.txt").write_text(MARKER)
    return {"papers": len(papers), "queries": len(queries), "val": int((queries["split"] == "val").sum()),
            "test": int((queries["split"] == "test").sum())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--layout", choices=("inputs", "processed"), default="inputs",
                   help="inputs: the small set used by the analysis demos; processed: a full P1 data/processed-style folder "
                        "(papers, contexts, index_contexts, edges, queries, qrels) for the evaluation-graph and fusion demos")
    args = p.parse_args(argv)
    counts = write_synthetic_processed(args.out) if args.layout == "processed" else write_synthetic_inputs(args.out)
    print(f"Wrote SYNTHETIC {args.layout} to {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
