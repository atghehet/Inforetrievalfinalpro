"""End-to-end data build.

    python -m citeds.build                       # reads data/raw/, writes data/processed/
    python -m citeds.build --raw path/to/unarXive --field cs.CL --max-papers 50000

Outputs (see citeds/schema.py for columns):
    papers.parquet, contexts.parquet, index_contexts.parquet, edges.parquet,
    queries.parquet, qrels/qrels.<split>.txt, topics/topics.<split>.tsv,
    docs/dataset_stats.md, data/processed/build_info.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pyarrow.parquet as pq

from . import config, schema
from .leakage import assert_no_leakage, build_index_contexts
from .splits import assign_query_ids, make_queries, write_trec_files
from .stats import compute_stats
from .tables import build_tables
from .unarxive import iter_papers

log = logging.getLogger("citeds.build")


def write(df, path: Path, sch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(schema.conform(df, sch), path)
    log.info("wrote %s (%d rows)", path, len(df))


def run(raw_dir: Path, field: str, max_papers: int, seed: int, max_queries: int) -> dict:
    t0 = time.time()
    config.ensure_dirs()
    log.info("reading %s (field=%s, max_papers=%d)", raw_dir, field, max_papers)
    papers, contexts, edges, clog = build_tables(iter_papers(raw_dir, field), max_papers)

    queries = make_queries(contexts, papers, seed=seed, max_queries=max_queries)
    index_contexts = build_index_contexts(contexts, queries)

    # Drop queries whose relevant paper lost all its index contexts to near-dup
    # removal (rare). Their sentences stay excluded from the index, which is safe.
    remaining = index_contexts.groupby("cited_paper_id").size()
    keep = queries["cited_paper_id"].map(remaining).fillna(0) >= config.MIN_REMAINING_CONTEXTS
    clog["queries_dropped_no_index_context"] = int((~keep).sum())
    queries = assign_query_ids(queries[keep])

    assert_no_leakage(index_contexts, queries)
    log.info("leakage check passed")

    write(papers, config.PAPERS_PATH, schema.PAPERS)
    write(contexts, config.CONTEXTS_PATH, schema.CONTEXTS)
    write(index_contexts, config.INDEX_CONTEXTS_PATH, schema.CONTEXTS)
    write(edges, config.EDGES_PATH, schema.EDGES)
    write(queries, config.QUERIES_PATH, schema.QUERIES)
    write_trec_files(queries)

    config.STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.STATS_PATH.write_text(compute_stats(papers, contexts, index_contexts, edges, queries, clog))
    info = {"raw_dir": str(raw_dir), "field": field, "max_papers": max_papers, "seed": seed,
            "max_queries": max_queries, "seconds": round(time.time() - t0, 1),
            "counts": {"papers": len(papers), "contexts": len(contexts),
                       "index_contexts": len(index_contexts), "edges": len(edges),
                       "queries": len(queries)},
            "cleaning_log": clog}
    (config.PROCESSED_DIR / "build_info.json").write_text(json.dumps(info, indent=2))
    log.info("done in %.1fs: %s", info["seconds"], info["counts"])
    return info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=config.RAW_DIR, help="folder with unarXive JSONL files")
    ap.add_argument("--field", default=config.FIELD)
    ap.add_argument("--max-papers", type=int, default=config.MAX_PAPERS)
    ap.add_argument("--max-queries", type=int, default=config.MAX_QUERIES)
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(a.raw, a.field, a.max_papers, a.seed, a.max_queries)


if __name__ == "__main__":
    main()
