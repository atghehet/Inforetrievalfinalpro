"""Facts about the citation graph that the report states (direction check, leakage-safe graph size, prior versus popularity).

    python -m p4.experiments.graph_facts --processed data/processed --graph-dir <run_main>/eval_graph --out <NEW json> [--label real]

Everything is computed from P1's tables and the evaluation graph written by p4.graph.eval_graph; nothing is tuned and no run file is read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _spearman(a, b) -> float | None:
    """Spearman correlation, or None when it is undefined (a constant input) so the JSON never contains NaN."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return None
    return float(spearmanr(a[ok], b[ok])[0])


def graph_facts(processed: Path, graph_dir: Path) -> dict:
    papers = pd.read_parquet(processed / "papers.parquet")
    for column, default in (("title", ""), ("year", np.nan), ("citation_count", np.nan)):         # optional: a table without them is still usable
        if column not in papers.columns:
            papers[column] = default
    edges = pd.read_parquet(processed / "edges.parquet")
    queries = pd.read_parquet(processed / "queries.parquet", columns=["query_id", "split", "cited_paper_id", "citing_paper_id"])
    priors = pd.read_parquet(graph_dir / "priors.parquet").set_index("paper_id")
    manifest = json.loads((graph_dir / "MANIFEST.json").read_text(encoding="utf-8"))

    year = papers.set_index("paper_id")["year"]
    ey = pd.DataFrame({"citing": edges["citing_paper_id"].map(year), "cited": edges["cited_paper_id"].map(year)}).dropna()
    direction = {"edges_with_both_years": int(len(ey)),
                 "citing_year_>=_cited_year": float((ey["citing"] >= ey["cited"]).mean()) if len(ey) else None,
                 "citing_year_<_cited_year": float((ey["citing"] < ey["cited"]).mean()) if len(ey) else None}

    full_in = edges.groupby("cited_paper_id").size().reindex(papers["paper_id"], fill_value=0)
    safe_in = priors["citation_count_eval"].reindex(papers["paper_id"])
    nodes = len(papers)
    test_targets = queries[queries["split"] == "test"]["cited_paper_id"]
    val_targets = queries[queries["split"] == "val"]["cited_paper_id"]
    out = {
        "protocol": manifest.get("protocol"), "data_label": manifest.get("data_label"), "hold_out_splits": manifest.get("hold_out_splits"),
        "nodes": nodes, "edges_full": int(len(edges)), "edge_removal": manifest["edge_removal"],
        "direction_check": direction,
        "papers_never_cited_full_graph": float((full_in == 0).mean()),
        "papers_never_cited_safe_graph": float((safe_in == 0).mean()),
        "test_targets": {"n_queries": int(len(test_targets)), "full_count_zero": float((full_in.reindex(test_targets).to_numpy() == 0).mean()),
                         "safe_count_zero": float((safe_in.reindex(test_targets).to_numpy() == 0).mean()),
                         "safe_count_mean": float(safe_in.reindex(test_targets).mean()), "full_count_mean": float(full_in.reindex(test_targets).mean())},
        "val_targets": {"n_queries": int(len(val_targets)), "safe_count_zero": float((safe_in.reindex(val_targets).to_numpy() == 0).mean())},
        "pagerank": manifest["pagerank"],
        "spearman_pagerank_d085_vs_safe_count": _spearman(priors["pagerank_d085"], priors["citation_count_eval"]),
        "spearman_safe_count_vs_papers_citation_count": _spearman(safe_in, papers.set_index("paper_id")["citation_count"].reindex(safe_in.index)),
        "max_pagerank_d085": float(priors["pagerank_d085"].max()), "min_pagerank_d085": float(priors["pagerank_d085"].min()),
    }
    top = priors["pagerank_d085"].sort_values(ascending=False).head(10)
    titles = papers.set_index("paper_id")["title"]
    out["top10_pagerank_d085"] = [{"paper_id": p, "pagerank": float(v), "safe_count": int(priors.loc[p, "citation_count_eval"]),
                                   "title": str(titles.get(p, ""))[:90]} for p, v in top.items()]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--graph-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"error: {args.out} already exists; refusing to overwrite", file=sys.stderr)
        return 2
    try:
        facts = graph_facts(args.processed, args.graph_dir)
    except (FileNotFoundError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
