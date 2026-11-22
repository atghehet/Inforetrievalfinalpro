"""Build the leakage-safe EVALUATION GRAPH and the authority priors derived from it.

    python -m p4.graph.eval_graph --processed <P1 data/processed> --out <NEW folder> --data-label real|synthetic|demo

Protocol (PROPOSED, pending team approval; see p4/report/proposals/leakage_safe_graph_protocol.md)
  1. Drop every edge whose CITING paper has a query in a held-out split (default: val and test). P1 assigns splits per
     citing paper, so this removes every edge that could have produced a held-out query.
  2. From the remaining edges compute, over ALL papers of papers.parquet (isolated papers included):
       citation_count_eval   in-corpus citing papers per paper (the leakage-safe replacement for papers.citation_count)
       pagerank_d050 / _d085 / _d095   PageRank with damping 0.5, 0.85, 0.95 (the plan's three values; --dampings changes them)
  3. Optionally (`--strict-index`) write index_contexts_strict.parquet: the index without any context written by a
     held-out citing paper (the "strict" setting of the index sensitivity analysis).

Outputs (all in the new folder; an existing non-empty folder is refused):
    priors.parquet            paper_id + the columns above
    edges_eval.parquet        the evaluation graph (citing_paper_id, cited_paper_id)
    MANIFEST.json             protocol name, hold-out splits, counts, PageRank convergence, input file hashes, data label
    index_contexts_strict.parquet   only with --strict-index

The manifest is read by the fusion tool, which refuses a prior that was not built with every split it is evaluated on held out.
Nothing here reads or writes anything in the repository or in P1's folders.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from .leakage_audit import DST, HELD_OUT, SRC, citation_counts, leakage_safe_edges
from .pagerank import pagerank

PROTOCOL = "leakage-safe-v1 (proposed, pending team approval)"
DEFAULT_DAMPINGS = (0.5, 0.85, 0.95)
DATA_LABELS = ("real", "synthetic", "demo")
PRIOR_COLUMNS_PREFIX = "pagerank_d"


class EvalGraphError(ValueError):
    """The processed folder is unusable or the output folder is not safe to write."""


def damping_column(d: float) -> str:
    """0.5 -> 'pagerank_d050', 0.85 -> 'pagerank_d085', 0.95 -> 'pagerank_d095'."""
    return f"{PRIOR_COLUMNS_PREFIX}{int(round(d * 100)):03d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(processed: Path, name: str, columns: list[str]) -> pd.DataFrame:
    path = processed / name
    if not path.is_file():
        raise EvalGraphError(f"{name} not found in {processed}")
    try:
        return pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError, OSError) as err:
        raise EvalGraphError(f"{name}: cannot read columns {columns} ({err})") from None


def build(processed_dir, out_dir, *, data_label: str, dampings=DEFAULT_DAMPINGS, hold_out=HELD_OUT,
          strict_index: bool = False) -> dict:
    """Build every artefact and return the manifest. Refuses a non-empty output folder."""
    if data_label not in DATA_LABELS:
        raise EvalGraphError(f"data label must be one of {DATA_LABELS}, got {data_label!r}")
    processed, out = Path(processed_dir), Path(out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise EvalGraphError(f"output folder exists and is not empty, refusing to overwrite: {out}")
    for d in dampings:
        if not 0 <= d < 1:
            raise EvalGraphError(f"damping must be in [0, 1), got {d}")
    hold_out = tuple(hold_out)
    if not hold_out or not set(hold_out) <= {"train", "val", "test"}:
        raise EvalGraphError(f"hold-out splits must be a non-empty subset of train, val, test; got {hold_out}")

    papers = _read(processed, "papers.parquet", ["paper_id"])
    queries = _read(processed, "queries.parquet", ["query_id", "split", SRC, DST])
    edges = _read(processed, "edges.parquet", [SRC, DST])
    safe, removal = leakage_safe_edges(edges, queries, hold_out)
    ids = papers["paper_id"].astype(str)

    priors = pd.DataFrame({"paper_id": ids})
    priors["citation_count_eval"] = ids.map(citation_counts(safe)).fillna(0).astype(int).to_numpy()
    pagerank_info = {}
    for d in dampings:
        result = pagerank(safe, nodes=ids, damping=d)          # SRC/DST are PageRank's default column names
        priors[damping_column(d)] = ids.map(result.scores).to_numpy()
        pagerank_info[damping_column(d)] = {"damping": d, "iterations": result.iterations, "converged": result.converged,
                                            "final_delta": result.final_delta, "n_nodes": result.n_nodes,
                                            "n_edges": result.n_edges}

    out.mkdir(parents=True, exist_ok=True)
    priors.to_parquet(out / "priors.parquet", index=False)
    safe.to_parquet(out / "edges_eval.parquet", index=False)
    strict_info = None
    if strict_index:
        index_contexts = pd.read_parquet(processed / "index_contexts.parquet")
        held = set(queries.loc[queries["split"].isin(hold_out), SRC])
        strict = index_contexts[~index_contexts[SRC].isin(held)].reset_index(drop=True)
        strict.to_parquet(out / "index_contexts_strict.parquet", index=False)
        strict_info = {"index_contexts_total": int(len(index_contexts)), "removed": int(len(index_contexts) - len(strict)),
                       "kept": int(len(strict))}
    manifest = {
        "protocol": PROTOCOL, "data_label": data_label, "hold_out_splits": list(hold_out),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "processed_dir": str(processed), "papers": int(len(papers)), "queries_per_split": queries["split"].value_counts().sort_index().to_dict(),
        "edge_removal": removal, "pagerank": pagerank_info, "prior_columns": [c for c in priors.columns if c != "paper_id"],
        "papers_with_zero_eval_citations": int((priors["citation_count_eval"] == 0).sum()),
        "strict_index": strict_info,
        "input_sha256": {name: _sha256(processed / name) for name in ("papers.parquet", "queries.parquet", "edges.parquet")
                         if (processed / name).is_file()},
        "note": "citation_count_eval and the PageRank columns exclude every edge of a citing paper that has a query in a held-out split"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_priors(folder) -> tuple[pd.DataFrame, dict]:
    """Read priors.parquet and MANIFEST.json written by `build`."""
    folder = Path(folder)
    if not (folder / "priors.parquet").is_file() or not (folder / "MANIFEST.json").is_file():
        raise EvalGraphError(f"{folder} does not look like an evaluation-graph folder (priors.parquet / MANIFEST.json missing)")
    return pd.read_parquet(folder / "priors.parquet"), json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", required=True, type=Path, help="P1's data/processed folder (read-only)")
    p.add_argument("--out", required=True, type=Path, help="NEW (or empty) output folder")
    p.add_argument("--data-label", required=True, choices=DATA_LABELS)
    p.add_argument("--dampings", default="0.5,0.85,0.95", help="comma-separated, default %(default)s")
    p.add_argument("--hold-out", default="val,test", help="splits whose citing papers' edges are removed, default %(default)s")
    p.add_argument("--strict-index", action="store_true", help="also write the strict index (no contexts by held-out citing papers)")
    args = p.parse_args(argv)
    try:
        dampings = [float(x) for x in args.dampings.split(",") if x.strip()]
        manifest = build(args.processed, args.out, data_label=args.data_label, dampings=dampings,
                         hold_out=[s.strip() for s in args.hold_out.split(",") if s.strip()], strict_index=args.strict_index)
    except (EvalGraphError, ValueError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    r = manifest["edge_removal"]
    print(f"[{args.data_label}] evaluation graph: {r['edges_kept']} of {r['edges_total']} edges kept "
          f"({r['edges_removed']} removed for {r['held_out_citing_papers']} held-out citing papers); "
          f"priors: {', '.join(manifest['prior_columns'])}\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
