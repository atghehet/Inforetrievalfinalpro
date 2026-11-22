"""Shared paths and helpers for P3 scripts.

All paths come from citeds.config (repo-relative, overridable with the
CITEDS_DATA environment variable), so the scripts run on any machine.
Run P3 scripts from the repository root, e.g.  python p3/src/dense_baseline.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from citeds import config, io  # noqa: E402

RUNS_DIR = REPO_ROOT / "runs"
P2_RUNS_DIR = RUNS_DIR / "p2"                 # P2 experiment output (--out)
DENSE_DIR = RUNS_DIR / "p3" / "dense"
HYBRID_DIR = RUNS_DIR / "p3" / "hybrid"
CACHE_DIR = REPO_ROOT / "cache" / "p3"
SPLITS = ("val", "test")                     # P1 split names; P2 calls val "validation"
P2_SPLIT = {"val": "validation", "test": "test"}
TOP_K = 1000                                  # cap per query: keeps run files small


def qrels_path(split: str) -> Path:
    return io.qrels_path(split)


def load_qrels(path: Path) -> dict[str, set[str]]:
    rel: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                rel.setdefault(p[0], set()).add(p[2])
    return rel


def load_run(path: Path) -> dict[str, dict[str, int]]:
    """qid -> {docid: rank}."""
    run: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 6:
                run.setdefault(p[0], {})[p[2]] = int(p[3])
    return run


def write_run(path: Path, ranked: dict[str, list[tuple[str, float]]], tag: str, top_k: int = TOP_K) -> None:
    """ranked: qid -> [(docid, score), ...] sorted best first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for qid in sorted(ranked):
            for rank, (did, score) in enumerate(ranked[qid][:top_k], start=1):
                fh.write(f"{qid} Q0 {did} {rank} {score:.6f} {tag}\n")


def evaluate(run: dict[str, dict[str, int]], rel: dict[str, set[str]]) -> dict[str, float]:
    """MRR@10, Recall@10, Recall@100, nDCG@10 over all judged queries (missing = 0)."""
    mrr = r10 = r100 = ndcg = 0.0
    for qid, relevant in rel.items():
        ranks = [r for d, r in run.get(qid, {}).items() if d in relevant]
        if not ranks:
            continue
        r = min(ranks)
        if r <= 10:
            mrr += 1 / r
            r10 += 1
            ndcg += 1 / math.log2(r + 1)
        if r <= 100:
            r100 += 1
    n = max(len(rel), 1)
    return {"queries": len(rel), "MRR@10": mrr / n, "Recall@10": r10 / n,
            "Recall@100": r100 / n, "nDCG@10": ndcg / n}


def rrf(runs: list[dict[str, dict[str, int]]], k: int) -> dict[str, list[tuple[str, float]]]:
    """Reciprocal Rank Fusion of several runs; ties broken by paper ID."""
    fused: dict[str, list[tuple[str, float]]] = {}
    for qid in set().union(*runs):
        scores: dict[str, float] = {}
        for run in runs:
            for did, rank in run.get(qid, {}).items():
                scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
        fused[qid] = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused
