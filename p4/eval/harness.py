"""Evaluate one or more runs against the same qrels.

    from p4.eval.harness import evaluate_runs, evaluate_run_files

    result = evaluate_run_files({"bm25": "bm25.test.run", "dense": "dense.test.run"},
                                "qrels.test.txt")
    result.summary      # one row per system: mean of every metric
    result.per_query    # one row per (system, query): the raw scores
    result.wide("rr@10")  # queries x systems table, ready for a paired bootstrap

Fairness rules:
  * Every system is scored on EXACTLY the queries in the qrels. A query the
    system returned nothing for scores 0 (it is not dropped, which would
    flatter the system). Queries in the run but not in the qrels are ignored.
  * A run that shares too few queries with the qrels is almost certainly being
    scored against the WRONG SPLIT (for example a validation run against test
    qrels). That would report all-zero scores that look like a terrible system,
    so `evaluate_runs` raises RunQrelsMismatchError instead (see `min_coverage`).
    An EMPTY run (no queries at all) is still allowed: it is a system that
    retrieved nothing and scores 0.
  * Metrics use the ranking as given in the run (no extra truncation), so a
    run that stops at 100 documents and one that lists everything are both
    scored on what they returned. This only matters for MAP, because the
    @K metrics look at the top K only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .metrics import average_precision, ndcg_at_k, recall_at_k, reciprocal_rank
from .trec_io import Qrels, Run, check_run_queries, read_qrels, read_run

DEFAULT_CUTOFFS = (1, 5, 10, 20, 50, 100)
# A run must contain at least this fraction of the judged queries. P2 can legitimately
# return nothing for a few queries (no query word is in the index), but a run that misses
# half or more of the judged queries is treated as the wrong split. 0 switches the check off.
DEFAULT_MIN_COVERAGE = 0.5


class RunQrelsMismatchError(ValueError):
    """A (non-empty) run covers too few of the judged queries: probably the wrong split."""

# Per-query columns:  recall@K, rr@K, ndcg@K for each cutoff K, plus `ap`.
# Summary columns are the same names averaged, with friendlier titles:
#   recall@10 -> Recall@10, rr@10 -> MRR@10, ndcg@10 -> nDCG@10, ap -> MAP.
_SUMMARY_NAMES = {"recall": "Recall", "rr": "MRR", "ndcg": "nDCG"}


@dataclass
class EvalResult:
    per_query: pd.DataFrame            # columns: system, query_id, n_retrieved, metrics...
    summary: pd.DataFrame              # one row per system
    warnings: dict[str, dict[str, list[str]]] = field(default_factory=dict)  # system -> {"missing": [...], "extra": [...]}

    def metric_columns(self) -> list[str]:
        return [c for c in self.per_query.columns if c not in ("system", "query_id", "n_retrieved")]

    def wide(self, metric: str) -> pd.DataFrame:
        """Table with one row per query and one column per system.

        All systems share the same rows (the qrels queries, sorted), so two
        columns can be compared query by query, as a paired bootstrap needs.
        """
        if metric not in self.metric_columns():
            raise KeyError(f"unknown metric {metric!r}; choose from {self.metric_columns()}")
        return self.per_query.pivot(index="query_id", columns="system", values=metric)


def _clean_cutoffs(cutoffs) -> tuple[int, ...]:
    ks = tuple(sorted(set(cutoffs)))
    if not ks or any((not isinstance(k, int)) or k < 1 for k in ks):
        raise ValueError(f"cutoffs must be whole numbers >= 1, got {cutoffs!r}")
    return ks


def check_coverage(system: str, run: Run, qrels: Qrels, min_coverage: float) -> None:
    """Raise RunQrelsMismatchError if a non-empty run covers < min_coverage of the judged queries."""
    if not run or min_coverage <= 0:
        return                      # empty run = retrieved nothing (allowed); 0 = check disabled
    found = len(set(run) & set(qrels))
    coverage = found / len(qrels)
    if coverage < min_coverage:
        missing = sorted(set(qrels) - set(run))[:3]
        extra = sorted(set(run) - set(qrels))[:3]
        raise RunQrelsMismatchError(
            f"system {system!r}: only {found} of {len(qrels)} judged queries ({coverage:.0%}) appear in the run, "
            f"below the required {min_coverage:.0%}. The run has {len(run)} queries (e.g. {extra}) and the qrels "
            f"expect e.g. {missing}. Was the run scored against the wrong split? "
            f"(pass min_coverage=0 to score it anyway)")


def _score_query(ranked: list[str], relevant, cutoffs) -> dict[str, float]:
    row = {}
    for k in cutoffs:
        row[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        row[f"rr@{k}"] = reciprocal_rank(ranked, relevant, k)
        row[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
    row["ap"] = average_precision(ranked, relevant)
    return row


def evaluate_runs(runs: dict[str, Run], qrels: Qrels, cutoffs=DEFAULT_CUTOFFS,
                  min_coverage: float = DEFAULT_MIN_COVERAGE) -> EvalResult:
    """Score every run in `runs` ({system_name: run}) against `qrels`.

    Raises RunQrelsMismatchError (before scoring anything) if any non-empty run
    contains fewer than `min_coverage` of the judged queries.
    """
    if not runs:
        raise ValueError("no runs to evaluate")
    if not qrels:
        raise ValueError("qrels are empty")
    if not 0 <= min_coverage <= 1:
        raise ValueError(f"min_coverage must be between 0 and 1, got {min_coverage!r}")
    cutoffs = _clean_cutoffs(cutoffs)
    for system, run in runs.items():         # fail fast: check every system before scoring
        check_coverage(system, run, qrels, min_coverage)
    query_ids = sorted(qrels)
    rows, warnings = [], {}
    for system, run in runs.items():
        warnings[system] = check_run_queries(run, qrels)
        for qid in query_ids:
            ranked = [doc for doc, _score in run.get(qid, [])]   # missing query -> [] -> all zeros
            rows.append({"system": system, "query_id": qid, "n_retrieved": len(ranked),
                         **_score_query(ranked, qrels[qid], cutoffs)})
    per_query = pd.DataFrame(rows)
    return EvalResult(per_query, _summarise(per_query, list(runs)), warnings)


def _summarise(per_query: pd.DataFrame, system_order: list[str]) -> pd.DataFrame:
    metric_cols = [c for c in per_query.columns if c not in ("system", "query_id", "n_retrieved")]
    means = per_query.groupby("system", sort=False)[metric_cols].mean().reindex(system_order)
    out = pd.DataFrame({
        "n_queries": per_query.groupby("system", sort=False).size().reindex(system_order),
        "n_no_results": per_query[per_query["n_retrieved"] == 0].groupby("system").size()
                        .reindex(system_order, fill_value=0),
    })
    for col in metric_cols:
        out[_summary_name(col)] = means[col]
    return out.reset_index()


def _summary_name(col: str) -> str:
    if col == "ap":
        return "MAP"
    kind, k = col.split("@")
    return f"{_SUMMARY_NAMES[kind]}@{k}"


summary_name = _summary_name      # public alias: per-query column -> friendly name, e.g. "rr@10" -> "MRR@10"


def per_query_column(name: str) -> str:
    """Turn a summary name such as "MRR@10" or "MAP" into the per-query column ("rr@10", "ap").

    Per-query names ("rr@10", "ndcg@10") are returned unchanged, so either style can be used.
    """
    if name.lower() == "map" or name == "ap":
        return "ap"
    kind, sep, k = name.partition("@")
    by_summary = {v.lower(): key for key, v in _SUMMARY_NAMES.items()}     # {"recall": "recall", "mrr": "rr", ...}
    column = by_summary.get(kind.lower(), kind.lower() if kind.lower() in _SUMMARY_NAMES else None)
    if not sep or column is None or not k.isdigit():
        raise ValueError(f"unknown metric name {name!r}; use e.g. MRR@10, Recall@10, nDCG@10, MAP")
    return f"{column}@{k}"


def evaluate_run_files(run_paths: dict[str, str | Path], qrels_path: str | Path,
                       cutoffs=DEFAULT_CUTOFFS, min_coverage: float = DEFAULT_MIN_COVERAGE) -> EvalResult:
    """Convenience wrapper: read the files, then call `evaluate_runs`.

    `run_paths` maps a system name (your label for the table) to a run file.
    """
    runs = {name: read_run(path) for name, path in run_paths.items()}
    return evaluate_runs(runs, read_qrels(qrels_path), cutoffs, min_coverage)
