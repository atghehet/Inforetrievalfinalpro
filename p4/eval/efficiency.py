"""Efficiency comparison from P2's experiment logs.

P2's `python -m p2.experiment` writes two logs (see lexical/p2/experiment.py):

  efficiency.csv   index, build_seconds, bytes
                   one row per index variant: wall-clock build time and file size
  tuning.csv       run, variant, model, parameters, validation_mrr_at_10, median_ms, mean_ms
                   one row per tried configuration: search time per query on the
                   validation queries (excluding opening the index and writing results)

This module only READS and JOINS those numbers. What the logs do NOT contain is
not invented: there is no memory measurement, no test-set timing, and no timing at
all for P3's dense and hybrid systems. Timings come from whatever machine P2 used,
so compare them with each other, not with numbers from another computer.

The two files are joined on the index name: a `run` in tuning.csv is
"<index name>-<model>-<parameter set number>", e.g. "context-w0-plain-all-bm25-0"
belongs to the index "context-w0-plain-all".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

EFFICIENCY_COLUMNS = ("index", "build_seconds", "bytes")
TUNING_COLUMNS = ("run", "variant", "model", "parameters", "validation_mrr_at_10", "median_ms", "mean_ms")
NOT_RECORDED = ("memory usage", "test-set query timing", "timing for P3 dense/hybrid systems")


class EfficiencyFormatError(ValueError):
    """An efficiency or tuning CSV is missing, empty or does not have P2's columns."""


@dataclass
class EfficiencySummary:
    index_cost: pd.DataFrame          # one row per index: build_seconds, bytes, size_mib
    runs: pd.DataFrame | None         # one row per tuned configuration, joined with its index cost
    best_per_variant: pd.DataFrame | None   # per (variant, model): the configuration with the best validation MRR@10
    notes: list[str] = field(default_factory=list)


def _read_csv(path, required, label) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as err:
        raise EfficiencyFormatError(f"{path.name}: cannot read as CSV ({err})") from None
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise EfficiencyFormatError(f"{path.name}: missing column(s) {missing}; "
                                    f"expected {list(required)} (found {list(df.columns)})")
    if df.empty:
        raise EfficiencyFormatError(f"{path.name}: no data rows")
    return df


def _numeric(df: pd.DataFrame, columns, path) -> None:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any() or (df[col] < 0).any():
            raise EfficiencyFormatError(f"{Path(path).name}: column {col!r} must hold non-negative numbers")


def read_efficiency(path) -> pd.DataFrame:
    """Read P2's efficiency.csv: index, build_seconds, bytes (+ size_mib computed from bytes)."""
    df = _read_csv(path, EFFICIENCY_COLUMNS, "efficiency")
    _numeric(df, ("build_seconds", "bytes"), path)
    if df["index"].duplicated().any():
        raise EfficiencyFormatError(f"{Path(path).name}: duplicate index names")
    df["size_mib"] = df["bytes"] / 1024 ** 2
    return df[["index", "build_seconds", "bytes", "size_mib"]].reset_index(drop=True)


def read_tuning(path) -> pd.DataFrame:
    """Read P2's tuning.csv and derive the index name each run was built on."""
    df = _read_csv(path, TUNING_COLUMNS, "tuning")
    _numeric(df, ("validation_mrr_at_10", "median_ms", "mean_ms"), path)
    # run = "<index>-<model>-<n>": drop the last two '-' parts (model names contain no '-')
    df["index"] = df["run"].str.rsplit("-", n=2).str[0]
    return df


def summarise(efficiency_path, tuning_path=None) -> EfficiencySummary:
    """Join P2's efficiency.csv (and optionally tuning.csv) into comparison tables."""
    index_cost = read_efficiency(efficiency_path)
    notes = [f"Not recorded in P2's logs, so not reported: {', '.join(NOT_RECORDED)}.",
             "Timings are from the machine that ran P2's experiment; compare them with each other only."]
    if tuning_path is None:
        return EfficiencySummary(index_cost, None, None, notes)
    tuning = read_tuning(tuning_path)
    runs = tuning.merge(index_cost, on="index", how="left")
    unmatched = sorted(set(runs.loc[runs["build_seconds"].isna(), "index"]))
    if unmatched:
        notes.append(f"tuning.csv runs whose index is not in efficiency.csv (cost left blank): {unmatched}")
    runs = runs[["index", "variant", "model", "parameters", "validation_mrr_at_10", "median_ms", "mean_ms",
                 "build_seconds", "bytes", "size_mib", "run"]]
    best_rows = runs.sort_values(["validation_mrr_at_10", "run"], ascending=[False, True]) \
                    .groupby(["variant", "model"], sort=False).head(1) \
                    .sort_values(["variant", "model"]).reset_index(drop=True)
    return EfficiencySummary(index_cost, runs.reset_index(drop=True), best_rows, notes)
