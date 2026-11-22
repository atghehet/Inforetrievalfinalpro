"""Group queries (by popularity, year, citation function, ...) and compare systems inside the groups.

Everything works on the per-query scores of the P4 harness (`EvalResult.per_query`), so it uses exactly the same
numbers as the evaluation reports.

Group rules
  * Groups are defined on the RELEVANT paper of each query (its citation count, year, ...), except citation function,
    which describes the query sentence itself.
  * A query whose value is missing goes to an explicit "unknown" group; it is never dropped.
  * Every group is reported with its number of queries. A group with fewer than `min_group_size` queries is SMALL:
    its mean is shown, but no confidence interval, no system comparison, and it is left out of every trend or
    group-difference test.
  * Intervals are bootstrap intervals over the queries in the group. System differences use the paired tests of
    p4.eval.significance on exactly the queries of the group.
  * These are associations in observational data. A popular paper also has more citing sentences to index and tends to be
    older; nothing here separates those effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from p4.eval.significance import (DEFAULT_ALPHA, DEFAULT_CONFIDENCE, DEFAULT_N_BOOT, DEFAULT_SEED, bootstrap_ci,
                                  compare_pair, holm_adjust)

DEFAULT_MIN_GROUP_SIZE = 10
UNKNOWN = "unknown"

CITATION_BINS = [-1, 0, 2, 10, 50, np.inf]                       # P1's buckets (citeds/stats.py)
CITATION_LABELS = ["0", "1-2", "3-10", "11-50", ">50"]
CONTEXT_BINS = [-1, 0, 1, 5, 20, 100, np.inf]                    # indexable contexts available for the relevant paper
CONTEXT_LABELS = ["0", "1", "2-5", "6-20", "21-100", ">100"]
AGE_BINS = [-np.inf, -1, 1, 3, 5, np.inf]                        # citing year minus cited year
AGE_LABELS = ["<0", "0-1", "2-3", "4-5", ">5"]
FUNCTION_ORDER = ["background", "method", "result"]


def _bucket(values: pd.Series, bins, labels) -> tuple[pd.Series, list[str]]:
    cut = pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels).astype(object)
    groups = cut.where(cut.notna(), UNKNOWN).astype(str)
    order = list(labels) + ([UNKNOWN] if (groups == UNKNOWN).any() else [])
    return groups, order


def citation_groups(citation_count: pd.Series) -> tuple[pd.Series, list[str]]:
    """P1's popularity buckets 0 | 1-2 | 3-10 | 11-50 | >50 (papers.citation_count). Empty buckets stay in the order."""
    return _bucket(citation_count, CITATION_BINS, CITATION_LABELS)


def context_groups(n_index_contexts: pd.Series) -> tuple[pd.Series, list[str]]:
    return _bucket(n_index_contexts, CONTEXT_BINS, CONTEXT_LABELS)


def age_groups(citation_age: pd.Series) -> tuple[pd.Series, list[str]]:
    return _bucket(citation_age, AGE_BINS, AGE_LABELS)


def year_groups(year: pd.Series, bins: list[int] | None = None) -> tuple[pd.Series, list[str]]:
    """One group per year (default), or coarser groups from `bins`.

    bins=[2019, 2021] gives "<=2019", "2020-2021", ">=2022". Missing years form the "unknown" group.
    """
    values = pd.to_numeric(year, errors="coerce")
    known = values.dropna()
    if bins is None:
        order = [str(int(y)) for y in sorted(known.unique())]
        groups = values.map(lambda y: str(int(y)) if y == y else UNKNOWN)
    else:
        edges = sorted(int(b) for b in bins)
        if not edges or len(set(edges)) != len(edges):
            raise ValueError(f"year bins must be distinct whole numbers, got {bins!r}")
        labels = [f"<={edges[0]}"] + [f"{lo + 1}-{hi}" for lo, hi in zip(edges, edges[1:])] + [f">={edges[-1] + 1}"]

        def label(y):
            if y != y:
                return UNKNOWN
            for edge, name in zip(edges, labels):
                if y <= edge:
                    return name
            return labels[-1]
        groups, order = values.map(label), labels
    if (groups == UNKNOWN).any():
        order = order + [UNKNOWN]
    return groups.astype(str), order


def function_groups(function: pd.Series, confidence: pd.Series | None = None,
                    min_confidence: float | None = None) -> tuple[pd.Series, list[str]]:
    """background | method | result | unlabelled. With min_confidence, weaker predictions become 'unlabelled (low confidence)'."""
    groups = function.where(function.notna(), "unlabelled").astype(object)
    order = list(FUNCTION_ORDER)
    if min_confidence is not None and confidence is not None:
        low = function.notna() & (pd.to_numeric(confidence, errors="coerce") < min_confidence)
        groups = groups.where(~low, "unlabelled (low confidence)")
        if low.any():
            order.append("unlabelled (low confidence)")
    if (groups == "unlabelled").any():
        order.append("unlabelled")
    return groups.astype(str), order


# ---------------------------------------------------------------------------------------------------
def _attach(per_query: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    return per_query.assign(group=per_query["query_id"].map(groups))


def summarise_groups(per_query: pd.DataFrame, groups: pd.Series, order: list[str], metrics: list[str], *,
                     min_group_size: int = DEFAULT_MIN_GROUP_SIZE, n_boot: int = DEFAULT_N_BOOT,
                     seed: int = DEFAULT_SEED, confidence: float = DEFAULT_CONFIDENCE) -> pd.DataFrame:
    """Mean score per (system, group, metric) with sample size, bootstrap interval and a small-group flag.

    Long format: system, group, metric, n_queries, small_group, mean, ci_low, ci_high. Groups with no queries appear
    with n_queries 0 and a NaN mean so that their emptiness is visible. Small groups get no interval.
    """
    if min_group_size < 1:
        raise ValueError(f"min_group_size must be >= 1, got {min_group_size}")
    data = _attach(per_query, groups)
    rows = []
    for system, by_system in data.groupby("system", sort=False):
        for group in order:
            members = by_system[by_system["group"] == group]
            n = len(members)
            small = n < min_group_size
            for metric in metrics:
                values = members[metric].to_numpy(float)
                mean = float(values.mean()) if n else float("nan")
                low = high = float("nan")
                if not small and n >= 2:
                    _, low, high = bootstrap_ci(values, n_boot, seed, confidence)
                rows.append({"system": system, "group": group, "metric": metric, "n_queries": n,
                             "small_group": small, "mean": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows, columns=["system", "group", "metric", "n_queries", "small_group", "mean", "ci_low", "ci_high"])


def group_sizes(groups: pd.Series, order: list[str], min_group_size: int) -> pd.DataFrame:
    counts = groups.value_counts()
    return pd.DataFrame({"group": order, "n_queries": [int(counts.get(g, 0)) for g in order],
                         "small_group": [counts.get(g, 0) < min_group_size for g in order]})


def compare_within_groups(per_query: pd.DataFrame, groups: pd.Series, order: list[str], metrics: list[str], *,
                          baseline: str, min_group_size: int = DEFAULT_MIN_GROUP_SIZE, n_boot: int = DEFAULT_N_BOOT,
                          seed: int = DEFAULT_SEED, alpha: float = DEFAULT_ALPHA, test: str = "wilcoxon") -> pd.DataFrame:
    """Compare every system with `baseline` inside each non-small group (paired, same queries).

    Holm correction is applied per metric over all (system, group) comparisons of this call.
    """
    systems = list(dict.fromkeys(per_query["system"]))
    if baseline not in systems:
        raise ValueError(f"baseline {baseline!r} is not one of the systems {systems}")
    data = _attach(per_query, groups)
    rows = []
    for metric in metrics:
        for group in order:
            members = data[data["group"] == group]
            if members["query_id"].nunique() < min_group_size:
                continue
            wide = members.pivot(index="query_id", columns="system", values=metric)
            for system in systems:
                if system == baseline:
                    continue
                r = compare_pair(wide[system], wide[baseline], metric=metric, system_name=system, baseline_name=baseline,
                                 n_boot=n_boot, seed=seed, alpha=alpha, test=test)
                rows.append({"system": system, "baseline": baseline, "group": group, "metric": metric,
                             "n_queries": r.n_queries, "mean_system": r.mean_system, "mean_baseline": r.mean_baseline,
                             "mean_diff": r.mean_diff, "ci_low": r.ci_low, "ci_high": r.ci_high, "p_value": r.p_value,
                             "direction": r.direction, "direction_agrees": r.direction_agrees, "note": r.note})
    columns = ["system", "baseline", "group", "metric", "n_queries", "mean_system", "mean_baseline", "mean_diff", "ci_low",
               "ci_high", "p_value", "p_holm", "significant_holm", "direction", "direction_agrees", "note"]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns)
    out["p_holm"] = np.nan
    for metric, idx in out.groupby("metric").groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_value"])
    out["significant_holm"] = (out["p_holm"] < alpha) & (out["direction"] != "tie") & out["direction_agrees"]
    return out[columns]


# ---------------------------------------------------------------------------------------------------
def trend_tests(per_query: pd.DataFrame, x: pd.Series, groups: pd.Series, order: list[str], metrics: list[str], *,
                x_name: str, min_group_size: int = DEFAULT_MIN_GROUP_SIZE, n_boot: int = DEFAULT_N_BOOT,
                seed: int = DEFAULT_SEED, confidence: float = DEFAULT_CONFIDENCE, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Spearman rank correlation between a numeric group variable `x` and each per-query metric, for every system.

    Only queries that belong to NON-SMALL groups and have a known x are used; the number used and the number excluded are
    reported. The trend is "supported" only if at least two non-small groups remain. rho comes with a bootstrap interval
    (queries resampled) and a p-value; p_holm corrects across systems for each metric. A correlation is an association,
    not evidence that x causes better or worse retrieval.
    """
    sizes = group_sizes(groups, order, min_group_size).set_index("group")
    eligible_groups = [g for g in order if not sizes.loc[g, "small_group"] and g != UNKNOWN]
    xs = pd.to_numeric(x, errors="coerce")
    rows = []
    for system, by_system in per_query.groupby("system", sort=False):
        by_system = by_system.assign(group=by_system["query_id"].map(groups), x=by_system["query_id"].map(xs))
        usable = by_system[by_system["group"].isin(eligible_groups) & by_system["x"].notna()]
        excluded = len(by_system) - len(usable)
        for metric in metrics:
            row = {"system": system, "metric": metric, "x": x_name, "n_queries_used": len(usable),
                   "n_queries_excluded": excluded, "n_groups_used": usable["group"].nunique(),
                   "spearman_rho": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                   "supported": False, "note": ""}
            if usable["group"].nunique() < 2:
                row["note"] = "not testable: fewer than 2 groups reach the minimum group size"
            elif usable["x"].nunique() < 2 or usable[metric].nunique() < 2:
                row["note"] = "not testable: x or the metric is constant over the usable queries"
            else:
                rho, p = stats.spearmanr(usable["x"], usable[metric])
                row.update(spearman_rho=float(rho), p_value=float(p), supported=True)
                row["ci_low"], row["ci_high"] = _bootstrap_spearman(usable["x"].to_numpy(float),
                                                                   usable[metric].to_numpy(float), n_boot, seed, confidence)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    out["significant_holm"] = False
    for metric, idx in out.groupby("metric").groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_value"])
    out["significant_holm"] = (out["p_holm"] < alpha) & out["supported"]
    return out


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, confidence: float) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        if len(set(x[pick])) < 2 or len(set(y[pick])) < 2:
            continue                                           # a resample with no variation has no correlation
        values.append(stats.spearmanr(x[pick], y[pick])[0])
    if len(values) < max(10, n_boot // 2):
        return float("nan"), float("nan")
    tail = (1 - confidence) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return float(low), float(high)


def group_difference_tests(per_query: pd.DataFrame, groups: pd.Series, order: list[str], metrics: list[str], *,
                           min_group_size: int = DEFAULT_MIN_GROUP_SIZE, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Kruskal-Wallis test: do the scores of a system differ between (non-small) groups? For unordered groups such as
    citation function. Needs at least two non-small groups; p_holm corrects across systems per metric."""
    sizes = group_sizes(groups, order, min_group_size).set_index("group")
    eligible = [g for g in order if not sizes.loc[g, "small_group"] and g != UNKNOWN and not g.startswith("unlabelled")]
    rows = []
    for system, by_system in per_query.groupby("system", sort=False):
        by_system = by_system.assign(group=by_system["query_id"].map(groups))
        for metric in metrics:
            samples = [by_system.loc[by_system["group"] == g, metric].to_numpy(float) for g in eligible]
            row = {"system": system, "metric": metric, "n_groups_used": len(eligible),
                   "n_queries_used": int(sum(len(s) for s in samples)), "h_statistic": np.nan, "p_value": np.nan,
                   "supported": False, "note": ""}
            if len(eligible) < 2:
                row["note"] = "not testable: fewer than 2 groups reach the minimum group size"
            elif len(np.unique(np.concatenate(samples))) < 2:
                row["note"] = "not testable: the metric is constant over the usable queries"
            else:
                h, p = stats.kruskal(*samples)
                row.update(h_statistic=float(h), p_value=float(p), supported=True)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for metric, idx in out.groupby("metric").groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_value"])
    out["significant_holm"] = (out["p_holm"] < alpha) & out["supported"]
    return out
