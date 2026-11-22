"""Paired significance testing for comparing two retrieval systems.

Why paired? Both systems answer the SAME queries, so for every query we can take
the difference in score (system minus baseline). Looking at these per-query
differences removes the "some queries are just harder" noise. Two ingredients:

  * Paired bootstrap confidence interval: resample the queries (with replacement)
    many times, recompute the mean difference each time, and take the 2.5 % and
    97.5 % percentiles. If the interval excludes 0 the improvement is unlikely
    to be an accident of which queries we happened to have.
  * A paired test for a p-value: Wilcoxon signed-rank (default; no assumption that
    differences are normal) or a paired t-test.

Significance is judged ONLY from these per-query tests. Two systems whose average
scores differ are not "significantly different" until the paired test says so.

Conventions
  * difference = score(system) - score(baseline); every metric is higher = better.
  * Wilcoxon: zero differences are removed BEFORE the test, so scipy chooses between its exact
    and approximate calculation using the number of NON-ZERO differences (the real sample size).
    Verified against a brute-force exact test: exact up to 13 non-zero differences (ties and
    zeros included), a normal approximation with tie correction above that (within about 0.05
    of exact for 14-17 tied differences, closer as the sample grows).
  * Direction: `direction` is the sign of the MEAN difference. The Wilcoxon test looks at ranks
    instead, so one huge gain can make the mean positive while many small losses make the
    rank test favour the baseline. When the two disagree the comparison is never called
    significant (`direction_agrees` is False and the note says so).
  * Differences smaller than ZERO_TOL (1e-12) count as zero, so tiny floating-point
    noise between two systems that rank identically is not mistaken for a difference.
  * Both score vectors must cover EXACTLY the same query IDs (else ValueError).
    Queries with a missing (NaN) score in either system are dropped and counted
    in `n_dropped`; nothing is silently filled in.
  * The bootstrap uses numpy's seeded generator, so the same inputs, `n_boot`
    and `seed` always give the same interval.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 42
DEFAULT_CONFIDENCE = 0.95
DEFAULT_ALPHA = 0.05
TESTS = ("wilcoxon", "ttest")
ZERO_TOL = 1e-12
MIN_N_BOOT = 100            # fewer resamples give a meaningless interval
_CHUNK_CELLS = 2_000_000    # bootstrap in blocks so memory stays small for large query sets


@dataclass
class PairedComparison:
    metric: str
    system: str
    baseline: str
    n_queries: int            # queries actually used (after dropping NaNs)
    n_dropped: int            # queries dropped because a score was missing
    mean_system: float
    mean_baseline: float
    mean_diff: float          # mean of (system - baseline)
    ci_low: float             # bootstrap interval for mean_diff
    ci_high: float
    ci_level: float
    p_value: float            # NaN when the test cannot be run (see note)
    p_holm: float             # filled in by compare_systems (Holm correction over its comparisons)
    test: str
    significant: bool         # p_value < alpha AND the means differ (NOT corrected for several comparisons)
    significant_holm: bool    # p_holm < alpha: use this when one baseline is compared with several systems
    direction_agrees: bool    # False if the Wilcoxon signed-rank evidence points opposite to the mean difference
    direction: str            # "system_better" | "baseline_better" | "tie"
    conclusion: str           # one readable sentence
    note: str                 # caveats: small sample, identical scores, ...

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- bootstrap
def bootstrap_ci(diffs, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                 confidence: float = DEFAULT_CONFIDENCE) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) of the per-query differences `diffs`.

    Percentile bootstrap: draw n queries with replacement, take the mean, repeat
    n_boot times, and read off the (1-confidence)/2 and 1-(1-confidence)/2 quantiles.
    With a single query there is nothing to resample, so the interval is NaN.
    """
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or len(d) == 0:
        raise ValueError("diffs must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(d)):
        raise ValueError("diffs must be finite numbers")
    if n_boot < MIN_N_BOOT:
        raise ValueError(f"n_boot must be at least {MIN_N_BOOT}, got {n_boot}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1 (exclusive), got {confidence}")
    n, mean = len(d), float(d.mean())
    if n == 1:
        return mean, float("nan"), float("nan")
    if np.ptp(d) == 0:                      # every query has the same difference: nothing varies
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    rows = max(1, _CHUNK_CELLS // n)
    means = np.empty(n_boot)
    filled = 0
    while filled < n_boot:
        block = min(rows, n_boot - filled)
        picks = rng.integers(0, n, size=(block, n))       # query indices drawn with replacement
        means[filled:filled + block] = d[picks].mean(axis=1)
        filled += block
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return mean, float(low), float(high)


# --------------------------------------------------------------------------- paired tests
def paired_test(diffs, test: str = "wilcoxon") -> tuple[float, str]:
    """Two-sided paired test that the mean/median difference is 0. Returns (p_value, note).

    p_value is NaN (with an explanation in the note) when the test is not defined.
    All differences zero -> p = 1.0 by convention (no evidence of any difference).
    """
    if test not in TESTS:
        raise ValueError(f"test must be one of {TESTS}, got {test!r}")
    d = np.asarray(diffs, dtype=float)
    d = np.where(np.abs(d) < ZERO_TOL, 0.0, d)
    nonzero = int(np.count_nonzero(d))
    if len(d) == 0:
        raise ValueError("no differences to test")
    if nonzero == 0:
        return 1.0, "all per-query differences are zero (identical scores)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # scipy warns about ties / small samples; we report that in `note`
        if test == "wilcoxon":
            # Drop the zeros ourselves so scipy sizes its exact/approximate choice by the non-zero count.
            p = float(stats.wilcoxon(d[d != 0], zero_method="wilcox", alternative="two-sided").pvalue)
            note = ""
            if nonzero < 6:
                note = (f"only {nonzero} non-zero difference(s): a two-sided Wilcoxon test cannot "
                        "reach p < 0.05 with fewer than 6")
            return p, note
        if len(d) < 2:
            return float("nan"), "paired t-test needs at least 2 queries"
        if np.ptp(d) == 0:
            return float("nan"), "paired t-test is undefined: every difference is identical (zero variance)"
        return float(stats.ttest_1samp(d, 0.0).pvalue), ""


def rank_direction(diffs) -> int:
    """+1 if the positive differences carry more signed-rank weight than the negative ones, -1 if less, 0 if equal.

    Ranks are of |difference| with ties averaged and zeros ignored, the same quantities the
    Wilcoxon signed-rank test uses, so this is the direction that test is actually reacting to.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.abs(d) >= ZERO_TOL]
    if len(d) == 0:
        return 0
    ranks = stats.rankdata(np.abs(d))
    plus, minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return int(np.sign(plus - minus))


def holm_adjust(p_values) -> list[float]:
    """Holm step-down correction for several comparisons made at once. NaN p-values are skipped."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    for position, i in enumerate(order):
        running = max(running, min(1.0, (len(valid) - position) * p[i]))   # keep adjusted p-values non-decreasing
        adjusted[i] = running
    return adjusted.tolist()


# --------------------------------------------------------------------------- comparisons
def _align(system: pd.Series, baseline: pd.Series) -> tuple[pd.Series, pd.Series]:
    for name, s in (("system", system), ("baseline", baseline)):
        if not isinstance(s, pd.Series):
            raise TypeError(f"{name} scores must be a pandas Series indexed by query ID")
        if s.index.has_duplicates:
            raise ValueError(f"{name} scores contain duplicate query IDs")
    only_system = sorted(set(system.index) - set(baseline.index))
    only_baseline = sorted(set(baseline.index) - set(system.index))
    if only_system or only_baseline:
        raise ValueError("systems must be compared on exactly the same queries; "
                         f"only in system: {only_system[:5]}, only in baseline: {only_baseline[:5]}")
    return system, baseline.reindex(system.index)


def compare_pair(system: pd.Series, baseline: pd.Series, *, metric: str = "", system_name: str = "system",
                 baseline_name: str = "baseline", n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                 confidence: float = DEFAULT_CONFIDENCE, alpha: float = DEFAULT_ALPHA,
                 test: str = "wilcoxon") -> PairedComparison:
    """Compare two systems' per-query scores (Series indexed by query ID)."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1 (exclusive), got {alpha}")
    system, baseline = _align(system, baseline)
    keep = system.notna() & baseline.notna()
    n_dropped = int((~keep).sum())
    a, b = system[keep].to_numpy(float), baseline[keep].to_numpy(float)
    if len(a) == 0:
        raise ValueError("no queries with a score in both systems")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("scores must be finite numbers (or NaN for missing)")
    diffs = a - b
    mean_diff, low, high = bootstrap_ci(diffs, n_boot, seed, confidence)
    p, test_note = paired_test(diffs, test)
    notes = [test_note] if test_note else []
    if n_dropped:
        notes.append(f"{n_dropped} query(ies) dropped because a score was missing")
    if len(a) == 1:
        notes.append("single query: confidence interval undefined")
    mean_sign = 0 if abs(mean_diff) < ZERO_TOL else (1 if mean_diff > 0 else -1)
    rank_sign = rank_direction(diffs) if test == "wilcoxon" else mean_sign      # a t-test follows the mean by construction
    agrees = rank_sign in (0, mean_sign)
    if not agrees:
        favoured = {1: system_name, -1: baseline_name}[rank_sign]
        by_mean = {1: system_name, -1: baseline_name, 0: "neither system"}[mean_sign]
        notes.append(f"the mean difference favours {by_mean} but the signed-rank sums favour {favoured} "
                     "(a few large differences dominate the mean); not called significant")
    if mean_sign == 0:
        direction, significant = "tie", False
    else:
        direction = "system_better" if mean_sign > 0 else "baseline_better"
        significant = bool(not np.isnan(p) and p < alpha and agrees)
    return PairedComparison(
        metric=metric, system=system_name, baseline=baseline_name, n_queries=len(a), n_dropped=n_dropped,
        mean_system=float(a.mean()), mean_baseline=float(b.mean()), mean_diff=mean_diff,
        ci_low=low, ci_high=high, ci_level=confidence, p_value=p, p_holm=float("nan"), test=test,
        significant=significant, significant_holm=significant, direction_agrees=agrees, direction=direction,   # one comparison: Holm = raw
        conclusion=_conclusion(system_name, baseline_name, direction, p, alpha, test, agrees),
        note="; ".join(notes))


def _conclusion(system, baseline, direction, p, alpha, test, agrees=True) -> str:
    if direction == "tie":
        return f"{system} and {baseline} have the same mean score"
    better = system if direction == "system_better" else baseline
    if not agrees:
        return (f"{better} has the higher mean score, but the {test} signed-rank evidence points the other way "
                f"(p={p:.3g}); not called significant")
    if np.isnan(p):
        return f"{better} has the higher mean score; significance could not be tested"
    verdict = "significant" if p < alpha else "NOT significant"
    return (f"{better} has the higher mean score; difference is {verdict} "
            f"({test} p={p:.3g}, alpha={alpha}, before multiple-comparison correction)")


def compare_systems(wide: pd.DataFrame, *, baseline: str | None = None, pairs=None, metric: str = "",
                    **options) -> pd.DataFrame:
    """Compare several systems in a `wide` table (rows = queries, columns = systems).

    This is exactly the table `EvalResult.wide(metric)` returns. By default every
    other system is compared with `baseline`. If `baseline` is omitted the FIRST COLUMN is used;
    note that EvalResult.wide() sorts its columns alphabetically, so name the baseline explicitly.
    Pass `pairs=[(system, baseline), ...]` for other comparisons. `options` are
    forwarded to compare_pair (n_boot, seed, confidence, alpha, test).

    The result has one row per comparison. `p_holm` is the p-value after Holm
    correction for doing several comparisons on the same metric, and
    `significant_holm` applies alpha to it. When one baseline is compared with
    several systems, prefer `significant_holm`: testing many systems makes a
    "lucky" small p-value likely by chance alone.
    """
    systems = list(wide.columns)
    if pairs is None:
        baseline = systems[0] if baseline is None else baseline
        if baseline not in systems:
            raise ValueError(f"baseline {baseline!r} is not one of the systems {systems}")
        pairs = [(s, baseline) for s in systems if s != baseline]
    for s, b in pairs:
        for name in (s, b):
            if name not in systems:
                raise ValueError(f"unknown system {name!r}; available: {systems}")
    rows = [compare_pair(wide[s], wide[b], metric=metric, system_name=s, baseline_name=b, **options).to_dict()
            for s, b in pairs]
    out = pd.DataFrame(rows, columns=list(PairedComparison.__dataclass_fields__))
    if len(out):
        out["p_holm"] = holm_adjust(out["p_value"])
        out["significant_holm"] = ((out["p_holm"] < options.get("alpha", DEFAULT_ALPHA))
                                   & (out["direction"] != "tie") & out["direction_agrees"])
    return out


def compare_result(result, metrics, **options) -> pd.DataFrame:
    """Run `compare_systems` for several metrics of an `EvalResult` and stack the rows.

    `metrics` are per-query column names such as "rr@10" or "ndcg@10".
    Holm correction is applied separately for each metric.
    """
    frames = [compare_systems(result.wide(m), metric=m, **options) for m in metrics]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(PairedComparison.__dataclass_fields__))
