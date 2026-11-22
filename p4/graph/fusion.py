"""Authority-prior fusion: combine a retrieval score with a citation-graph score (PageRank or citation count).

    python -m p4.graph.fusion tune  --data-label L --run RUN --qrels QRELS_VAL --prior-dir EVAL_GRAPH_DIR --out OUTDIR
    python -m p4.graph.fusion apply --data-label L --run RUN --prior-dir EVAL_GRAPH_DIR --best OUTDIR/fusion_best.json --out FUSED.run

This is research question 3 of the plan: does a PageRank authority prior improve ranking, or does it only reinforce popularity bias?

Method (re-ranking of the documents a system already retrieved)
  1. Per query, normalise the retrieval scores of the retrieved list ("minmax", "zscore" or "rank").
  2. Corpus-wide, transform the prior ("identity" or "log": PageRank spans orders of magnitude) and rescale it to [0, 1].
  3. Combine, with weight lambda:
       convex     s = (1 - lambda) * score + lambda * prior          (lambda in [0, 1])
       additive   s = score + lambda * prior                         (lambda >= 0)
       raw        s = score + lambda * log(prior)                    (lambda >= 0)   <- the formula in docs/P4_HANDOFF.md
     `raw` uses the system's own scores as they are (no per-query normalisation) and log(prior) with no rescaling; a prior value of 0 (or
     below RAW_LOG_EPSILON) is replaced by RAW_LOG_EPSILON = 1e-12 so the logarithm is finite. It is selected with `--mode raw`, which
     implies `--score-norm none --prior-transform log_raw`; any other combination with `raw` is refused. Because the raw scores are not
     on a common scale, the raw lambda grid is wider than the default one (DEFAULT_RAW_LAMBDAS) and lambda must be tuned on validation.
     lambda = 0 reproduces the original ranking. Ties are broken by the original rank, then by document id, so results are deterministic.
  4. `tune` evaluates a grid of (prior column, lambda) on the VALIDATION queries only and picks the best by a validation metric
     (ties go to the smaller lambda, so "no prior" wins unless the prior strictly helps). `apply` then re-ranks any run with the frozen choice;
     the test split is scored once, afterwards, by the normal evaluation tool.

Safety guards
  * The prior folder must come from p4.graph.eval_graph. Its manifest must show that the split being evaluated was HELD OUT when the
    graph was built; otherwise the prior contains the queries' own citations and the tool refuses (`--allow-leaky-prior` exists only to
    demonstrate the leak on synthetic data and stamps the output).
  * The data label given on the command line must equal the label in the prior's manifest.
  * Documents of a run that have no prior value get the lowest prior (0; in `raw` mode log(RAW_LOG_EPSILON)) and are counted in the report.

Limits: only documents already in a run can be moved; a document the retrieval system did not return cannot be promoted by its prior.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from p4.eval.harness import evaluate_runs
from p4.eval.trec_io import TrecFormatError, read_qrels, read_run

from .eval_graph import DATA_LABELS, EvalGraphError, load_priors

SCORE_NORMS = ("minmax", "zscore", "rank", "none")           # "none" (raw scores) is only valid with mode "raw"
PRIOR_TRANSFORMS = ("identity", "log", "log_raw")            # "log_raw" (log, no rescaling) is only valid with mode "raw"
MODES = ("convex", "additive", "raw")
DEFAULT_LAMBDAS = tuple(round(x, 2) for x in np.linspace(0, 1, 11))
DEFAULT_RAW_LAMBDAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)   # log(PageRank) is about -5 to -12, so small weights matter
RAW_LOG_EPSILON = 1e-12                                       # stands in for a prior value of 0 in `raw` mode
METRIC = "MRR@10"


class FusionError(ValueError):
    """Bad fusion settings, an unsafe prior, or inconsistent inputs."""


def resolve_settings(mode: str, score_norm: str | None, transform: str | None) -> tuple[str, str]:
    """Fill in the defaults for a mode and refuse inconsistent combinations. Returns (score_norm, transform).

    convex / additive default to minmax scores and a log prior rescaled to [0, 1]; `raw` implies raw scores and log(prior) unscaled.
    """
    if mode not in MODES:
        raise FusionError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "raw":
        score_norm, transform = score_norm or "none", transform or "log_raw"
        if (score_norm, transform) != ("none", "log_raw"):
            raise FusionError(f"mode 'raw' means score + lambda*log(prior) with raw scores: it needs score_norm 'none' and prior transform "
                              f"'log_raw', got {score_norm!r} and {transform!r}")
        return score_norm, transform
    score_norm, transform = score_norm or "minmax", transform or "log"
    if score_norm == "none" or transform == "log_raw":
        raise FusionError(f"score_norm 'none' and prior transform 'log_raw' are only valid with mode 'raw' (got mode {mode!r})")
    return score_norm, transform


# ---------------------------------------------------------------------------------------------- normalisation
def normalise_scores(scores: list[float], method: str) -> list[float]:
    """Normalise one query's retrieval scores (best-first list). A list with no spread maps to all zeros."""
    if method not in SCORE_NORMS:
        raise FusionError(f"score normalisation must be one of {SCORE_NORMS}, got {method!r}")
    values = np.asarray(scores, dtype=float)
    n = len(values)
    if n == 0:
        return []
    if method == "none":
        return values.tolist()
    if method == "minmax":
        span = values.max() - values.min()
        return ((values - values.min()) / span).tolist() if span > 0 else [0.0] * n
    if method == "zscore":
        std = values.std()
        return ((values - values.mean()) / std).tolist() if std > 0 else [0.0] * n
    order = np.argsort(-values, kind="stable")                 # rank: best = 1.0, worst = 0.0
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return (1.0 - ranks / (n - 1)).tolist() if n > 1 else [1.0]


def prepare_prior(prior: pd.Series, transform: str) -> pd.Series:
    """Transform a prior (index = paper_id) and rescale it to [0, 1] over the whole corpus.

    transform "log_raw" is the exception: it returns log(max(value, RAW_LOG_EPSILON)) with NO rescaling (for mode "raw").
    """
    if transform not in PRIOR_TRANSFORMS:
        raise FusionError(f"prior transform must be one of {PRIOR_TRANSFORMS}, got {transform!r}")
    values = pd.to_numeric(prior, errors="coerce").astype(float)
    if values.isna().any() or (values < 0).any() or not np.isfinite(values).all():
        raise FusionError("prior values must be finite and non-negative")
    if transform == "log_raw":
        return np.log(values.clip(lower=RAW_LOG_EPSILON))
    if transform == "log":
        positive = values[values > 0]
        if len(positive) == 0:
            values = values * 0.0                                    # nothing to separate
        else:
            eps = positive.min() / 2 if (values == 0).any() else 0.0   # zeros (e.g. citation counts) sit just below the smallest value
            values = np.log(values + eps)
    span = values.max() - values.min() if len(values) else 0.0
    return (values - values.min()) / span if span > 0 else values * 0.0


# ---------------------------------------------------------------------------------------------- fusion
def fuse_run(run: dict, prior01: pd.Series, lam: float, *, mode: str = "convex", score_norm: str = "minmax") -> tuple[dict, dict]:
    """Re-rank every query of `run`. Returns (fused run, stats).

    `prior01` is a prepared prior in [0, 1] (mode "raw": a prior from prepare_prior(..., "log_raw"), i.e. log values, unscaled).
    """
    if mode not in MODES:
        raise FusionError(f"mode must be one of {MODES}, got {mode!r}")
    if (mode == "raw") != (score_norm == "none"):
        raise FusionError("score_norm 'none' goes with mode 'raw' and only with it")
    if lam < 0 or (mode == "convex" and lam > 1) or not math.isfinite(lam):
        raise FusionError(f"lambda must be in [0, 1] for 'convex' and >= 0 for 'additive' and 'raw', got {lam!r}")
    floor = math.log(RAW_LOG_EPSILON) if mode == "raw" else 0.0     # the prior given to documents that have none
    lookup = prior01.to_dict()
    fused, n_docs, n_missing = {}, 0, 0
    for qid, ranked in run.items():
        docs = [d for d, _ in ranked]
        norm = normalise_scores([s for _, s in ranked], score_norm)
        rows = []
        for original_rank, (doc, base) in enumerate(zip(docs, norm), start=1):
            p = lookup.get(doc)
            n_docs += 1
            if p is None:
                n_missing += 1
                p = floor
            score = (1 - lam) * base + lam * p if mode == "convex" else base + lam * p
            rows.append((score, original_rank, doc))
        rows.sort(key=lambda r: (-r[0], r[1], r[2]))              # best fused score; ties: original rank, then id
        fused[qid] = [(doc, score) for score, _, doc in rows]
    return fused, {"n_docs": n_docs, "n_docs_without_prior": n_missing}


def write_run(run: dict, path, tag: str) -> None:
    path = Path(path)
    if path.exists():
        raise FusionError(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for qid in sorted(run):
            for rank, (doc, score) in enumerate(run[qid], start=1):
                fh.write(f"{qid} Q0 {doc} {rank} {score:.10g} {tag}\n")


# ---------------------------------------------------------------------------------------------- guards
def check_prior_manifest(manifest: dict, *, split: str, data_label: str, allow_leaky: bool) -> list[str]:
    """Return warnings; raise FusionError if the prior is unsafe for `split` or labelled differently."""
    warnings = []
    if manifest.get("data_label") != data_label:
        raise FusionError(f"data label mismatch: you declared {data_label!r} but the prior folder was built as "
                          f"{manifest.get('data_label')!r}; refusing to mix them")
    held = manifest.get("hold_out_splits") or []
    if split not in held:
        message = (f"the prior was built with hold-out splits {held}, which do not include {split!r}: its graph still contains "
                   f"the citations behind the {split} queries (leakage)")
        if not allow_leaky:
            raise FusionError(message + ". Rebuild it with p4.graph.eval_graph (use --hold-out including this split)")
        warnings.append("LEAKY PRIOR ALLOWED: " + message)
    return warnings


# ---------------------------------------------------------------------------------------------- tuning
def sweep(run: dict, qrels: dict, priors: dict[str, pd.Series], lambdas, *, mode: str = "convex", score_norm: str | None = None,
          transform: str | None = None, min_coverage: float = 0.5) -> pd.DataFrame:
    """Evaluate every (prior, lambda) on the given queries. lambda = 0 is always included (the no-prior baseline)."""
    score_norm, transform = resolve_settings(mode, score_norm, transform)
    lams = sorted({0.0, *[float(x) for x in lambdas]})
    rows = []
    for name, prior in priors.items():
        prior01 = prepare_prior(prior, transform)
        for lam in lams:
            fused, stats = fuse_run(run, prior01, lam, mode=mode, score_norm=score_norm)
            summary = evaluate_runs({"fused": fused}, qrels, (1, 5, 10, 20, 50, 100), min_coverage=min_coverage).summary.iloc[0]
            rows.append({"prior": name, "lambda": lam, "mode": mode, "score_norm": score_norm, "prior_transform": transform,
                         "n_queries": int(summary["n_queries"]), "MRR@10": summary["MRR@10"], "Recall@10": summary["Recall@10"],
                         "nDCG@10": summary["nDCG@10"], "MAP": summary["MAP"],
                         "n_docs_without_prior": stats["n_docs_without_prior"]})
    return pd.DataFrame(rows)


def best_setting(table: pd.DataFrame, metric: str = METRIC) -> dict:
    """Highest validation metric; ties go to the smaller lambda, then to the prior listed first."""
    order = {name: i for i, name in enumerate(dict.fromkeys(table["prior"]))}
    ranked = table.assign(_p=table["prior"].map(order)).sort_values([metric, "lambda", "_p"], ascending=[False, True, True])
    best = ranked.iloc[0]
    baseline = table[table["lambda"] == 0.0].iloc[0]
    return {"prior": best["prior"], "lambda": float(best["lambda"]), "metric": metric, "validation_value": float(best[metric]),
            "baseline_validation_value": float(baseline[metric]),
            "improves_on_validation": bool(best[metric] > baseline[metric] + 1e-12),
            "mode": best["mode"], "score_norm": best["score_norm"], "prior_transform": best["prior_transform"]}


def tune_fusion(run: dict, qrels_val: dict, prior_folder, *, data_label: str, prior_columns=None, lambdas=None,
                mode: str = "convex", score_norm: str | None = None, transform: str | None = None, metric: str = METRIC,
                allow_leaky_prior: bool = False) -> tuple[pd.DataFrame, dict]:
    """Choose (prior, lambda) on VALIDATION queries. Takes no test data by construction.

    `lambdas` defaults to DEFAULT_LAMBDAS, or DEFAULT_RAW_LAMBDAS in mode "raw".
    """
    score_norm, transform = resolve_settings(mode, score_norm, transform)
    if lambdas is None:
        lambdas = DEFAULT_RAW_LAMBDAS if mode == "raw" else DEFAULT_LAMBDAS
    priors_df, manifest = load_priors(prior_folder)
    warnings = check_prior_manifest(manifest, split="val", data_label=data_label, allow_leaky=allow_leaky_prior)
    columns = list(prior_columns or manifest["prior_columns"])
    missing = [c for c in columns if c not in priors_df.columns]
    if missing:
        raise FusionError(f"prior column(s) {missing} not in the prior folder; available: {manifest['prior_columns']}")
    priors = {c: priors_df.set_index("paper_id")[c] for c in columns}
    table = sweep(run, qrels_val, priors, lambdas, mode=mode, score_norm=score_norm, transform=transform)
    best = best_setting(table, metric)
    best.update({"data_label": data_label, "prior_protocol": manifest.get("protocol"), "prior_hold_out_splits": manifest.get("hold_out_splits"),
                 "prior_folder": str(prior_folder), "n_validation_queries": int(table["n_queries"].iloc[0]),
                 "warnings": warnings, "tuned_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    table.insert(0, "data_label", data_label)
    return table, best


def apply_fusion(run: dict, prior_folder, best: dict, *, data_label: str, split: str = "test",
                 allow_leaky_prior: bool = False) -> tuple[dict, dict]:
    """Re-rank a run with a frozen (prior, lambda) choice from `tune_fusion`."""
    if best.get("data_label") != data_label:
        raise FusionError(f"data label mismatch: tuning was {best.get('data_label')!r}, you declared {data_label!r}")
    priors_df, manifest = load_priors(prior_folder)
    warnings = check_prior_manifest(manifest, split=split, data_label=data_label, allow_leaky=allow_leaky_prior)
    if best["prior"] not in priors_df.columns:
        raise FusionError(f"prior column {best['prior']!r} not in the prior folder")
    prior01 = prepare_prior(priors_df.set_index("paper_id")[best["prior"]], best["prior_transform"])
    fused, stats = fuse_run(run, prior01, best["lambda"], mode=best["mode"], score_norm=best["score_norm"])
    stats["warnings"] = warnings
    return fused, stats


# ---------------------------------------------------------------------------------------------- CLI
def _common(sp):
    sp.add_argument("--data-label", required=True, choices=DATA_LABELS)
    sp.add_argument("--run", required=True, type=Path, help="TREC run to re-rank")
    sp.add_argument("--prior-dir", required=True, type=Path, help="folder written by p4.graph.eval_graph")
    sp.add_argument("--allow-leaky-prior", action="store_true", help="only to DEMONSTRATE the leak on synthetic data")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="p4.graph.fusion", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tune", help="choose prior and lambda on the validation queries")
    _common(t)
    t.add_argument("--qrels", required=True, type=Path, help="VALIDATION qrels")
    t.add_argument("--out", required=True, type=Path, help="new folder for fusion_tuning.csv and fusion_best.json")
    t.add_argument("--prior", action="append", help="prior column to try (repeatable; default: all in the manifest)")
    t.add_argument("--lambdas", help="comma-separated weights (default: 0,0.1,...,1; for --mode raw: 0,0.001,...,3)")
    t.add_argument("--mode", choices=MODES, default="convex", help="raw = score + lambda*log(prior), the docs/P4_HANDOFF.md formula")
    t.add_argument("--score-norm", choices=SCORE_NORMS, help="default minmax (mode raw: none)")
    t.add_argument("--prior-transform", choices=PRIOR_TRANSFORMS, help="default log (mode raw: log_raw)")
    a = sub.add_parser("apply", help="re-rank a run with the frozen choice")
    _common(a)
    a.add_argument("--best", required=True, type=Path, help="fusion_best.json from `tune`")
    a.add_argument("--split", choices=("val", "test"), default="test", help="split the run answers (default test)")
    a.add_argument("--out", required=True, type=Path, help="fused run file to write (must not exist)")
    a.add_argument("--tag", help="run name written in the sixth column")
    return p.parse_args(argv)


def run_cli(args) -> int:
    run = read_run(args.run)
    if args.command == "tune":
        if args.out.exists() and any(args.out.iterdir()):
            raise FusionError(f"{args.out} is not empty; use a new folder")
        qrels = read_qrels(args.qrels)
        table, best = tune_fusion(run, qrels, args.prior_dir, data_label=args.data_label, prior_columns=args.prior,
                                  lambdas=[float(x) for x in args.lambdas.split(",") if x.strip()] if args.lambdas else None, mode=args.mode,
                                  score_norm=args.score_norm, transform=args.prior_transform, allow_leaky_prior=args.allow_leaky_prior)
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out / "fusion_tuning.csv", index=False)
        (args.out / "fusion_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        verdict = "improves on" if best["improves_on_validation"] else "does NOT improve on"
        print(f"[{args.data_label}] best on validation: {best['prior']} lambda={best['lambda']} {best['metric']}={best['validation_value']:.4f} "
              f"({verdict} the no-prior baseline {best['baseline_validation_value']:.4f}; {best['n_validation_queries']} queries)")
        for w in best["warnings"]:
            print("warning:", w, file=sys.stderr)
        return 0
    best = json.loads(args.best.read_text(encoding="utf-8"))
    fused, stats = apply_fusion(run, args.prior_dir, best, data_label=args.data_label, split=args.split,
                                allow_leaky_prior=args.allow_leaky_prior)
    tag = args.tag or f"fused_{best['prior']}_{best['mode']}_l{best['lambda']}"
    write_run(fused, args.out, tag)
    print(f"[{args.data_label}] wrote {args.out} ({len(fused)} queries, {stats['n_docs_without_prior']} of {stats['n_docs']} documents had no prior)")
    for w in stats["warnings"]:
        print("warning:", w, file=sys.stderr)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run_cli(args)
    except (FusionError, EvalGraphError, TrecFormatError, FileNotFoundError, ValueError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
