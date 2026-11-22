"""Run the complete P4 experiment protocol, in order, from one configuration file.

    python -m p4.experiments.run_all --data-label synthetic --processed <P1 processed> --config runs.json --out <NEW folder>

REAL DATA IS REFUSED unless `--confirm-integrated-project` is given. That flag is the team's statement that P1, P2 and P3 have synced
their latest files, the integrated project was verified, and the leakage protocol was approved. Until then use synthetic or demo data.

Steps (each writes into its own sub-folder of --out; a failing step stops the run with a clear message)
  1. leakage gate      p4.graph.leakage_audit: splits disjoint by citing paper; every context file given to P2/P3 free of query
                       contexts and query sentences. Any failure stops the run before a single result is computed.
  2. evaluation graph  p4.graph.eval_graph: leakage-safe edges, citation_count_eval, PageRank at damping 0.5 / 0.85 / 0.95
  3. evaluation        p4.eval.cli on the TEST split: all systems, metrics, paired tests, efficiency (if the P2 logs are given)
  4. fusion            per system: tune the authority prior on VALIDATION, apply the frozen choice to TEST, then evaluate base and fused together
  5. analyses          popularity, coldstart, year, errors (and function if labels exist), using the leakage-safe citation counts

Configuration (JSON)
  {"systems": {"bm25": {"val": "bm25.validation.run", "test": "bm25.test.run"}, "dense": {...}},
   "baseline": "bm25",
   "qrels": {"val": "qrels.val.txt", "test": "qrels.test.txt"},                  # optional: default <processed>/qrels/qrels.<split>.txt
   "context_files": ["p2_input/contexts.jsonl"],                                  # JSONL files P2/P3 index (leakage gate)
   "efficiency": {"efficiency_csv": "...", "tuning_csv": "..."},                  # optional (P2 logs)
   "functions": "context_functions.parquet", "index_contexts": "index_contexts.parquet",   # optional
   "fusion_systems": ["bm25", "dense"],                           # optional: fuse only these (default all; fusing every system is slow)
   "analysis_systems": ["bm25", "dense", "bm25+authority"]}       # optional: run the analyses on these systems only (default all)

Relative paths in the configuration are resolved against the folder holding the configuration file.
Every output carries the data label; EXPERIMENT_MANIFEST.json records the settings, inputs and which steps ran.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from p4.analysis import cli as analysis_cli
from p4.eval import cli as eval_cli
from p4.graph import eval_graph, fusion, leakage_audit

DATA_LABELS = ("real", "synthetic", "demo")
DEFAULT_ANALYSES = ("popularity", "coldstart", "year", "errors")


class ExperimentError(RuntimeError):
    """A step failed or the configuration is unusable."""


def load_config(path: Path, processed: Path, check_files: bool = True) -> dict:
    """Read the configuration. With check_files=False missing input files are not an error here (the verifier reports them itself)."""
    path = Path(path)
    if not path.is_file():
        raise ExperimentError(f"configuration file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ExperimentError(f"{path.name} is not valid JSON: {err}") from None
    base = path.resolve().parent

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else base / p

    systems = raw.get("systems")
    if not isinstance(systems, dict) or not systems:
        raise ExperimentError("configuration needs a non-empty 'systems' mapping")
    cfg = {"systems": {}, "context_files": [resolve(p) for p in raw.get("context_files", [])]}
    for name, files in systems.items():
        if not isinstance(files, dict) or not {"val", "test"} <= set(files):
            raise ExperimentError(f"system {name!r} needs 'val' and 'test' run files")
        cfg["systems"][name] = {"val": resolve(files["val"]), "test": resolve(files["test"])}
    cfg["baseline"] = raw.get("baseline") or next(iter(cfg["systems"]))
    if cfg["baseline"] not in cfg["systems"]:
        raise ExperimentError(f"baseline {cfg['baseline']!r} is not one of the systems {list(cfg['systems'])}")
    qrels = raw.get("qrels") or {}
    cfg["qrels"] = {s: resolve(qrels[s]) if s in qrels else processed / "qrels" / f"qrels.{s}.txt" for s in ("val", "test")}
    eff = raw.get("efficiency") or {}
    cfg["efficiency"] = {k: resolve(v) for k, v in eff.items() if k in ("efficiency_csv", "tuning_csv")}
    for key in ("functions", "index_contexts"):
        cfg[key] = resolve(raw[key]) if raw.get(key) else None
    for key in ("fusion_systems", "analysis_systems"):
        cfg[key] = list(raw[key]) if raw.get(key) else None
    unknown = [n for n in cfg["fusion_systems"] or [] if n not in cfg["systems"]]
    if unknown:
        raise ExperimentError(f"fusion_systems {unknown} are not in systems {list(cfg['systems'])}")
    missing = [str(p) for files in cfg["systems"].values() for p in files.values() if not p.is_file()]
    missing += [str(p) for p in cfg["qrels"].values() if not p.is_file()]
    missing += [str(p) for p in cfg["context_files"] if not p.is_file()]
    if missing and check_files:
        raise ExperimentError("input file(s) not found: " + "; ".join(missing))
    return cfg


def _step(name: str, code: int, steps: list) -> None:
    steps.append({"step": name, "exit_code": code})
    if code != 0:
        raise ExperimentError(f"step '{name}' failed (exit code {code}); see the message above. Nothing after it was run")


def run(args) -> dict:
    if args.data_label == "real" and not args.confirm_integrated_project:
        raise ExperimentError("REAL data refused: pass --confirm-integrated-project only after P1, P2 and P3 have synced their final "
                              "files, the integrated project is verified and the leakage protocol is approved")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ExperimentError(f"{out} is not empty; use a new folder")
    processed = Path(args.processed)
    if not processed.is_dir():
        raise ExperimentError(f"processed folder not found: {processed}")
    cfg = load_config(args.config, processed)
    label = args.data_label
    stats = ["--n-boot", str(args.n_boot), "--min-group-size", str(args.min_group_size)]
    steps: list[dict] = []
    started = dt.datetime.now(dt.timezone.utc)

    # 1. leakage gate -------------------------------------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    report = leakage_audit.audit(processed, context_files=cfg["context_files"])
    (out / "leakage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    problems = []
    if not report["split_audit"]["disjoint_by_citing_paper"]:
        problems.append("queries of one citing paper are in more than one split")
    problems += [f"context file {f['file']} leaks (not in index / query context / query sentence)" for f in report.get("context_files", []) if not f["ok"]]
    steps.append({"step": "leakage gate", "passed": not problems, "problems": problems})
    if problems:
        raise ExperimentError("LEAKAGE GATE FAILED, no result was computed: " + "; ".join(problems))

    # 2. evaluation graph ----------------------------------------------------------------------------------------
    graph_dir = out / "eval_graph"
    manifest = eval_graph.build(processed, graph_dir, data_label=label, strict_index=args.strict_index)
    steps.append({"step": "evaluation graph", "edges_kept": manifest["edge_removal"]["edges_kept"],
                  "edges_removed": manifest["edge_removal"]["edges_removed"]})

    base_runs = {name: files["test"] for name, files in cfg["systems"].items()}
    eff = ["--efficiency", str(cfg["efficiency"]["efficiency_csv"])] if "efficiency_csv" in cfg["efficiency"] else []
    if eff and "tuning_csv" in cfg["efficiency"]:
        eff += ["--tuning", str(cfg["efficiency"]["tuning_csv"])]

    # 3. evaluation of the base systems on TEST ------------------------------------------------------------------
    run_args = [a for name, path in base_runs.items() for a in ("--run", f"{name}={path}")]
    code = eval_cli.main(["--qrels", str(cfg["qrels"]["test"]), *run_args, "--out", str(out / "eval_test"), "--baseline", cfg["baseline"],
                          "--n-boot", str(args.n_boot), "--note", f"data label: {label}", "--quiet", *eff])
    _step("evaluation (test)", code, steps)

    # 4. fusion: tune on VALIDATION, apply to TEST ---------------------------------------------------------------
    fused_runs = {}
    fusion_extra = ["--lambdas", args.fusion_lambdas] if args.fusion_lambdas else []
    for name, files in cfg["systems"].items():
        if cfg["fusion_systems"] is not None and name not in cfg["fusion_systems"]:
            continue
        tune_dir = out / "fusion" / name
        code = fusion.main(["tune", "--data-label", label, "--run", str(files["val"]), "--qrels", str(cfg["qrels"]["val"]),
                            "--prior-dir", str(graph_dir), "--out", str(tune_dir), "--mode", args.fusion_mode, *fusion_extra])
        _step(f"fusion tune ({name})", code, steps)
        fused = out / "fusion" / name / "fused_test.run"
        code = fusion.main(["apply", "--data-label", label, "--run", str(files["test"]), "--prior-dir", str(graph_dir),
                            "--best", str(tune_dir / "fusion_best.json"), "--out", str(fused), "--tag", f"{name}+authority"])
        _step(f"fusion apply ({name})", code, steps)
        fused_runs[f"{name}+authority"] = fused
    all_runs = {**base_runs, **fused_runs}
    run_args = [a for name, path in all_runs.items() for a in ("--run", f"{name}={path}")]
    eval_run_args = run_args
    if cfg["analysis_systems"] is not None:
        unknown = [n for n in cfg["analysis_systems"] if n not in all_runs]
        if unknown:
            raise ExperimentError(f"analysis_systems {unknown} are not systems or fused systems {list(all_runs)}")
        run_args = [a for name, path in all_runs.items() if name in cfg["analysis_systems"] for a in ("--run", f"{name}={path}")]
    code = eval_cli.main(["--qrels", str(cfg["qrels"]["test"]), *eval_run_args, "--out", str(out / "eval_fusion_test"), "--baseline", cfg["baseline"],
                          "--n-boot", str(args.n_boot), "--note", f"data label: {label}; fused systems use the leakage-safe prior", "--quiet"])
    _step("evaluation of base and fused systems (test)", code, steps)

    # 5. analyses (leakage-safe citation counts) -------------------------------------------------------------------
    for analysis in args.analyses:
        extra = []
        if analysis == "function":
            if not cfg["functions"]:
                steps.append({"step": "analysis function", "skipped": "no 'functions' file in the configuration (labels are never invented)"})
                continue
            extra += ["--functions", str(cfg["functions"])]
        if analysis == "coldstart" and cfg["index_contexts"]:
            extra += ["--index-contexts", str(cfg["index_contexts"])]
        code = analysis_cli.main([analysis, "--data-label", label, "--qrels", str(cfg["qrels"]["test"]), *run_args,
                                  "--queries", str(processed / "queries.parquet"), "--papers", str(processed / "papers.parquet"),
                                  "--citation-counts", str(graph_dir / "priors.parquet"), "--baseline", cfg["baseline"],
                                  "--out", str(out / "analysis"), "--quiet", *stats, *extra])
        _step(f"analysis {analysis}", code, steps)

    result = {"data_label": label, "started_utc": started.isoformat(timespec="seconds"),
              "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "processed": str(processed),
              "protocol": eval_graph.PROTOCOL, "systems": list(cfg["systems"]), "baseline": cfg["baseline"],
              "settings": {"n_boot": args.n_boot, "min_group_size": args.min_group_size, "strict_index": args.strict_index,
                           "analyses": list(args.analyses), "fusion_mode": args.fusion_mode, "fusion_systems": cfg["fusion_systems"],
                           "analysis_systems": cfg["analysis_systems"]},
              "confirmed_integrated_project": bool(args.confirm_integrated_project), "steps": steps}
    (out / "EXPERIMENT_MANIFEST.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-label", required=True, choices=DATA_LABELS)
    p.add_argument("--processed", required=True, type=Path, help="P1's data/processed folder (read-only)")
    p.add_argument("--config", required=True, type=Path, help="JSON configuration (see the module docstring)")
    p.add_argument("--out", required=True, type=Path, help="NEW output folder")
    p.add_argument("--confirm-integrated-project", action="store_true",
                   help="REQUIRED for real data: the team confirms the integrated project is verified")
    p.add_argument("--strict-index", action="store_true", help="also write the strict index for the sensitivity analysis")
    p.add_argument("--analyses", nargs="+", default=list(DEFAULT_ANALYSES), choices=sorted(analysis_cli.ANALYSES))
    p.add_argument("--fusion-mode", choices=fusion.MODES, default="convex",
                   help="convex (default), additive, or raw = score + lambda*log(prior), the docs/P4_HANDOFF.md formula")
    p.add_argument("--fusion-lambdas", help="comma-separated weights for the tuning grid (default: the mode's own grid)")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--min-group-size", type=int, default=10)
    args = p.parse_args(argv)
    try:
        result = run(args)
    except (ExperimentError, eval_graph.EvalGraphError, FileNotFoundError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"[{result['data_label']}] completed {len(result['steps'])} steps; manifest: {args.out / 'EXPERIMENT_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
