"""Discover the teammates' runs from runs/MANIFEST.csv, pair validation with test, and write the `runs.json` that run_all / verify_integration read.

    python -m p4.experiments.discover_runs --manifest runs/MANIFEST.csv --out runs.json --report runs_discovery.json [--processed data/processed]
    python -m p4.experiments.discover_runs --check-config runs.json [--processed data/processed]       # check a hand-written runs.json

Nothing is guessed from file names alone. A system is accepted only when its validation and test run are tied together by METADATA that the
teammates' own scripts write, and the file CONTENTS agree with that metadata:

  P2 lexical   runs/p2/best.json maps a key (e.g. "context_bm25") to the configuration chosen on validation, including its `tag`.
               The test run is p2/test/<key>.run, the validation run p2/validation/<tag>.run, and BOTH files must carry that tag in column 6.
  P2 BM25      p2/fusion/bm25.{validation,test}.run (the BM25 side of P3's hybrid); both must carry the tag in p2/fusion/selection.txt.
               When that tag is also a best.json key it is the same system, so it is validated but not listed twice.
  P3 dense     p3/dense/<name>.{val,test}.run, both tagged `dense_<name>`; the set of validation runs must match selection.json.
  P3 hybrid    p3/hybrid/hybrid_rrf.{val,test}.run, both tagged `hybrid_rrf_k<k>` with k = selection.json's rrf_k; its dense system must be a
               valid dense pair and its BM25 side must be the valid p2 fusion pair.

Checks on every pair (each failure names the system and is an ERROR that excludes the system):
  MISSING_FILE   listed in the manifest (or required by metadata) but absent       UNPAIRED        only one of validation/test exists
  DUPLICATE      the manifest lists a run twice                                    TAG_MISMATCH    column 6 differs from the metadata
  SPLIT_MISMATCH the manifest's split column disagrees with the file's role        BAD_FORMAT      a row that does not have 6 fields
  SPLIT_OVERLAP  validation and test runs answer some of the same queries          WRONG_SPLIT     (with --processed) queries belong to the other split
  LOW_COVERAGE   (with --processed) the run answers < 50 % of the split's queries  SELECTION_MISMATCH  hybrid/dense metadata inconsistent
  MISSING_METADATA best.json / selection.json absent or unreadable
Warnings: UNEXPECTED (a run no metadata claims), STALE_MANIFEST (manifest and disk disagree). INFO: counts.

`runs.json` is only written when there are no errors (or with --allow-partial, which lists what was excluded). A hand-written runs.json
keeps working exactly as before: pass it to run_all / verify_integration; --check-config applies the pair checks to it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path, PurePosixPath

from p4.eval.trec_io import read_qrels

MANIFEST_COLUMNS = ("member", "system", "split", "path", "queries", "lines", "six_columns")
MIN_COVERAGE = 0.5
DEFAULT_BASELINE = "p2_abstract_bm25"          # RQ1's reference: abstract-only BM25; overridable with --baseline


class DiscoveryError(RuntimeError):
    """The manifest itself is unusable (missing, wrong columns)."""


# ---------------------------------------------------------------------------------------------- reading
def read_manifest(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise DiscoveryError(f"manifest not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in MANIFEST_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DiscoveryError(f"{path.name} lacks column(s) {missing}; expected {list(MANIFEST_COLUMNS)}")
        return list(reader)


def scan_run(path: Path) -> dict:
    """Light read of a run file: row count, query ids, tags and malformed rows (never raises on content)."""
    qids, tags, bad, rows = set(), set(), 0, 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            f = line.split()
            if len(f) != 6:
                bad += 1
                continue
            qids.add(f[0])
            tags.add(f[5])
    return {"rows": rows, "qids": qids, "tags": tags, "bad_rows": bad}


def _load_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"{path} not found"
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        return None, f"{path} is not valid JSON ({err})"


# ---------------------------------------------------------------------------------------------- the discovery
class _Found:
    def __init__(self):
        self.problems: list[dict] = []
        self.pairs: dict[str, dict] = {}                  # system -> {"val", "test", "tag", "origin"} (relative paths + expected tag)
        self.scans: dict[Path, dict] = {}

    def add(self, severity: str, code: str, message: str, system: str | None = None):
        self.problems.append({"severity": severity, "code": code, "system": system, "message": message})

    def errors_for(self, system: str) -> list[dict]:
        return [p for p in self.problems if p["system"] == system and p["severity"] == "error"]

    def scan(self, path: Path) -> dict:
        if path not in self.scans:
            self.scans[path] = scan_run(path)
        return self.scans[path]


def _rel(path_text: str) -> str:
    p = PurePosixPath(path_text.replace("\\", "/"))
    return str(p.relative_to("runs")) if p.parts and p.parts[0] == "runs" else str(p)


def check_pair(found: _Found, system: str, val: Path, test: Path, expected_tag: str | None, qrels: dict | None = None) -> bool:
    """Content checks for one validation/test pair. Returns True if no error was added."""
    before = len(found.errors_for(system))
    scans = {}
    for role, path in (("validation", val), ("test", test)):
        if not path.is_file():
            found.add("error", "MISSING_FILE", f"{role} run {path} does not exist", system)
            continue
        scans[role] = s = found.scan(path)
        if s["rows"] == 0:
            found.add("error", "BAD_FORMAT", f"{role} run {path.name} is empty", system)
        if s["bad_rows"]:
            found.add("error", "BAD_FORMAT", f"{role} run {path.name} has {s['bad_rows']} row(s) that do not have 6 fields", system)
        if expected_tag is not None and s["rows"] and s["tags"] != {expected_tag}:
            found.add("error", "TAG_MISMATCH", f"{role} run {path.name} carries tag(s) {sorted(s['tags'])}, metadata says {expected_tag!r}", system)
    if len(scans) == 2:
        shared = scans["validation"]["qids"] & scans["test"]["qids"]
        if shared:
            found.add("error", "SPLIT_OVERLAP", f"validation and test runs answer {len(shared)} of the same queries "
                      f"(e.g. {sorted(shared)[:3]}); one of them is the wrong split", system)
    if qrels:
        for role, split in (("validation", "val"), ("test", "test")):
            if role not in scans or not scans[role]["qids"] or split not in qrels:
                continue
            judged, qids = set(qrels[split]), scans[role]["qids"]
            other = set(qrels["test" if split == "val" else "val"]) if ("test" if split == "val" else "val") in qrels else set()
            foreign = qids - judged
            if foreign and len(qids & other) > len(qids & judged):
                found.add("error", "WRONG_SPLIT", f"{role} run answers queries of the OTHER split ({len(qids & other)} of {len(qids)})", system)
            elif foreign:
                found.add("error", "WRONG_SPLIT", f"{role} run has {len(foreign)} queries that are not in the {split} qrels (e.g. {sorted(foreign)[:3]})", system)
            elif len(qids) / len(judged) < MIN_COVERAGE:
                found.add("error", "LOW_COVERAGE", f"{role} run answers {len(qids)} of {len(judged)} {split} queries", system)
    return len(found.errors_for(system)) == before


def discover(manifest_path, *, processed=None, root=None, baseline: str | None = None) -> dict:
    manifest_path = Path(manifest_path)
    rows = read_manifest(manifest_path)
    runs_dir = manifest_path.resolve().parent
    root = Path(root) if root else runs_dir.parent
    found = _Found()
    qrels = None
    if processed:
        qrels = {}
        for split in ("val", "test"):
            q = Path(processed) / "qrels" / f"qrels.{split}.txt"
            if q.is_file():
                qrels[split] = read_qrels(q)
            else:
                found.add("warn", "STALE_MANIFEST", f"{q} not found: the split check (WRONG_SPLIT / LOW_COVERAGE) is skipped for {split}")

    # index the manifest by path; duplicates are errors
    by_path: dict[str, list[dict]] = {}
    for r in rows:
        by_path.setdefault(_rel(r["path"]), []).append(r)
    for rel, group in by_path.items():
        if len(group) > 1:
            found.add("error", "DUPLICATE", f"{rel} is listed {len(group)} times in the manifest")
    claimed: set[str] = set()

    def locate(rel: str, system: str, role_split: str) -> Path | None:
        """Resolve a manifest entry for `system`; record DUPLICATE / MISSING_FILE / SPLIT_MISMATCH against it."""
        claimed.add(rel)
        group = by_path.get(rel)
        if not group:
            found.add("error", "MISSING_FILE", f"{rel} is required but not listed in the manifest", system)
            return None
        if len(group) > 1:
            found.add("error", "DUPLICATE", f"{rel} is listed {len(group)} times in the manifest", system)
        if group[0]["split"] != role_split:
            found.add("error", "SPLIT_MISMATCH", f"manifest says split {group[0]['split']!r} for {rel}, which is used as {role_split!r}", system)
        if str(group[0]["six_columns"]).strip().lower() not in ("true", "1"):
            found.add("error", "BAD_FORMAT", f"manifest marks {rel} as not six-column", system)
        path = root / group[0]["path"]
        if not path.is_file():
            found.add("error", "MISSING_FILE", f"{rel} is in the manifest but the file {path} does not exist", system)
            return None
        return path

    paths: dict[str, tuple[Path | None, Path | None, str | None]] = {}      # system -> (val, test, expected tag)

    # ---- P2 -----------------------------------------------------------------------------------------------
    best, err = _load_json(runs_dir / "p2" / "best.json")
    best_tags: dict[str, str] = {}
    if err:
        found.add("error", "MISSING_METADATA", f"P2 best.json unusable: {err}; P2 runs cannot be paired", "p2")
    else:
        for key, cfg in best.items():
            name = f"p2_{key}"
            tag = cfg.get("tag") if isinstance(cfg, dict) else None
            if not tag:
                found.add("error", "MISSING_METADATA", f"best.json entry {key!r} has no tag", name)
                continue
            best_tags[tag] = name
            paths[name] = (locate(f"p2/validation/{tag}.run", name, "val"), locate(f"p2/test/{key}.run", name, "test"), tag)
    sel_file = runs_dir / "p2" / "fusion" / "selection.txt"
    fusion_tag = sel_file.read_text(encoding="utf-8").strip() if sel_file.is_file() else None
    fusion_val_rel, fusion_test_rel = "p2/fusion/bm25.validation.run", "p2/fusion/bm25.test.run"
    has_fusion = fusion_val_rel in by_path or fusion_test_rel in by_path
    if has_fusion:
        if fusion_tag is None:
            found.add("error", "MISSING_METADATA", "p2/fusion/selection.txt is missing: the BM25 fusion pair cannot be tied to a configuration", "p2_bm25_fusion")
        else:
            paths["p2_bm25_fusion"] = (locate(fusion_val_rel, "p2_bm25_fusion", "val"), locate(fusion_test_rel, "p2_bm25_fusion", "test"), fusion_tag)

    # ---- P3 dense -----------------------------------------------------------------------------------------
    dense: dict[str, dict[str, str]] = {}
    for rel in sorted(by_path):
        parts = PurePosixPath(rel).parts
        if len(parts) == 3 and parts[:2] == ("p3", "dense"):
            stem = parts[2].removesuffix(".run")
            name, _, split = stem.rpartition(".")
            if split in ("val", "test") and name:
                dense.setdefault(name, {})[split] = rel
            else:
                found.add("warn", "UNEXPECTED", f"{rel}: cannot read a system name and split (expected <name>.val.run or <name>.test.run)")
                claimed.add(rel)
    for name, files in sorted(dense.items()):
        system = f"p3_{name}"
        for split in ("val", "test"):
            if split not in files:
                found.add("error", "UNPAIRED", f"dense system {name!r} has no {split} run in the manifest", system)
        paths[system] = tuple(locate(files[s], system, s) if s in files else None for s in ("val", "test")) + (f"dense_{name}",)
    selection, sel_err = _load_json(runs_dir / "p3" / "hybrid" / "selection.json")

    # ---- P3 hybrid ----------------------------------------------------------------------------------------
    hybrid_rels = {"val": "p3/hybrid/hybrid_rrf.val.run", "test": "p3/hybrid/hybrid_rrf.test.run"}
    if any(r in by_path for r in hybrid_rels.values()):
        system = "p3_hybrid_rrf"
        if sel_err:
            found.add("error", "MISSING_METADATA", f"hybrid selection.json unusable: {sel_err}", system)
        else:
            tag = f"hybrid_rrf_k{selection.get('rrf_k')}"
            for split, rel in hybrid_rels.items():
                if rel not in by_path:
                    found.add("error", "UNPAIRED", f"hybrid has no {split} run in the manifest", system)
            paths[system] = tuple(locate(hybrid_rels[s], system, s) if hybrid_rels[s] in by_path else None for s in ("val", "test")) + (tag,)
    if selection and not sel_err:
        listed = set((selection.get("dense_validation_mrr10") or {}))
        have = {n for n, f in dense.items() if "val" in f}
        if listed - have:
            found.add("error", "SELECTION_MISMATCH", f"selection.json lists dense system(s) {sorted(listed - have)} that have no validation run in the manifest", "p3_hybrid_rrf")
        if have - listed and listed:
            found.add("warn", "STALE_MANIFEST", f"dense validation run(s) {sorted(have - listed)} are not in selection.json: the hybrid choice was made without them")

    # ---- every pair: content checks -------------------------------------------------------------------------
    for system, (val, test, tag) in paths.items():
        if val is not None and test is not None:
            check_pair(found, system, val, test, tag, qrels)
    if "p3_hybrid_rrf" in paths and selection and not sel_err:
        dense_name = selection.get("dense_system")
        if f"p3_{dense_name}" not in paths:
            found.add("error", "SELECTION_MISMATCH", f"selection.json chose dense system {dense_name!r}, which is not in the manifest", "p3_hybrid_rrf")
        elif found.errors_for(f"p3_{dense_name}"):
            found.add("error", "SELECTION_MISMATCH", f"its dense system {dense_name!r} failed validation", "p3_hybrid_rrf")
        bm25_sel = selection.get("bm25_system")
        if fusion_tag and bm25_sel and bm25_sel != fusion_tag:
            found.add("error", "SELECTION_MISMATCH", f"selection.json used BM25 {bm25_sel!r} but p2/fusion/selection.txt says {fusion_tag!r}", "p3_hybrid_rrf")
        if "p2_bm25_fusion" in paths and found.errors_for("p2_bm25_fusion"):
            found.add("error", "SELECTION_MISMATCH", "its BM25 side (p2 fusion pair) failed validation", "p3_hybrid_rrf")

    # ---- manifest versus disk, unclaimed runs, the manifest's own counts -----------------------------------------
    grid = [rel for rel in by_path if rel.startswith("p2/validation/") and rel not in claimed]
    for rel in sorted(by_path):
        if rel in claimed or rel in grid:
            continue
        found.add("warn", "UNEXPECTED", f"{rel} is in the manifest but no metadata (best.json / selection.json / naming) claims it; ignored")
    if grid:
        found.add("info", "UNEXPECTED", f"{len(grid)} P2 validation run(s) are tuning-grid runs, not systems (expected)")
    on_disk = {p.relative_to(runs_dir).as_posix() for p in runs_dir.rglob("*.run")}
    for rel in sorted(on_disk - set(by_path)):
        found.add("warn", "STALE_MANIFEST", f"{rel} exists on disk but is not in the manifest (regenerate it with scripts/make_manifest.py)")
    for rel, group in by_path.items():
        path = root / group[0]["path"]
        if path.is_file() and path in found.scans:
            s = found.scans[path]
            if str(s["rows"]) != str(group[0]["lines"]).strip() or str(len(s["qids"])) != str(group[0]["queries"]).strip():
                found.add("warn", "STALE_MANIFEST", f"{rel}: manifest says {group[0]['queries']} queries / {group[0]['lines']} lines, "
                          f"the file has {len(s['qids'])} / {s['rows']}")

    return _assemble(found, paths, best_tags, fusion_tag, runs_dir, processed, baseline)


def _assemble(found: _Found, paths: dict, best_tags: dict, fusion_tag, runs_dir: Path, processed, baseline: str | None) -> dict:
    systems, excluded = {}, {}
    for system, (val, test, tag) in paths.items():
        errs = [p["message"] for p in found.errors_for(system)]
        if system == "p2_bm25_fusion" and fusion_tag in best_tags and not errs:
            found.add("info", "UNEXPECTED", f"p2 fusion pair has the same configuration as {best_tags[fusion_tag]}: validated, not listed twice")
            continue
        if errs or val is None or test is None:
            excluded[system] = errs or ["incomplete"]
        else:
            systems[system] = {"val": str(val.resolve()), "test": str(test.resolve()), "tag": tag}
    for p in found.problems:                                   # a system that never got paths (e.g. best.json missing) is excluded too
        if p["severity"] == "error" and p["system"] and p["system"] not in paths:
            excluded.setdefault(p["system"], []).append(p["message"])
    if baseline is not None and baseline not in systems:
        found.add("error", "SELECTION_MISMATCH", f"baseline {baseline!r} is not a valid system; valid: {sorted(systems)}")
    chosen = baseline or (DEFAULT_BASELINE if DEFAULT_BASELINE in systems else next(iter(systems), None))
    errors = [p for p in found.problems if p["severity"] == "error"]
    config: dict = {"systems": {n: {"val": s["val"], "test": s["test"]} for n, s in systems.items()}}
    if chosen:
        config["baseline"] = chosen
    for key, name in (("efficiency_csv", "efficiency.csv"), ("tuning_csv", "tuning.csv")):
        if (runs_dir / "p2" / name).is_file():
            config.setdefault("efficiency", {})[key] = str((runs_dir / "p2" / name).resolve())
    if processed:
        for key, name in (("functions", "context_functions.parquet"), ("index_contexts", "index_contexts.parquet")):
            if (Path(processed) / name).is_file():
                config[key] = str((Path(processed) / name).resolve())
    config["_discovery"] = {"complete": not errors, "excluded": excluded, "baseline_default": baseline is None,
                            "note": "generated by p4.experiments.discover_runs; pairing verified from metadata and file contents"}
    return {"config": config, "systems": systems, "excluded": excluded, "problems": found.problems, "errors": len(errors),
            "warnings": sum(p["severity"] == "warn" for p in found.problems)}


# ---------------------------------------------------------------------------------------------- a hand-written runs.json
def check_config(config_path, *, processed=None) -> dict:
    """Apply the content checks (tags, overlap, split) to a manual runs.json. Its tags are not known, so only consistency is checked."""
    config_path = Path(config_path)
    raw, err = _load_json(config_path)
    if err:
        raise DiscoveryError(err)
    base = config_path.resolve().parent
    found, qrels = _Found(), None
    if processed:
        qrels = {s: read_qrels(Path(processed) / "qrels" / f"qrels.{s}.txt") for s in ("val", "test")
                 if (Path(processed) / "qrels" / f"qrels.{s}.txt").is_file()}
    for name, files in (raw.get("systems") or {}).items():
        val, test = (base / files[s] if not Path(files[s]).is_absolute() else Path(files[s]) for s in ("val", "test"))
        if val.resolve() == test.resolve():
            found.add("error", "SPLIT_OVERLAP", "the validation and test entries are the same file", name)
            continue
        check_pair(found, name, val, test, None, qrels)
        for role, path in (("validation", val), ("test", test)):
            if path.is_file() and len(found.scan(path)["tags"]) > 1:
                found.add("warn", "TAG_MISMATCH", f"{role} run {path.name} has more than one tag in column 6", name)
    errors = [p for p in found.problems if p["severity"] == "error"]
    return {"problems": found.problems, "errors": len(errors), "systems_checked": len(raw.get("systems") or {})}


# ---------------------------------------------------------------------------------------------- output
def format_problems(problems: list[dict]) -> str:
    order = {"error": 0, "warn": 1, "info": 2}
    lines = []
    for p in sorted(problems, key=lambda p: (order[p["severity"]], p["code"], p["system"] or "")):
        lines.append(f"[{p['severity'].upper():5s}] {p['code']:<18s} {('(' + p['system'] + ') ') if p['system'] else ''}{p['message']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, help="runs/MANIFEST.csv")
    ap.add_argument("--root", type=Path, help="folder the manifest's paths are relative to (default: the parent of the manifest's folder)")
    ap.add_argument("--processed", type=Path, help="P1's data/processed folder: also checks the runs' queries against the qrels")
    ap.add_argument("--out", type=Path, help="runs.json to write (must not exist)")
    ap.add_argument("--report", type=Path, help="JSON report of the discovery (must not exist)")
    ap.add_argument("--baseline", help=f"baseline system name (default {DEFAULT_BASELINE} if present, else the first system)")
    ap.add_argument("--allow-partial", action="store_true", help="write runs.json for the valid systems even though some were excluded")
    ap.add_argument("--check-config", type=Path, help="check a hand-written runs.json instead of discovering")
    args = ap.parse_args(argv)
    try:
        if args.check_config:
            result = check_config(args.check_config, processed=args.processed)
            print(format_problems(result["problems"]) or "no problems")
            print(f"checked {result['systems_checked']} system(s): {result['errors']} error(s)")
            return 0 if not result["errors"] else 2
        if not args.manifest or not args.out:
            ap.error("--manifest and --out are required (or use --check-config)")
        for target in (args.out, args.report):
            if target and Path(target).exists():
                raise DiscoveryError(f"{target} already exists; refusing to overwrite")
        result = discover(args.manifest, processed=args.processed, root=args.root, baseline=args.baseline)
    except (DiscoveryError, FileNotFoundError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(format_problems(result["problems"]))
    print(f"\n{len(result['systems'])} system(s) paired and verified, {len(result['excluded'])} excluded, "
          f"{result['errors']} error(s), {result['warnings']} warning(s)")
    for name in result["systems"]:
        print(f"  ok  {name}")
    for name, why in result["excluded"].items():
        print(f"  --  {name}: {why[0]}")
    if args.report:
        Path(args.report).write_text(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2), encoding="utf-8")
    if result["errors"] and not args.allow_partial:
        print("runs.json NOT written because of the errors above (fix them, or use --allow-partial)", file=sys.stderr)
        return 2
    if not result["systems"]:
        print("runs.json NOT written: no valid system", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(result["config"], indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
