"""Verify an integrated project before any experiment is run on it (read-only).

    python -m p4.experiments.verify_integration --processed <P1 processed> --config runs.json --out report.json

Uses the same configuration file as `p4.experiments.run_all` (optionally with a "p2_check" block). Checks, each reported as
pass / fail / warn / skipped with counts:

  S1  P1 tables exist with the columns of citeds/schema.py (papers, contexts, index_contexts, edges, queries) and qrels files exist
  S2  context_functions.parquet (if present): columns, labels in {background, method, result}, covers every context
  I1  IDs: paper_id and query_id unique; edges, queries and index contexts point at known papers / contexts; split names valid
  Q1  qrels agree with queries.parquet: same queries per split, one relevant paper, equal to the query's cited paper
  R1  every run file: documents known to papers.parquet; share of the split's judged queries answered; queries not in the split
  L1  leakage gate (p4.graph.leakage_audit): splits disjoint by citing paper; context files given to P2/P3 free of query contexts
  P1  (optional) P2's tuning.csv: P4's harness reproduces every logged validation MRR@10 from the validation run files

"p2_check": {"tuning_csv": "...", "validation_dir": "...", "qrels": "qrels.val.txt"}   in the configuration enables P1.
Exit code 0 = no failure (warnings allowed), 2 = at least one failure or a configuration error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from p4.eval.harness import evaluate_runs
from p4.eval.trec_io import TrecFormatError, read_qrels, read_run
from p4.graph import leakage_audit

EXPECTED = {
    "papers.parquet": ["paper_id", "title", "abstract", "year", "categories", "citation_count", "num_contexts"],
    "contexts.parquet": ["context_id", "sentence_id", "citing_paper_id", "cited_paper_id", "text", "prev_text", "next_text", "section",
                         "num_cited_in_sentence"],
    "index_contexts.parquet": ["context_id", "sentence_id", "citing_paper_id", "cited_paper_id", "text", "prev_text", "next_text", "section",
                               "num_cited_in_sentence"],
    "edges.parquet": ["citing_paper_id", "cited_paper_id", "num_contexts"],
    "queries.parquet": ["query_id", "split", "context_id", "sentence_id", "citing_paper_id", "cited_paper_id", "text", "cited_citation_count"],
}
FUNCTION_LABELS = {"background", "method", "result"}
SPLITS = ("train", "val", "test")


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, check_id: str, description: str, status: str, detail: str = "", **data) -> None:
        self.checks.append({"id": check_id, "description": description, "status": status, "detail": detail, **data})

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "fail"]


def _columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq
    return list(pq.ParquetFile(path).schema_arrow.names)


def check_p1_tables(processed: Path, report: Report) -> dict:
    """S1, S2, I1, Q1. Returns the tables needed by later checks (papers, queries, ...)."""
    tables: dict = {}
    for name, columns in EXPECTED.items():
        path = processed / name
        if not path.is_file():
            report.add("S1", f"{name} exists with the P1 columns", "fail", "file not found")
            continue
        found = _columns(path)
        missing = [c for c in columns if c not in found]
        report.add("S1", f"{name} exists with the P1 columns", "fail" if missing else "pass",
                   f"missing columns {missing}" if missing else f"{len(found)} columns", missing_columns=missing)
        if not missing or name in ("papers.parquet", "queries.parquet", "edges.parquet", "index_contexts.parquet"):
            tables[name] = pd.read_parquet(path, columns=[c for c in columns if c in found])
    for split in SPLITS:
        ok = (processed / "qrels" / f"qrels.{split}.txt").is_file()
        report.add("S1", f"qrels/qrels.{split}.txt exists", "pass" if ok else "fail", "" if ok else "file not found")
    fpath = processed / "context_functions.parquet"
    if not fpath.is_file():
        report.add("S2", "context_functions.parquet (citation-function labels)", "skipped", "file not present (the function analysis cannot run)")
    else:
        f = pd.read_parquet(fpath)
        bad_cols = [c for c in ("context_id", "function", "confidence") if c not in f.columns]
        if bad_cols:
            report.add("S2", "context_functions.parquet columns", "fail", f"missing {bad_cols}")
        else:
            bad_labels = sorted(set(f["function"].dropna()) - FUNCTION_LABELS)
            contexts = tables.get("index_contexts.parquet")
            uncovered = int((~contexts["context_id"].isin(set(f["context_id"]))).sum()) if contexts is not None else None
            status = "fail" if bad_labels or uncovered else "pass"
            report.add("S2", "context_functions.parquet: valid labels and covers every index context", status,
                       f"unknown labels {bad_labels}; index contexts without a label: {uncovered}")
    papers, queries = tables.get("papers.parquet"), tables.get("queries.parquet")
    edges, index_contexts = tables.get("edges.parquet"), tables.get("index_contexts.parquet")
    if papers is not None:
        dup = int(papers["paper_id"].duplicated().sum())
        report.add("I1", "paper_id is unique", "fail" if dup else "pass", f"{dup} duplicates")
    if queries is not None:
        dup = int(queries["query_id"].duplicated().sum())
        report.add("I1", "query_id is unique", "fail" if dup else "pass", f"{dup} duplicates")
        bad = sorted(set(queries["split"]) - set(SPLITS))
        report.add("I1", "split names are train / val / test", "fail" if bad else "pass", f"unexpected {bad}")
    if papers is not None:
        known = set(papers["paper_id"])
        for label, table, cols in (("edges", edges, ["citing_paper_id", "cited_paper_id"]), ("queries", queries, ["citing_paper_id", "cited_paper_id"]),
                                   ("index contexts", index_contexts, ["cited_paper_id"])):
            if table is not None:
                unknown = int(sum((~table[c].isin(known)).sum() for c in cols))
                report.add("I1", f"{label} point only at papers in papers.parquet", "fail" if unknown else "pass", f"{unknown} unknown references")
    if queries is not None:
        for split in SPLITS:
            path = processed / "qrels" / f"qrels.{split}.txt"
            if not path.is_file():
                continue
            try:
                qrels = read_qrels(path)
            except TrecFormatError as err:
                report.add("Q1", f"qrels.{split}.txt agrees with queries.parquet", "fail", str(err))
                continue
            part = queries[queries["split"] == split].set_index("query_id")["cited_paper_id"]
            only_q, only_qrels = set(part.index) - set(qrels), set(qrels) - set(part.index)
            wrong = [q for q in set(part.index) & set(qrels) if set(qrels[q]) != {part[q]}]
            report.add("Q1", f"qrels.{split}.txt agrees with queries.parquet", "fail" if (only_q or only_qrels or wrong) else "pass",
                       f"{len(part)} queries; missing from qrels {len(only_q)}; not in queries {len(only_qrels)}; different relevant paper {len(wrong)}")
    return tables


def check_runs(cfg: dict, tables: dict, report: Report) -> None:
    """R1: documents known, queries answered, queries from other splits."""
    papers, queries = tables.get("papers.parquet"), tables.get("queries.parquet")
    if papers is None or queries is None:
        report.add("R1", "run files", "skipped", "papers.parquet or queries.parquet unavailable")
        return
    known = set(papers["paper_id"])
    for name, files in cfg["systems"].items():
        for split in ("val", "test"):
            path = files[split]
            try:
                run = read_run(path)
            except (TrecFormatError, FileNotFoundError) as err:
                report.add("R1", f"{name} {split} run file is valid", "fail", str(err))
                continue
            judged = set(queries.loc[queries["split"] == split, "query_id"])
            docs = {d for ranked in run.values() for d, _ in ranked}
            unknown = len(docs - known)
            answered = len(judged & set(run))
            share = answered / len(judged) if judged else 0.0
            other = len(set(run) - judged)
            status = "fail" if unknown or share < 1.0 else ("warn" if other else "pass")
            report.add("R1", f"{name} {split} run: documents known, all judged queries answered", status,
                       f"{answered}/{len(judged)} judged queries answered ({share:.1%}); documents not in papers.parquet: {unknown}; "
                       f"queries outside this split: {other}", system=name, split=split, answered_share=share, unknown_documents=unknown)


def check_leakage(processed: Path, cfg: dict, report: Report) -> None:
    try:
        audit = leakage_audit.audit(processed, context_files=cfg["context_files"])
    except (FileNotFoundError, KeyError, ValueError) as err:
        report.add("L1", "leakage gate", "fail", f"could not run the audit: {err}")
        return
    disjoint = audit["split_audit"]["disjoint_by_citing_paper"]
    report.add("L1", "splits are disjoint by citing paper", "pass" if disjoint else "fail",
               f"{audit['split_audit']['citing_papers']} citing papers; shared across splits: {audit['split_audit']['citing_papers_in_more_than_one_split']}")
    for entry in audit.get("context_files", []):
        report.add("L1", f"context file {Path(entry['file']).name}: only leakage-controlled index contexts", "pass" if entry["ok"] else "fail",
                   f"{entry['rows']} rows; not in index {entry['context_ids_not_in_index']}; query contexts {entry['query_context_ids']}; "
                   f"query sentences {entry['sentences_equal_to_a_query_text']}")
    if not cfg["context_files"]:
        report.add("L1", "context files given to P2/P3 checked", "skipped", "no 'context_files' in the configuration")


def check_p2(cfg: dict, report: Report) -> None:
    block = cfg.get("p2_check")
    if not block:
        report.add("P1", "P2's logged validation MRR@10 reproduced by P4's harness", "skipped", "no 'p2_check' block in the configuration")
        return
    tuning = pd.read_csv(block["tuning_csv"])
    qrels = read_qrels(block["qrels"])
    worst, mismatches, n = 0.0, [], 0
    for row in tuning.itertuples():
        path = Path(block["validation_dir"]) / f"{row.run}.run"
        if not path.is_file():
            mismatches.append(f"{row.run}: run file missing")
            continue
        mrr = evaluate_runs({"x": read_run(path)}, qrels, (10,), min_coverage=0).summary.loc[0, "MRR@10"]
        n += 1
        worst = max(worst, abs(mrr - row.validation_mrr_at_10))
        if abs(mrr - row.validation_mrr_at_10) > 1e-9:
            mismatches.append(f"{row.run}: P2 logged {row.validation_mrr_at_10:.6f}, P4 computes {mrr:.6f}")
    report.add("P1", "P2's logged validation MRR@10 reproduced by P4's harness", "fail" if mismatches else "pass",
               f"{n} runs compared, largest difference {worst:.2e}" + (f"; {len(mismatches)} problems, e.g. {mismatches[:2]}" if mismatches else ""))


def verify(processed, config_path) -> Report:
    from .run_all import load_config
    processed = Path(processed)
    cfg = load_config(config_path, processed, check_files=False)          # missing files become failed checks, not an abort
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if raw.get("p2_check"):
        base = Path(config_path).resolve().parent
        cfg["p2_check"] = {k: (Path(v) if Path(v).is_absolute() else base / v) for k, v in raw["p2_check"].items()}
    report = Report()
    tables = check_p1_tables(processed, report)
    check_runs(cfg, tables, report)
    check_leakage(processed, cfg, report)
    check_p2(cfg, report)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="JSON report to write (must not exist)")
    args = p.parse_args(argv)
    if args.out.exists():
        print(f"error: {args.out} already exists", file=sys.stderr)
        return 2
    from .run_all import ExperimentError
    try:
        report = verify(args.processed, args.config)
    except (ExperimentError, FileNotFoundError, KeyError, ValueError, TrecFormatError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"checks": report.checks, "failed": len(report.failed)}, indent=2), encoding="utf-8")
    for c in report.checks:
        print(f"[{c['status'].upper():7s}] {c['id']} {c['description']}: {c['detail']}")
    print(f"\n{len(report.failed)} failed, {sum(c['status'] == 'warn' for c in report.checks)} warnings; report: {args.out}")
    return 2 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
