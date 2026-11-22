"""Tests for p4.experiments.verify_integration. SYNTHETIC data (P1's generated sample and P3's committed synthetic runs) only."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from p4.experiments import verify_integration as VI

REPO = Path(__file__).resolve().parent.parent.parent
P3 = REPO / "p3" / "outputs"

pytestmark = pytest.mark.skipif(not (P3 / "hybrid_test.run").exists(), reason="P3's committed synthetic outputs are not present")


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    """A SYNTHETIC P1-style processed folder, written once."""
    from p4.analysis.synthetic_demo import write_synthetic_processed
    folder = tmp_path_factory.mktemp("pristine") / "processed"
    write_synthetic_processed(folder)
    return folder


@pytest.fixture
def project(pristine, tmp_path):
    """A fresh copy of the processed folder plus a configuration that points at P3's committed runs and P2's logs."""
    processed = tmp_path / "processed"
    shutil.copytree(pristine, processed)
    config = {"systems": {"bm25": {"val": str(P3 / "p2_quick/fusion/bm25.validation.run"), "test": str(P3 / "p2_quick/fusion/bm25.test.run")}},
              "p2_check": {"tuning_csv": str(P3 / "p2_quick/tuning.csv"), "validation_dir": str(P3 / "p2_quick/validation"),
                           "qrels": str(processed / "qrels" / "qrels.val.txt")}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return {"processed": processed, "config": path, "root": tmp_path, "cfg": config}


def run_verify(project):
    return VI.verify(project["processed"], project["config"])


def by_id(report, check_id, contains=""):
    return [c for c in report.checks if c["id"] == check_id and contains in c["description"]]


def test_a_consistent_synthetic_project_passes_every_check(project):
    report = run_verify(project)
    assert report.failed == [], [c["detail"] for c in report.failed]
    statuses = {(c["id"], c["status"]) for c in report.checks}
    assert ("S1", "pass") in statuses and ("I1", "pass") in statuses and ("Q1", "pass") in statuses
    assert ("L1", "pass") in statuses and ("R1", "pass") in statuses and ("P1", "pass") in statuses
    assert ("S2", "skipped") in statuses                                                  # no citation-function labels yet
    p2 = by_id(report, "P1")[0]
    assert "13 runs compared" in p2["detail"]                                             # all of P2's logged validation MRRs are reproduced


def test_cli_prints_a_summary_and_writes_json(project, capsys):
    out = project["root"] / "report.json"
    assert VI.main(["--processed", str(project["processed"]), "--config", str(project["config"]), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "[PASS   ] S1" in printed and "0 failed" in printed
    saved = json.loads(out.read_text())
    assert saved["failed"] == 0 and len(saved["checks"]) > 15
    assert VI.main(["--processed", str(project["processed"]), "--config", str(project["config"]), "--out", str(out)]) == 2     # no overwrite


# ---- every kind of inconsistency is caught -----------------------------------------------------------------------
def test_missing_column_in_a_p1_table(project):
    path = project["processed"] / "papers.parquet"
    pd.read_parquet(path).drop(columns=["citation_count"]).to_parquet(path, index=False)
    s1 = [c for c in by_id(run_verify(project), "S1", "papers.parquet")][0]
    assert s1["status"] == "fail" and s1["missing_columns"] == ["citation_count"]


def test_missing_table_and_missing_qrels(project):
    (project["processed"] / "edges.parquet").unlink()
    (project["processed"] / "qrels" / "qrels.test.txt").unlink()
    report = run_verify(project)
    assert by_id(report, "S1", "edges.parquet")[0]["status"] == "fail"
    assert by_id(report, "S1", "qrels.test.txt")[0]["status"] == "fail"


def test_duplicate_ids_and_unknown_references(project):
    papers = pd.read_parquet(project["processed"] / "papers.parquet")
    pd.concat([papers, papers.iloc[[0]]]).to_parquet(project["processed"] / "papers.parquet", index=False)
    edges = pd.read_parquet(project["processed"] / "edges.parquet")
    pd.concat([edges, edges.iloc[[0]].assign(cited_paper_id="NOT-A-PAPER")]).to_parquet(project["processed"] / "edges.parquet", index=False)
    report = run_verify(project)
    assert by_id(report, "I1", "paper_id is unique")[0]["status"] == "fail"
    assert by_id(report, "I1", "edges point only")[0]["status"] == "fail"


def test_invalid_split_name(project):
    queries = pd.read_parquet(project["processed"] / "queries.parquet")
    queries.loc[0, "split"] = "validation"
    queries.to_parquet(project["processed"] / "queries.parquet", index=False)
    assert by_id(run_verify(project), "I1", "split names")[0]["status"] == "fail"


def test_qrels_that_disagree_with_the_queries(project):
    path = project["processed"] / "qrels" / "qrels.test.txt"
    lines = path.read_text().splitlines()
    first = lines[0].split()
    lines[0] = f"{first[0]} 0 SOME-OTHER-PAPER 1"                                              # wrong relevant paper
    path.write_text("".join(line + chr(10) for line in lines[:-1]))                            # and one query dropped
    q1 = by_id(run_verify(project), "Q1", "qrels.test.txt")[0]
    assert q1["status"] == "fail" and "missing from qrels 1" in q1["detail"] and "different relevant paper 1" in q1["detail"]


def test_run_with_unknown_documents_or_unanswered_queries(project):
    run = project["root"] / "partial.run"
    full = (P3 / "p2_quick/fusion/bm25.test.run").read_text().splitlines()
    kept = [l for l in full if not l.startswith(full[0].split()[0] + " ")]                     # drop one whole query
    kept.append("q000999 Q0 NOT-A-PAPER 1 1.0 x")                                              # an unknown document for an unknown query
    run.write_text("".join(l + chr(10) for l in kept))
    project["cfg"]["systems"]["bm25"]["test"] = str(run)
    project["config"].write_text(json.dumps(project["cfg"]))
    r1 = by_id(run_verify(project), "R1", "bm25 test")[0]
    assert r1["status"] == "fail" and r1["unknown_documents"] == 1 and r1["answered_share"] < 1.0


def test_queries_from_another_split_only_warn(project):
    run = project["root"] / "all.run"
    val = (P3 / "p2_quick/fusion/bm25.validation.run").read_text()
    test = (P3 / "p2_quick/fusion/bm25.test.run").read_text()
    run.write_text(test + val)                                                                 # test run that also contains validation queries
    project["cfg"]["systems"]["bm25"]["test"] = str(run)
    project["config"].write_text(json.dumps(project["cfg"]))
    r1 = by_id(run_verify(project), "R1", "bm25 test")[0]
    assert r1["status"] == "warn" and "queries outside this split: 23" in r1["detail"]


def test_splits_sharing_a_citing_paper_fail_the_leakage_gate(project):
    queries = pd.read_parquet(project["processed"] / "queries.parquet")
    donor = queries[queries.split == "val"].iloc[0]["citing_paper_id"]
    queries.loc[queries[queries.split == "test"].index[0], "citing_paper_id"] = donor
    queries.to_parquet(project["processed"] / "queries.parquet", index=False)
    assert by_id(run_verify(project), "L1", "disjoint")[0]["status"] == "fail"


def test_a_context_file_with_a_query_context_fails_and_a_clean_one_passes(project):
    queries = pd.read_parquet(project["processed"] / "queries.parquet")
    index = pd.read_parquet(project["processed"] / "index_contexts.parquet")
    clean = project["root"] / "clean.jsonl"
    clean.write_text("".join(json.dumps({"context_id": c, "cited_paper_id": p, "sentence": t}) + chr(10)
                             for c, p, t in zip(index.context_id[:50], index.cited_paper_id[:50], index.text[:50])))
    leaky = project["root"] / "leaky.jsonl"
    leaky.write_text(json.dumps({"context_id": queries.iloc[0]["context_id"], "cited_paper_id": queries.iloc[0]["cited_paper_id"],
                                 "sentence": queries.iloc[0]["text"]}) + chr(10))
    project["cfg"]["context_files"] = [str(clean), str(leaky)]
    project["config"].write_text(json.dumps(project["cfg"]))
    checks = by_id(run_verify(project), "L1", "context file")
    assert {c["description"].split()[2].rstrip(":"): c["status"] for c in checks} == {"clean.jsonl": "pass", "leaky.jsonl": "fail"}


def test_a_p2_log_that_the_harness_cannot_reproduce_fails(project):
    tuning = pd.read_csv(project["cfg"]["p2_check"]["tuning_csv"])
    tuning.loc[0, "validation_mrr_at_10"] += 0.05                                              # P2 "logged" a wrong number
    bad = project["root"] / "tuning.csv"
    tuning.to_csv(bad, index=False)
    project["cfg"]["p2_check"]["tuning_csv"] = str(bad)
    project["config"].write_text(json.dumps(project["cfg"]))
    p2 = by_id(run_verify(project), "P1")[0]
    assert p2["status"] == "fail" and "P2 logged" in p2["detail"]


def test_function_labels_are_validated_when_present(project):
    index = pd.read_parquet(project["processed"] / "index_contexts.parquet")
    labels = pd.DataFrame({"context_id": index.context_id, "function": "method", "confidence": 0.9})
    labels.to_parquet(project["processed"] / "context_functions.parquet", index=False)
    assert by_id(run_verify(project), "S2")[0]["status"] == "pass"
    labels.iloc[:3, labels.columns.get_loc("function")] = "opinion"
    labels.iloc[3:10] = labels.iloc[3:10].assign(context_id="no-such-context")                  # also leaves some contexts unlabelled
    labels.to_parquet(project["processed"] / "context_functions.parquet", index=False)
    s2 = by_id(run_verify(project), "S2")[0]
    assert s2["status"] == "fail" and "opinion" in s2["detail"]


def test_configuration_errors_exit_2(project, capsys, tmp_path):
    assert VI.main(["--processed", str(project["processed"]), "--config", str(tmp_path / "nope.json"), "--out", str(tmp_path / "o.json")]) == 2
    assert "configuration file not found" in capsys.readouterr().err
    project["cfg"]["systems"]["bm25"]["val"] = str(tmp_path / "missing.run")                   # a missing run file is a FAILED CHECK, not an abort
    project["config"].write_text(json.dumps(project["cfg"]))
    out = tmp_path / "o2.json"
    assert VI.main(["--processed", str(project["processed"]), "--config", str(project["config"]), "--out", str(out)]) == 2
    printed = capsys.readouterr().out
    assert "[FAIL   ] R1 bm25 val run file is valid" in printed and json.loads(out.read_text())["failed"] >= 1
