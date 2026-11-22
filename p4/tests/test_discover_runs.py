"""Tests for p4.experiments.discover_runs. TOY manifests, qrels and run files written to a temporary folder: no project data, no results."""
import csv
import json
from pathlib import PurePosixPath

import pytest

from p4.experiments import discover_runs as DR
from p4.experiments.run_all import load_config

VAL = ["v1", "v2", "v3", "v4"]
TEST = ["t1", "t2", "t3", "t4"]
ABS_TAG, CTX_TAG = "abstract-w0-plain-all-bm25-0", "context-w0-plain-all-bm25-0"
GRID_TAG = "abstract-w0-plain-all-tfidf-0"                                                   # a tuning-grid run that is not a system
SYSTEMS = {"p2_abstract_bm25", "p2_context_bm25", "p3_minilm_abstract", "p3_minilm_contexts", "p3_hybrid_rrf"}


def write_run(path, qids, tag):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{q} Q0 {d} {r} {10 - r}.0 {tag}\n" for q in qids for r, d in enumerate("ABC", 1)), encoding="utf-8")


def describe(runs, path):
    """The same rules as scripts/make_manifest.py (member, system, split from the folder layout)."""
    parts = path.relative_to(runs).parts
    name = path.name.removesuffix(".run")
    if parts[0] == "p2":
        member = "P2"
        system, split = (name.split(".")[0], name.split(".")[1]) if parts[1] == "fusion" else (name, parts[1])
    else:
        member = "P3"
        system, split = name.rsplit(".", 1)
        system = f"{parts[1]}/{system}"
    split = {"validation": "val"}.get(split, split)
    rows = [l.split() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"member": member, "system": system, "split": split, "path": "runs/" + path.relative_to(runs).as_posix(),
            "queries": len({r[0] for r in rows if len(r) == 6}), "lines": len(rows), "six_columns": all(len(r) == 6 for r in rows)}


def write_manifest(runs, extra_rows=()):
    rows = [describe(runs, p) for p in sorted(runs.rglob("*.run"))] + list(extra_rows)
    with open(runs / "MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(DR.MANIFEST_COLUMNS))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def project(tmp_path):
    runs = tmp_path / "runs"
    for key, tag in (("abstract_bm25", ABS_TAG), ("context_bm25", CTX_TAG)):
        write_run(runs / "p2" / "validation" / f"{tag}.run", VAL, tag)
        write_run(runs / "p2" / "test" / f"{key}.run", TEST, tag)
    write_run(runs / "p2" / "validation" / f"{GRID_TAG}.run", VAL, GRID_TAG)
    (runs / "p2" / "best.json").write_text(json.dumps({"abstract_bm25": {"tag": ABS_TAG, "model": "bm25"}, "context_bm25": {"tag": CTX_TAG, "model": "bm25"}}))
    write_run(runs / "p2" / "fusion" / "bm25.validation.run", VAL, CTX_TAG)
    write_run(runs / "p2" / "fusion" / "bm25.test.run", TEST, CTX_TAG)
    (runs / "p2" / "fusion" / "selection.txt").write_text(CTX_TAG + "\n")
    (runs / "p2" / "efficiency.csv").write_text("run,median_ms\nx,1\n")
    (runs / "p2" / "tuning.csv").write_text("run,median_ms\nx,1\n")
    for name in ("minilm_abstract", "minilm_contexts"):
        write_run(runs / "p3" / "dense" / f"{name}.val.run", VAL, f"dense_{name}")
        write_run(runs / "p3" / "dense" / f"{name}.test.run", TEST, f"dense_{name}")
    write_run(runs / "p3" / "hybrid" / "hybrid_rrf.val.run", VAL, "hybrid_rrf_k20")
    write_run(runs / "p3" / "hybrid" / "hybrid_rrf.test.run", TEST, "hybrid_rrf_k20")
    selection = {"dense_system": "minilm_contexts", "rrf_k": 20, "bm25_system": CTX_TAG,
                 "dense_validation_mrr10": {"minilm_abstract": 0.1, "minilm_contexts": 0.2}}
    (runs / "p3" / "hybrid" / "selection.json").write_text(json.dumps(selection))
    processed = tmp_path / "processed"
    (processed / "qrels").mkdir(parents=True)
    for split, qids in (("val", VAL), ("test", TEST)):
        (processed / "qrels" / f"qrels.{split}.txt").write_text("".join(f"{q} 0 A 1\n" for q in qids))
    write_manifest(runs)
    return {"root": tmp_path, "runs": runs, "processed": processed, "manifest": runs / "MANIFEST.csv", "selection": selection}


def run_discover(p, **kw):
    return DR.discover(p["manifest"], **kw)


def problems(result, code, severity=None):
    return [x for x in result["problems"] if x["code"] == code and (severity is None or x["severity"] == severity)]


# ---- the consistent project ----------------------------------------------------------------------------------
def test_consistent_toy_project_pairs_every_system_and_reports_no_error(project):
    result = run_discover(project, processed=project["processed"])
    assert result["errors"] == 0 and set(result["systems"]) == SYSTEMS and result["excluded"] == {}
    cfg = result["config"]
    runs = project["runs"].resolve()
    assert cfg["systems"]["p2_context_bm25"] == {"val": str(runs / "p2/validation" / f"{CTX_TAG}.run"), "test": str(runs / "p2/test/context_bm25.run")}
    assert cfg["systems"]["p3_hybrid_rrf"]["test"] == str(runs / "p3/hybrid/hybrid_rrf.test.run")
    assert cfg["baseline"] == "p2_abstract_bm25" and cfg["_discovery"]["complete"] is True
    assert set(cfg["efficiency"]) == {"efficiency_csv", "tuning_csv"}
    infos = [x["message"] for x in problems(result, "UNEXPECTED", "info")]
    assert any("tuning-grid" in m and "1 P2 validation" in m for m in infos)                # the grid run is expected, not a system
    assert any("same configuration as p2_context_bm25" in m for m in infos)                  # the fusion pair duplicates a P2 system
    assert "p2_bm25_fusion" not in result["systems"]


def test_the_generated_config_loads_in_run_all_like_a_manual_one(project, tmp_path):
    result = run_discover(project, processed=project["processed"])
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(result["config"]))
    cfg = load_config(path, project["processed"])
    assert set(cfg["systems"]) == SYSTEMS and cfg["baseline"] == "p2_abstract_bm25"
    manual = tmp_path / "manual.json"                                                         # a hand-written file still works unchanged
    manual.write_text(json.dumps({"systems": {"mine": {k: result["config"]["systems"]["p3_minilm_abstract"][k] for k in ("val", "test")}}}))
    assert list(load_config(manual, project["processed"])["systems"]) == ["mine"]


def test_baseline_can_be_chosen_and_must_exist(project):
    assert run_discover(project, baseline="p3_hybrid_rrf")["config"]["baseline"] == "p3_hybrid_rrf"
    bad = run_discover(project, baseline="nope")
    assert problems(bad, "SELECTION_MISMATCH") and bad["errors"] == 1


# ---- pairing is verified from metadata and contents, never from file names ----------------------------------------
def test_a_validation_run_with_the_wrong_tag_is_caught_even_though_the_file_name_matches(project):
    write_run(project["runs"] / "p2" / "validation" / f"{ABS_TAG}.run", VAL, CTX_TAG)         # the abstract file now holds the context run
    result = run_discover(project)
    assert "p2_abstract_bm25" in result["excluded"] and "p2_context_bm25" in result["systems"]
    [mismatch] = problems(result, "TAG_MISMATCH")
    assert mismatch["system"] == "p2_abstract_bm25" and ABS_TAG in mismatch["message"] and CTX_TAG in mismatch["message"]


def test_dense_runs_must_carry_their_own_name_in_the_tag(project):
    write_run(project["runs"] / "p3" / "dense" / "minilm_abstract.test.run", TEST, "dense_minilm_contexts")
    result = run_discover(project)
    assert problems(result, "TAG_MISMATCH")[0]["system"] == "p3_minilm_abstract" and "p3_minilm_abstract" in result["excluded"]
    assert "p3_hybrid_rrf" in result["systems"]                                              # it uses minilm_contexts, which is fine


def test_the_manifest_split_column_must_agree_with_the_role(project):
    rows = DR.read_manifest(project["manifest"])
    for r in rows:
        if r["path"].endswith("minilm_contexts.test.run"):
            r["split"] = "val"
    with open(project["manifest"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(DR.MANIFEST_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    result = run_discover(project)
    assert problems(result, "SPLIT_MISMATCH")[0]["system"] == "p3_minilm_contexts"
    assert "p3_minilm_contexts" in result["excluded"] and "p3_hybrid_rrf" in result["excluded"]        # the hybrid depends on it


def test_validation_and_test_runs_that_share_queries_are_rejected(project):
    write_run(project["runs"] / "p3" / "dense" / "minilm_abstract.test.run", TEST + VAL[:1], "dense_minilm_abstract")
    write_manifest(project["runs"])
    result = run_discover(project)
    assert problems(result, "SPLIT_OVERLAP")[0]["system"] == "p3_minilm_abstract"


def test_swapped_splits_are_found_only_with_the_qrels_and_say_so(project):
    p = project["runs"] / "p3" / "dense"
    write_run(p / "minilm_abstract.val.run", TEST, "dense_minilm_abstract")                  # "val" holds the test queries and vice versa
    write_run(p / "minilm_abstract.test.run", VAL, "dense_minilm_abstract")
    write_manifest(project["runs"])
    assert "p3_minilm_abstract" in run_discover(project)["systems"]                          # without qrels nothing contradicts the pair
    result = run_discover(project, processed=project["processed"])
    wrong = problems(result, "WRONG_SPLIT")
    assert {w["system"] for w in wrong} == {"p3_minilm_abstract"} and len(wrong) == 2 and "OTHER split" in wrong[0]["message"]
    assert "p3_minilm_abstract" in result["excluded"]


def test_a_run_that_answers_too_few_queries_is_flagged_with_the_qrels(project):
    write_run(project["runs"] / "p3" / "dense" / "minilm_abstract.test.run", TEST[:1], "dense_minilm_abstract")
    write_manifest(project["runs"])
    result = run_discover(project, processed=project["processed"])
    assert problems(result, "LOW_COVERAGE")[0]["system"] == "p3_minilm_abstract" and "1 of 4" in problems(result, "LOW_COVERAGE")[0]["message"]


# ---- missing, duplicate, unexpected -----------------------------------------------------------------------------
def test_a_run_listed_in_the_manifest_but_deleted_from_disk_is_missing(project):
    (project["runs"] / "p3" / "dense" / "minilm_abstract.test.run").unlink()                  # the manifest still lists it
    result = run_discover(project)
    [missing] = problems(result, "MISSING_FILE")
    assert missing["system"] == "p3_minilm_abstract" and "does not exist" in missing["message"] and "p3_minilm_abstract" in result["excluded"]


def test_a_dense_system_with_only_one_split_is_unpaired(project):
    (project["runs"] / "p3" / "dense" / "minilm_abstract.test.run").unlink()
    write_manifest(project["runs"])                                                          # the manifest is regenerated: the run is simply absent
    result = run_discover(project)
    assert problems(result, "UNPAIRED")[0]["system"] == "p3_minilm_abstract" and "no test run" in problems(result, "UNPAIRED")[0]["message"]
    assert any("selection.json" not in x["message"] for x in problems(result, "UNPAIRED"))
    assert "p3_minilm_abstract" not in result["systems"] and "p3_hybrid_rrf" in result["systems"]


def test_a_p2_system_in_best_json_without_a_test_run_is_missing(project):
    (project["runs"] / "p2" / "test" / "abstract_bm25.run").unlink()
    write_manifest(project["runs"])
    [missing] = problems(run_discover(project), "MISSING_FILE")
    assert missing["system"] == "p2_abstract_bm25" and "p2/test/abstract_bm25.run" in missing["message"] and "not listed" in missing["message"]


def test_duplicate_manifest_rows_are_errors_for_the_system_that_uses_them(project):
    rows = DR.read_manifest(project["manifest"])
    dup = next(r for r in rows if r["path"].endswith("context_bm25.run") and "/test/" in r["path"])
    write_manifest(project["runs"], extra_rows=[dup])
    result = run_discover(project)
    dups = problems(result, "DUPLICATE", "error")
    assert {d["system"] for d in dups} == {None, "p2_context_bm25"} and "p2_context_bm25" in result["excluded"]
    assert "p2_abstract_bm25" in result["systems"]


def test_unexpected_and_stale_files_are_warnings_that_do_not_exclude_anything(project):
    write_run(project["runs"] / "p2" / "test" / "extra_bm25.run", TEST, "extra")             # in the manifest, claimed by nothing
    write_manifest(project["runs"])
    write_run(project["runs"] / "p3" / "dense" / "late.val.run", VAL, "dense_late")          # on disk, not in the manifest
    result = run_discover(project)
    assert result["errors"] == 0 and set(result["systems"]) == SYSTEMS
    assert [x["message"] for x in problems(result, "UNEXPECTED", "warn")] and "p2/test/extra_bm25.run" in problems(result, "UNEXPECTED", "warn")[0]["message"]
    assert "p3/dense/late.val.run" in problems(result, "STALE_MANIFEST", "warn")[0]["message"]


def test_manifest_counts_that_disagree_with_the_file_are_stale(project):
    write_run(project["runs"] / "p3" / "dense" / "minilm_abstract.val.run", VAL[:2], "dense_minilm_abstract")     # manifest was written for 4 queries
    result = run_discover(project)
    assert any("minilm_abstract.val.run" in x["message"] and "manifest says 4" in x["message"] for x in problems(result, "STALE_MANIFEST", "warn"))


def test_malformed_rows_are_a_format_error(project):
    path = project["runs"] / "p3" / "dense" / "minilm_abstract.val.run"
    path.write_text(path.read_text() + "v1 Q0 A 4\n")                                          # four fields
    result = run_discover(project)
    assert problems(result, "BAD_FORMAT")[0]["system"] == "p3_minilm_abstract" and "6 fields" in problems(result, "BAD_FORMAT")[0]["message"]


# ---- selection metadata ------------------------------------------------------------------------------------------
def rewrite_selection(project, **changes):
    (project["runs"] / "p3" / "hybrid" / "selection.json").write_text(json.dumps({**project["selection"], **changes}))


def test_hybrid_tag_must_match_the_frozen_k(project):
    rewrite_selection(project, rrf_k=60)
    bad = problems(run_discover(project), "TAG_MISMATCH")                                    # one per file: the validation and the test run
    assert len(bad) == 2 and {b["system"] for b in bad} == {"p3_hybrid_rrf"} and all("hybrid_rrf_k60" in b["message"] for b in bad)


def test_hybrid_dense_and_bm25_choices_must_exist_and_agree(project):
    rewrite_selection(project, dense_system="specter_expanded")
    assert "not in the manifest" in problems(run_discover(project), "SELECTION_MISMATCH")[0]["message"]
    rewrite_selection(project, bm25_system="some-other-config")
    assert "p2/fusion/selection.txt" in problems(run_discover(project), "SELECTION_MISMATCH")[0]["message"]
    rewrite_selection(project, dense_validation_mrr10={"minilm_abstract": 0.1, "minilm_contexts": 0.2, "specter_abstract": 0.3})
    assert "no validation run" in problems(run_discover(project), "SELECTION_MISMATCH")[0]["message"]


def test_missing_selection_json_and_best_json_exclude_the_systems_that_need_them(project):
    (project["runs"] / "p3" / "hybrid" / "selection.json").unlink()
    result = run_discover(project)
    assert problems(result, "MISSING_METADATA")[0]["system"] == "p3_hybrid_rrf" and "p3_hybrid_rrf" in result["excluded"]
    assert "p3_minilm_abstract" in result["systems"]
    (project["runs"] / "p2" / "best.json").unlink()
    result = run_discover(project)
    assert [s for s in result["systems"] if s.startswith("p2_")] == ["p2_bm25_fusion"]        # no best.json: no per-configuration pairing; the fusion
    assert "p2" in result["excluded"]                                                         # pair keeps its own metadata (selection.txt) and survives
    assert {"p2/test/abstract_bm25.run", "p2/test/context_bm25.run"} <= {x["message"].split()[0] for x in problems(result, "UNEXPECTED", "warn")}


def test_a_p2_fusion_pair_with_a_new_configuration_is_listed_as_its_own_system(project):
    other = "fielded-w0-plain-all-bm25f-0"
    write_run(project["runs"] / "p2" / "fusion" / "bm25.validation.run", VAL, other)
    write_run(project["runs"] / "p2" / "fusion" / "bm25.test.run", TEST, other)
    (project["runs"] / "p2" / "fusion" / "selection.txt").write_text(other + "\n")
    rewrite_selection(project, bm25_system=other)
    result = run_discover(project)
    assert "p2_bm25_fusion" in result["systems"] and result["errors"] == 0


# ---- the command line ----------------------------------------------------------------------------------------------
def test_cli_writes_runs_json_and_report_and_never_overwrites(project, tmp_path, capsys):
    out, report = tmp_path / "out" / "runs.json", tmp_path / "out" / "report.json"
    out.parent.mkdir()
    args = ["--manifest", str(project["manifest"]), "--processed", str(project["processed"]), "--out", str(out), "--report", str(report)]
    assert DR.main(args) == 0
    printed = capsys.readouterr().out
    assert "5 system(s) paired and verified, 0 excluded, 0 error(s)" in printed and "ok  p3_hybrid_rrf" in printed
    assert set(json.loads(out.read_text())["systems"]) == SYSTEMS and json.loads(report.read_text())["errors"] == 0
    assert DR.main(args) == 2 and "already exists" in capsys.readouterr().err


def test_cli_with_errors_writes_nothing_unless_partial_is_allowed(project, tmp_path, capsys):
    (project["runs"] / "p3" / "dense" / "minilm_abstract.test.run").unlink()
    write_manifest(project["runs"])
    out = tmp_path / "runs.json"
    args = ["--manifest", str(project["manifest"]), "--out", str(out)]
    assert DR.main(args) == 2 and not out.exists()
    captured = capsys.readouterr()
    assert "[ERROR] UNPAIRED" in captured.out and "NOT written" in captured.err and "--  p3_minilm_abstract" in captured.out
    assert DR.main(args + ["--allow-partial"]) == 2                                          # still a failing exit code, but the file exists
    saved = json.loads(out.read_text())
    assert "p3_minilm_abstract" not in saved["systems"] and saved["_discovery"]["complete"] is False and "p3_minilm_abstract" in saved["_discovery"]["excluded"]


def test_cli_manifest_problems_exit_2(tmp_path, capsys):
    assert DR.main(["--manifest", str(tmp_path / "none.csv"), "--out", str(tmp_path / "o.json")]) == 2
    assert "manifest not found" in capsys.readouterr().err
    bad = tmp_path / "MANIFEST.csv"
    bad.write_text("member,system\nP2,x\n")
    assert DR.main(["--manifest", str(bad), "--out", str(tmp_path / "o.json")]) == 2
    assert "lacks column(s)" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        DR.main(["--out", str(tmp_path / "o.json")])                                          # --manifest is required


# ---- a hand-written runs.json is still supported --------------------------------------------------------------------
def test_check_config_accepts_a_consistent_manual_file_and_rejects_bad_pairs(project, tmp_path, capsys):
    runs = project["runs"]
    good = {"systems": {"a": {"val": str(runs / "p3/dense/minilm_abstract.val.run"), "test": str(runs / "p3/dense/minilm_abstract.test.run")}}}
    path = tmp_path / "manual.json"
    path.write_text(json.dumps(good))
    assert DR.main(["--check-config", str(path), "--processed", str(project["processed"])]) == 0
    assert "checked 1 system(s): 0 error(s)" in capsys.readouterr().out
    same = {"systems": {"same": {"val": good["systems"]["a"]["val"], "test": good["systems"]["a"]["val"]},
                        "swapped": {"val": good["systems"]["a"]["test"], "test": good["systems"]["a"]["val"]},
                        "gone": {"val": str(runs / "nope.run"), "test": good["systems"]["a"]["test"]}}}
    path.write_text(json.dumps(same))
    result = DR.check_config(path, processed=project["processed"])
    by_system = {(p["system"], p["code"]) for p in result["problems"]}
    assert ("same", "SPLIT_OVERLAP") in by_system and ("swapped", "WRONG_SPLIT") in by_system and ("gone", "MISSING_FILE") in by_system
    assert DR.main(["--check-config", str(path)]) == 2
    assert PurePosixPath("x").name == "x"                                                     # (keeps the import used)
    with pytest.raises(DR.DiscoveryError):
        DR.check_config(tmp_path / "missing.json")
