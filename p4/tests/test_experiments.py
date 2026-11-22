"""Tests for p4.experiments.run_all. DEMO and SYNTHETIC data only: this orchestrator is not run on real data in preparation."""
import json
from pathlib import Path

import pandas as pd
import pytest

from p4.experiments import run_all as RA
from p4.tests import graph_fixtures as GF

REPO = Path(__file__).resolve().parent.parent.parent
FAST = ["--n-boot", "100", "--min-group-size", "50", "--analyses", "popularity", "coldstart", "errors"]


@pytest.fixture
def demo(tmp_path):
    """A DEMO project: processed folder (leak scenario), run and qrels files, and a configuration."""
    processed = GF.leak_scenario(tmp_path / "processed")
    runs = tmp_path / "runs"
    runs.mkdir()
    for split in ("val", "test"):
        run, qrels = GF.leak_run_and_qrels(split)
        GF.write_run_file(runs / f"base.{split}.run", run, "base")
        GF.write_qrels_file(runs / f"qrels.{split}.txt", qrels)
    config = {"systems": {"base": {"val": "runs/base.val.run", "test": "runs/base.test.run"}},
              "qrels": {"val": "runs/qrels.val.txt", "test": "runs/qrels.test.txt"}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return {"processed": processed, "config": config_path, "root": tmp_path, "runs": runs}


def args_for(demo, out, *extra, label="demo"):
    return ["--data-label", label, "--processed", str(demo["processed"]), "--config", str(demo["config"]), "--out", str(out), *FAST, *extra]


def test_full_demo_run_executes_every_step_in_order_and_writes_the_manifest(demo, capsys):
    out = demo["root"] / "out"
    assert RA.main(args_for(demo, out)) == 0
    manifest = json.loads((out / "EXPERIMENT_MANIFEST.json").read_text())
    names = [s["step"] for s in manifest["steps"]]
    assert names == ["leakage gate", "evaluation graph", "evaluation (test)", "fusion tune (base)", "fusion apply (base)",
                     "evaluation of base and fused systems (test)", "analysis popularity", "analysis coldstart", "analysis errors"]
    assert manifest["data_label"] == "demo" and manifest["confirmed_integrated_project"] is False and manifest["steps"][0]["passed"]
    assert manifest["protocol"] == RA.eval_graph.PROTOCOL and manifest["baseline"] == "base"
    for path in ("leakage_audit.json", "eval_graph/priors.parquet", "eval_graph/MANIFEST.json", "eval_test/report.md", "fusion/base/fusion_best.json",
                 "fusion/base/fused_test.run", "eval_fusion_test/summary.csv", "analysis/popularity_report.md", "analysis/errors_report.md"):
        assert (out / path).exists(), path
    assert "completed 9 steps" in capsys.readouterr().out


def test_the_safe_prior_gives_no_gain_in_the_demo_scenario(demo):
    out = demo["root"] / "out"
    assert RA.main(args_for(demo, out)) == 0
    summary = pd.read_csv(out / "eval_fusion_test" / "summary.csv").set_index("system")
    assert list(summary.index) == ["base", "base+authority"]
    assert summary.loc["base", "MRR@10"] == pytest.approx(0.25)
    assert summary.loc["base+authority", "MRR@10"] == pytest.approx(0.25)                    # nothing to learn from a leakage-safe graph
    best = json.loads((out / "fusion/base/fusion_best.json").read_text())
    assert best["improves_on_validation"] is False and best["prior_hold_out_splits"] == ["val", "test"]


def test_analyses_use_the_leakage_safe_counts_and_every_output_carries_the_label(demo):
    out = demo["root"] / "out"
    assert RA.main(args_for(demo, out)) == 0
    report = (out / "analysis" / "popularity_report.md").read_text(encoding="utf-8")
    assert "citation_count_eval from" in report and "Data label: DEMO" in report
    assert all(pd.read_csv(f)["data_label"].eq("demo").all() for f in (out / "analysis").glob("*.csv"))


# ---- real data is refused ------------------------------------------------------------------------------------
def test_real_data_is_refused_without_the_integration_confirmation(demo, capsys):
    out = demo["root"] / "out"
    assert RA.main(args_for(demo, out, label="real")) == 2
    assert "REAL data refused" in capsys.readouterr().err and not out.exists()                 # nothing created at all


def test_the_confirmation_flag_is_recorded_when_given(demo):
    out = demo["root"] / "out"
    assert RA.main(args_for(demo, out, "--confirm-integrated-project", label="synthetic")) == 0
    assert json.loads((out / "EXPERIMENT_MANIFEST.json").read_text())["confirmed_integrated_project"] is True


# ---- the leakage gate stops the run before any result ----------------------------------------------------------
def test_a_context_file_containing_query_contexts_stops_the_run(demo, capsys):
    bad = demo["root"] / "contexts.jsonl"
    bad.write_text(json.dumps({"context_id": "ctxv01", "cited_paper_id": "Tv01", "sentence": "x"}) + chr(10))
    config = json.loads(demo["config"].read_text())
    config["context_files"] = ["contexts.jsonl"]
    demo["config"].write_text(json.dumps(config))
    out = demo["root"] / "out"
    assert RA.main(args_for(demo, out)) == 2
    assert "LEAKAGE GATE FAILED" in capsys.readouterr().err
    assert (out / "leakage_audit.json").exists() and not (out / "eval_test").exists() and not (out / "eval_graph").exists()


def test_splits_sharing_a_citing_paper_stop_the_run(demo, capsys):
    queries = pd.read_parquet(demo["processed"] / "queries.parquet")
    queries.loc[queries.query_id == "t01", "citing_paper_id"] = "Cv01"                          # one citing paper now in val and test
    queries.to_parquet(demo["processed"] / "queries.parquet", index=False)
    assert RA.main(args_for(demo, demo["root"] / "out")) == 2
    assert "more than one split" in capsys.readouterr().err


def test_a_clean_context_file_passes_the_gate(demo):
    ok = demo["root"] / "contexts.jsonl"
    ok.write_text(json.dumps({"context_id": "z1", "cited_paper_id": "D01", "sentence": "an unrelated sentence"}) + chr(10))
    config = json.loads(demo["config"].read_text())
    config["context_files"] = ["contexts.jsonl"]
    demo["config"].write_text(json.dumps(config))
    out = demo["root"] / "out"
    assert RA.main(args_for(demo, out)) == 0
    assert json.loads((out / "leakage_audit.json").read_text())["context_files"][0]["ok"] is True


# ---- configuration and step errors ---------------------------------------------------------------------------------
@pytest.mark.parametrize("mutate, message", [
    (lambda c: c.pop("systems"), "non-empty 'systems'"),
    (lambda c: c["systems"].update(bad={"val": "x"}), "needs 'val' and 'test'"),
    (lambda c: c.update(baseline="nobody"), "baseline 'nobody'"),
    (lambda c: c["systems"].update(ghost={"val": "runs/none.run", "test": "runs/none2.run"}), "input file(s) not found")])
def test_configuration_errors(demo, capsys, mutate, message):
    config = json.loads(demo["config"].read_text())
    mutate(config)
    demo["config"].write_text(json.dumps(config))
    assert RA.main(args_for(demo, demo["root"] / "out")) == 2
    assert message in capsys.readouterr().err


def test_missing_and_invalid_inputs(demo, capsys, tmp_path):
    assert RA.main(["--data-label", "demo", "--processed", str(demo["processed"]), "--config", str(tmp_path / "nope.json"),
                    "--out", str(tmp_path / "o1"), *FAST]) == 2
    (tmp_path / "bad.json").write_text("{not json")
    assert RA.main(["--data-label", "demo", "--processed", str(demo["processed"]), "--config", str(tmp_path / "bad.json"),
                    "--out", str(tmp_path / "o2"), *FAST]) == 2
    assert RA.main(["--data-label", "demo", "--processed", str(tmp_path / "nope"), "--config", str(demo["config"]),
                    "--out", str(tmp_path / "o3"), *FAST]) == 2
    err = capsys.readouterr().err
    assert "configuration file not found" in err and "not valid JSON" in err and "processed folder not found" in err


def test_a_non_empty_output_folder_is_refused(demo, capsys):
    out = demo["root"] / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    assert RA.main(args_for(demo, out)) == 2 and "is not empty" in capsys.readouterr().err
    assert (out / "keep.txt").read_text() == "mine"


def test_a_failing_step_stops_the_run_and_names_the_step(demo, capsys):
    other = demo["runs"] / "wrong_split.run"
    GF.write_run_file(other, {f"x{i}": [("D01", 1.0)] for i in range(5)})                        # queries that are not in the test qrels
    config = json.loads(demo["config"].read_text())
    config["systems"]["base"]["test"] = "runs/wrong_split.run"
    demo["config"].write_text(json.dumps(config))
    out = demo["root"] / "out"
    assert RA.main(args_for(demo, out)) == 2
    err = capsys.readouterr().err
    assert "wrong split" in err and "step 'evaluation (test)' failed" in err
    assert not (out / "fusion").exists() and not (out / "EXPERIMENT_MANIFEST.json").exists()     # nothing after the failing step ran


def test_function_analysis_is_skipped_not_invented_when_no_labels_are_configured(demo):
    out = demo["root"] / "out"
    assert RA.main(["--data-label", "demo", "--processed", str(demo["processed"]), "--config", str(demo["config"]), "--out", str(out),
                    "--n-boot", "100", "--min-group-size", "50", "--analyses", "function", "errors"]) == 0
    steps = {s["step"]: s for s in json.loads((out / "EXPERIMENT_MANIFEST.json").read_text())["steps"]}
    assert "no 'functions' file" in steps["analysis function"]["skipped"] and not (out / "analysis" / "function_report.md").exists()


# ---- SYNTHETIC end to end on P3's committed synthetic runs ----------------------------------------------------------
P3 = REPO / "p3" / "outputs"


@pytest.mark.skipif(not (P3 / "hybrid_test.run").exists(), reason="P3's committed synthetic outputs are not present")
def test_synthetic_end_to_end_on_the_committed_p2_p3_runs(tmp_path):
    from p4.analysis.synthetic_demo import write_synthetic_processed
    processed = tmp_path / "processed"
    write_synthetic_processed(processed)
    config = {"systems": {"bm25": {"val": str(P3 / "p2_quick/fusion/bm25.validation.run"), "test": str(P3 / "p2_quick/fusion/bm25.test.run")},
                          "hybrid": {"val": str(P3 / "hybrid_validation.run"), "test": str(P3 / "hybrid_test.run")}},
              "baseline": "bm25", "efficiency": {"efficiency_csv": str(P3 / "p2_quick/efficiency.csv"), "tuning_csv": str(P3 / "p2_quick/tuning.csv")}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    out = tmp_path / "out"
    assert RA.main(["--data-label", "synthetic", "--processed", str(processed), "--config", str(tmp_path / "config.json"), "--out", str(out),
                    "--n-boot", "100", "--min-group-size", "10", "--analyses", "popularity", "errors"]) == 0
    manifest = json.loads((out / "EXPERIMENT_MANIFEST.json").read_text())
    assert manifest["data_label"] == "synthetic" and all(s.get("exit_code", 0) == 0 for s in manifest["steps"])
    summary = pd.read_csv(out / "eval_fusion_test" / "summary.csv")
    assert list(summary["system"]) == ["bm25", "hybrid", "bm25+authority", "hybrid+authority"] and (summary["n_queries"] == 32).all()
    assert "SYNTHETIC" in (out / "analysis" / "errors_report.md").read_text(encoding="utf-8")
    assert (out / "eval_test" / "efficiency_index.csv").exists()


# ---- fusion_systems / analysis_systems / --fusion-mode (DEMO data) ---------------------------------------------------
def two_system_config(demo, **extra):
    config = {"systems": {"base": {"val": "runs/base.val.run", "test": "runs/base.test.run"},
                          "other": {"val": "runs/base.val.run", "test": "runs/base.test.run"}},
              "qrels": {"val": "runs/qrels.val.txt", "test": "runs/qrels.test.txt"}, **extra}
    demo["config"].write_text(json.dumps(config))


def test_fusion_and_analysis_can_be_limited_to_chosen_systems(demo):
    two_system_config(demo, fusion_systems=["other"], analysis_systems=["base", "other+authority"])
    out = demo["root"] / "out"
    assert RA.main(args_for(demo, out)) == 0
    assert (out / "fusion/other/fused_test.run").exists() and not (out / "fusion/base").exists()          # only 'other' was fused
    summary = pd.read_csv(out / "eval_fusion_test" / "summary.csv")
    assert list(summary["system"]) == ["base", "other", "other+authority"]                              # evaluation still covers every system
    manifest = json.loads((out / "EXPERIMENT_MANIFEST.json").read_text())
    assert manifest["settings"]["fusion_systems"] == ["other"] and manifest["settings"]["analysis_systems"] == ["base", "other+authority"]
    text = (out / "analysis" / "popularity_report.md").read_text()
    assert "other+authority" in text and "| other |" not in text                                           # analyses were limited


def test_unknown_system_names_in_the_limits_are_configuration_errors(demo, capsys):
    two_system_config(demo, fusion_systems=["nope"])
    assert RA.main(args_for(demo, demo["root"] / "o1")) == 2 and "fusion_systems ['nope']" in capsys.readouterr().err
    two_system_config(demo, analysis_systems=["nope"])
    assert RA.main(args_for(demo, demo["root"] / "o2")) == 2 and "analysis_systems ['nope']" in capsys.readouterr().err


def test_fusion_mode_raw_is_passed_to_the_tuning_and_recorded(demo):
    out = demo["root"] / "out"
    assert RA.main(args_for(demo, out, "--fusion-mode", "raw")) == 0
    best = json.loads((out / "fusion/base/fusion_best.json").read_text())
    assert (best["mode"], best["score_norm"], best["prior_transform"]) == ("raw", "none", "log_raw")
    assert json.loads((out / "EXPERIMENT_MANIFEST.json").read_text())["settings"]["fusion_mode"] == "raw"
    out2 = demo["root"] / "out2"
    assert RA.main(args_for(demo, out2, "--fusion-mode", "raw", "--fusion-lambdas", "0,0.5")) == 0
    assert list(pd.read_csv(out2 / "fusion/base/fusion_tuning.csv")["lambda"].unique()) == [0.0, 0.5]
