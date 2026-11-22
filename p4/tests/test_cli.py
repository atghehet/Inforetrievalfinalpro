"""Tests for the command-line runner p4.eval.cli (and the scripts/run_eval.py launcher)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from p4.eval.cli import main

REPO = Path(__file__).resolve().parent.parent.parent
N = 30     # judged queries in the test data (>= 25 keeps scipy's Wilcoxon on its fast path; tiny tied samples take ~0.5 s per test)


def write_run(path, ranks):
    """TREC run where query i finds its relevant paper D<i> at ranks[i] (None = never found)."""
    lines = []
    for i in range(1, N + 1):
        docs = [f"x{j}" for j in range(1, 4)]
        if ranks(i) is not None:
            docs.insert(ranks(i) - 1, f"D{i}")
        lines += [f"q{i} Q0 {d} {r} {10 - r}.0 sys" for r, d in enumerate(docs, start=1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def files(tmp_path):
    qrels = tmp_path / "test.qrels"
    qrels.write_text("".join(f"q{i} 0 D{i} 1\n" for i in range(1, N + 1)))
    return {"qrels": qrels,
            "good": write_run(tmp_path / "good.run", lambda i: 1),            # always rank 1
            "mid": write_run(tmp_path / "mid.run", lambda i: 2 if i % 2 else 1),
            "bad": write_run(tmp_path / "bad.run", lambda i: None),           # never found
            "out": tmp_path / "out"}


def run_cli(files, *extra, runs=("good", "mid", "bad")):
    argv = ["--qrels", str(files["qrels"]), "--out", str(files["out"]), "--n-boot", "200", "--quiet"]
    for name in runs:
        argv += ["--run", f"{name}={files[name]}"]
    return main(argv + list(extra))


def test_multiple_runs_produce_all_four_output_files(files, capsys):
    assert run_cli(files) == 0
    out = files["out"]
    assert sorted(p.name for p in out.iterdir()) == ["comparisons.csv", "per_query.csv", "report.md", "summary.csv"]
    summary = pd.read_csv(out / "summary.csv").set_index("system")
    assert list(summary.index) == ["good", "mid", "bad"]                       # order given on the command line
    assert summary.loc["good", "MRR@10"] == 1.0 and summary.loc["mid", "MRR@10"] == pytest.approx(0.75)
    assert summary.loc["bad", "n_no_results"] == 0 and summary.loc["bad", "MRR@10"] == 0.0   # present but never relevant
    assert {"Recall@1", "Recall@100", "nDCG@50", "MAP"} <= set(summary.columns)               # default cutoffs
    per_query = pd.read_csv(out / "per_query.csv")
    assert len(per_query) == 3 * N and set(per_query["system"]) == {"good", "mid", "bad"}
    comparisons = pd.read_csv(out / "comparisons.csv")
    assert len(comparisons) == 2 * 3                                           # 2 systems vs baseline x 3 default metrics
    assert set(comparisons["baseline"]) == {"good"}                           # first --run is the default baseline
    assert set(comparisons["metric"]) == {"rr@10", "ndcg@10", "recall@10"}
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "# Evaluation report" in report and "### MRR@10" in report
    assert capsys.readouterr().out.strip().startswith("Wrote summary.csv")    # --quiet: no table, just the file list


def test_summary_table_is_printed_unless_quiet(files, capsys):
    argv = ["--qrels", str(files["qrels"]), "--out", str(files["out"]), "--run", f"good={files['good']}",
            "--run", f"mid={files['mid']}", "--n-boot", "200"]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "MRR@10" in printed and "good" in printed and "mid" in printed


def test_system_names_default_to_file_names_and_baseline_can_be_chosen(files):
    argv = ["--qrels", str(files["qrels"]), "--out", str(files["out"]), "--n-boot", "200", "--quiet",
            "--run", str(files["good"]), "--run", f"weak={files['bad']}", "--baseline", "weak"]
    assert main(argv) == 0
    comparisons = pd.read_csv(files["out"] / "comparisons.csv")
    assert set(comparisons["system"]) == {"good"} and set(comparisons["baseline"]) == {"weak"}   # "good" = file stem
    assert (comparisons["mean_diff"] > 0).all()


def test_runs_glob_adds_every_matching_file_named_by_its_stem(files, tmp_path):
    argv = ["--qrels", str(files["qrels"]), "--out", str(files["out"]), "--n-boot", "200", "--quiet",
            "--runs-glob", str(tmp_path / "*.run")]
    assert main(argv) == 0
    assert list(pd.read_csv(files["out"] / "summary.csv")["system"]) == ["bad", "good", "mid"]    # sorted file order


def test_custom_cutoffs_and_test_metrics(files):
    assert run_cli(files, "--cutoffs", "1,3", "--test-metrics", "MRR@3,MAP,Recall@1") == 0
    summary = pd.read_csv(files["out"] / "summary.csv")
    assert {"Recall@1", "Recall@3", "MRR@1", "MRR@3", "nDCG@3"} <= set(summary.columns)
    assert "Recall@10" not in summary.columns
    metrics = set(pd.read_csv(files["out"] / "comparisons.csv")["metric"])
    assert "rr@3" in metrics and "recall@1" in metrics and "ap" in metrics


def test_headline_cutoff_falls_back_to_the_largest_when_10_is_absent(files):
    assert run_cli(files, "--cutoffs", "1,5") == 0
    assert set(pd.read_csv(files["out"] / "comparisons.csv")["metric"]) == {"rr@5", "ndcg@5", "recall@5"}


def test_same_seed_gives_identical_comparison_files(files, tmp_path):
    assert run_cli(files) == 0
    files2 = {**files, "out": tmp_path / "out2"}
    assert run_cli(files2) == 0
    assert (files["out"] / "comparisons.csv").read_bytes() == (files2["out"] / "comparisons.csv").read_bytes()
    assert (files["out"] / "summary.csv").read_bytes() == (files2["out"] / "summary.csv").read_bytes()


def test_a_single_run_is_evaluated_without_comparisons(files):
    assert run_cli(files, runs=("good",)) == 0
    assert pd.read_csv(files["out"] / "comparisons.csv").empty
    assert "Not computed (needs at least two systems)" in (files["out"] / "report.md").read_text(encoding="utf-8")


def test_note_is_written_to_the_report(files):
    assert run_cli(files, "--note", "SYNTHETIC sample - not real results") == 0
    assert "SYNTHETIC sample - not real results" in (files["out"] / "report.md").read_text(encoding="utf-8")


# ---- error handling: exit code 2 and a one-line message, never a traceback ----------------------
def expect_error(files, capsys, text, *extra, **kw):
    assert run_cli(files, *extra, **kw) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and text in err and "Traceback" not in err
    return err


def test_missing_run_file(files, capsys):
    files["ghost"] = files["out"].parent / "ghost.run"
    expect_error(files, capsys, "run file not found", runs=("good", "ghost"))


def test_missing_qrels_file(files, capsys):
    files["qrels"].unlink()
    expect_error(files, capsys, "qrels file not found")


def test_malformed_run_file_names_file_and_line(files, capsys):
    files["good"].write_text("q1 Q0 D1 one 1.0 sys\n")
    expect_error(files, capsys, "good.run, line 1")


def test_malformed_qrels_file(files, capsys):
    files["qrels"].write_text("q1 0 D1\n")
    expect_error(files, capsys, "expected 4 fields")


def test_run_for_the_wrong_split_is_rejected_unless_coverage_check_is_disabled(files, capsys, tmp_path):
    other = tmp_path / "val.run"
    other.write_text("".join(f"v{i} Q0 D{i} 1 1.0 sys\n" for i in range(1, 6)))     # validation query IDs
    files["val"] = other
    expect_error(files, capsys, "wrong split", runs=("good", "val"))
    files["out"] = tmp_path / "out_forced"
    assert run_cli(files, "--min-coverage", "0", runs=("good", "val")) == 0
    summary = pd.read_csv(files["out"] / "summary.csv").set_index("system")
    assert summary.loc["val", "MRR@10"] == 0.0 and summary.loc["val", "n_no_results"] == N


def test_duplicate_system_names_are_rejected(files, capsys, tmp_path):
    copy = write_run(tmp_path / "copy.run", lambda i: 1)
    argv = ["--qrels", str(files["qrels"]), "--out", str(files["out"]),
            "--run", f"same={files['good']}", "--run", f"same={copy}"]
    assert main(argv) == 2
    assert "two run files are called 'same'" in capsys.readouterr().err


def test_no_runs_given(files, capsys):
    assert main(["--qrels", str(files["qrels"]), "--out", str(files["out"])]) == 2
    assert "no run files given" in capsys.readouterr().err


def test_glob_matching_nothing(files, capsys, tmp_path):
    assert main(["--qrels", str(files["qrels"]), "--out", str(files["out"]),
                 "--runs-glob", str(tmp_path / "*.nothing")]) == 2
    assert "matched no files" in capsys.readouterr().err


@pytest.mark.parametrize("extra, text", [
    (["--cutoffs", "ten"], "--cutoffs must be whole numbers"),
    (["--cutoffs", "0,5"], "--cutoffs must be whole numbers >= 1"),
    (["--test-metrics", "MRR@10", "--cutoffs", "5"], "not in --cutoffs"),
    (["--test-metrics", "banana"], "unknown metric name"),
    (["--baseline", "nobody"], "is not one of the systems"),
    (["--tuning", "t.csv"], "--tuning needs --efficiency"),
    (["--alpha", "2"], "alpha"),
    (["--n-boot", "5"], "at least 100"),
    (["--min-coverage", "3"], "min_coverage"),
])
def test_bad_options(files, capsys, extra, text):
    expect_error(files, capsys, text, *extra)


def test_existing_outputs_are_not_overwritten_without_permission(files, capsys):
    assert run_cli(files) == 0
    expect_error(files, capsys, "already contains")
    (files["out"] / "report.md").write_text("old")
    assert run_cli(files, "--overwrite") == 0
    assert "# Evaluation report" in (files["out"] / "report.md").read_text(encoding="utf-8")


def test_out_path_that_is_a_file_is_rejected(files, capsys, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    files["out"] = blocker
    expect_error(files, capsys, "is not a folder")


def test_efficiency_files_are_written_and_reported(files, tmp_path):
    eff, tun = tmp_path / "efficiency.csv", tmp_path / "tuning.csv"
    eff.write_text("index,build_seconds,bytes\nctx,1.5,1048576\n")
    tun.write_text("run,variant,model,parameters,validation_mrr_at_10,median_ms,mean_ms\nctx-bm25-0,context,bm25,{},0.3,2.0,2.5\n")
    assert run_cli(files, "--efficiency", str(eff), "--tuning", str(tun)) == 0
    assert (files["out"] / "efficiency_index.csv").exists() and (files["out"] / "efficiency_runs.csv").exists()
    assert pd.read_csv(files["out"] / "efficiency_runs.csv").loc[0, "median_ms"] == 2.0
    assert "Index build cost" in (files["out"] / "report.md").read_text(encoding="utf-8")


def test_bad_efficiency_csv_is_a_clean_error(files, capsys, tmp_path):
    eff = tmp_path / "efficiency.csv"
    eff.write_text("index,seconds\nctx,1.0\n")
    expect_error(files, capsys, "missing column", "--efficiency", str(eff))
    expect_error({**files, "out": tmp_path / "o2"}, capsys, "efficiency file not found", "--efficiency", str(tmp_path / "none.csv"))


def test_scripts_launcher_runs_the_same_program(files):
    cmd = [sys.executable, "-B", str(REPO / "scripts" / "run_eval.py"), "--qrels", str(files["qrels"]),
           "--out", str(files["out"]), "--run", f"good={files['good']}", "--run", f"mid={files['mid']}",
           "--n-boot", "200", "--quiet"]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=files["out"].parent)    # run from elsewhere: no path assumptions
    assert done.returncode == 0, done.stderr
    assert (files["out"] / "report.md").exists()
    bad = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "run_eval.py"), "--help"], capture_output=True, text=True)
    assert bad.returncode == 0 and "--qrels" in bad.stdout


# ---- --name-from-tag ----------------------------------------------------------------------------------------------
def tagged_run(path, tag):
    path.write_text("".join(f"q{i} Q0 D{i} 1 1.0 {tag}" + chr(10) for i in range(1, N + 1)))
    return path


def test_name_from_tag_uses_the_sixth_column_for_globs_and_bare_paths(files, tmp_path):
    folder = tmp_path / "p2_like"
    folder.mkdir()
    tagged_run(folder / "context_bm25.run", "context-w0-plain-method-bm25-0")       # P2's frozen file name hides the winning configuration
    tagged_run(folder / "expanded_bm25.run", "expanded-w0-porter-all-bm25-1")
    argv = ["--qrels", str(files["qrels"]), "--out", str(files["out"]), "--n-boot", "200", "--quiet", "--name-from-tag",
            "--runs-glob", str(folder / "*.run"), "--run", f"named={files['good']}"]
    assert main(argv) == 0
    assert list(pd.read_csv(files["out"] / "summary.csv")["system"]) == ["named", "context-w0-plain-method-bm25-0", "expanded-w0-porter-all-bm25-1"]   # --run entries first, then glob matches


def test_name_from_tag_errors_are_clear(files, tmp_path, capsys):
    empty, short = tmp_path / "empty.run", tmp_path / "short.run"
    empty.write_text("")
    short.write_text("q1 Q0 D1 1 1.0 sys extra" + chr(10))
    for bad, text in ((empty, "run file is empty"), (short, "does not have 6 fields")):
        argv = ["--qrels", str(files["qrels"]), "--out", str(tmp_path / "o"), "--quiet", "--name-from-tag", "--run", str(bad)]
        assert main(argv) == 2 and text in capsys.readouterr().err
    a, b = tagged_run(tmp_path / "a.run", "same-tag"), tagged_run(tmp_path / "b.run", "same-tag")
    argv = ["--qrels", str(files["qrels"]), "--out", str(tmp_path / "o2"), "--quiet", "--name-from-tag", "--run", str(a), "--run", str(b)]
    assert main(argv) == 2 and "two run files are called 'same-tag'" in capsys.readouterr().err
    plain = ["--qrels", str(files["qrels"]), "--out", str(tmp_path / "o3"), "--quiet", "--run", str(a)]      # without the flag: file stem
    assert main(plain) == 0 and list(pd.read_csv(tmp_path / "o3" / "summary.csv")["system"]) == ["a"]
