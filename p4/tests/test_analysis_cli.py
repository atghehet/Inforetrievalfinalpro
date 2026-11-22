"""Tests for p4.analysis.cli, the synthetic-input generator, and compatibility with the committed P2/P3 outputs."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from p4.analysis.cli import main
from p4.tests import analysis_fixtures as F

REPO = Path(__file__).resolve().parent.parent.parent
FAST = ["--min-group-size", "50", "--n-boot", "200", "--quiet", "--analysis-date", "2026-01-02"]   # no group reaches 50: no paired tests


@pytest.fixture
def files(tmp_path):
    paths = F.write_files(tmp_path)
    paths["out"] = tmp_path / "out"
    return paths


def argv(files, analysis, *extra, label="synthetic", with_functions=False, with_index=False):
    args = [analysis, "--data-label", label, "--qrels", str(files["qrels"]), "--run", f"good={files['run_good']}",
            "--run", f"weak={files['run_weak']}", "--queries", str(files["queries"]), "--papers", str(files["papers"]),
            "--out", str(files["out"]), "--baseline", "good"]
    if with_functions:
        args += ["--functions", str(files["functions"])]
    if with_index:
        args += ["--index-contexts", str(files["index_contexts"])]
    return args + list(extra)


# ---- the data label is mandatory and propagates -----------------------------------------------------
def test_data_label_is_required_and_validated(files, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["errors", "--qrels", str(files["qrels"]), "--run", f"g={files['run_good']}", "--out", str(files["out"])])
    assert exit_info.value.code == 2 and "--data-label" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(argv(files, "errors", label="fake"))
    assert not files["out"].exists()                                               # nothing was written


@pytest.mark.parametrize("label", ["real", "synthetic", "demo"])
def test_label_appears_in_every_csv_and_the_report(files, label):
    assert main(argv(files, "popularity", *FAST, label=label)) == 0
    csvs = sorted(files["out"].glob("*.csv"))
    assert len(csvs) == 5 and all(pd.read_csv(f)["data_label"].eq(label).all() for f in csvs)
    report = (files["out"] / "popularity_report.md").read_text(encoding="utf-8")
    assert f"data label:** {label}" in report and ("NOT real findings" in report) == (label != "real")


# ---- the five commands ---------------------------------------------------------------------------------
@pytest.mark.parametrize("analysis, expected", [
    ("popularity", {"popularity_retrieval", "popularity_query_side_groups"}),
    ("coldstart", {"coldstart_citation_groups", "coldstart_corpus_buckets", "coldstart_contexts_groups"}),
    ("year", {"year_groups", "year_trend", "year_by_popularity_crosstab"}),
    ("function", {"function_groups", "function_group_tests"}),
    ("errors", {"errors_summary", "errors_per_query", "errors_pairwise", "errors_hard_queries"}),
])
def test_each_command_writes_its_tables_and_report(files, capsys, analysis, expected):
    assert main(argv(files, analysis, *FAST, with_functions=True, with_index=True)) == 0
    written = {f.stem for f in files["out"].glob("*.csv")}
    assert expected <= written and (files["out"] / f"{analysis}_report.md").exists()
    assert f"Wrote {analysis}_report.md" in capsys.readouterr().out


def test_full_run_with_comparisons_matches_the_hand_calculation(files):
    assert main(argv(files, "popularity", "--min-group-size", "10", "--n-boot", "200", "--quiet",
                     "--analysis-date", "2026-01-02")) == 0
    comp = pd.read_csv(files["out"] / "popularity_query_side_comparisons.csv")
    row = comp[(comp.metric == "rr@10") & (comp.group == "3-10")].iloc[0]
    assert row["system"] == "weak" and row["baseline"] == "good" and row["mean_diff"] == pytest.approx(-0.75)
    assert row["n_queries"] == 20 and bool(row["significant_holm"])


def test_year_bins_and_min_confidence_options_are_applied(files):
    assert main(argv(files, "year", *FAST, "--year-bins", "2019")) == 0
    assert set(pd.read_csv(files["out"] / "year_groups.csv")["group"]) == {"<=2019", ">=2020"}
    files["out"] = files["out"].parent / "out2"
    assert main(argv(files, "function", *FAST, "--min-confidence", "0.7", with_functions=True)) == 0
    assert "unlabelled (low confidence)" in set(pd.read_csv(files["out"] / "function_groups.csv")["group"])


def test_runs_glob_cutoffs_and_k(files):
    args = ["errors", "--data-label", "demo", "--qrels", str(files["qrels"]), "--runs-glob", str(files["qrels"].parent / "*.run"),
            "--out", str(files["out"]), "--k", "5", "--cutoffs", "1,5,10", *FAST]
    assert main(args) == 0
    summary = pd.read_csv(files["out"] / "errors_summary.csv")
    assert list(summary["system"]) == ["good", "weak"] and set(summary["k"]) == {5}
    assert "n_rank 2-5" in summary.columns


def test_reruns_with_a_fixed_date_are_byte_identical(files, tmp_path):
    assert main(argv(files, "errors", *FAST)) == 0
    first = {f.name: f.read_bytes() for f in files["out"].iterdir()}
    files["out"] = tmp_path / "again"
    assert main(argv(files, "errors", *FAST)) == 0
    assert first == {f.name: f.read_bytes() for f in files["out"].iterdir()}


# ---- refusals and errors (exit 2, one line, no traceback) ------------------------------------------------
def fails(files, capsys, analysis, text, *extra, **kw):
    assert main(argv(files, analysis, *FAST, *extra, **kw)) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and text in err and "Traceback" not in err


def test_function_command_without_labels_refuses_instead_of_inventing_them(files, capsys):
    fails(files, capsys, "function", "needs --functions")
    assert not (files["out"] / "function_report.md").exists()


def test_papers_are_required_for_every_analysis_except_errors(files, capsys):
    args = ["popularity", "--data-label", "demo", "--qrels", str(files["qrels"]), "--run", f"g={files['run_good']}",
            "--out", str(files["out"]), *FAST]
    assert main(args) == 2 and "needs --papers" in capsys.readouterr().err
    assert main(["errors", *args[1:]]) == 0                                            # errors can run on the runs and qrels alone


def test_missing_and_malformed_inputs(files, capsys):
    files["queries"].unlink()
    fails(files, capsys, "popularity", "queries file not found")
    files["queries"].write_text("not,a,table\n")                                       # text inside a .parquet file
    fails(files, capsys, "popularity", "cannot read queries as a parquet file")
    files["queries"] = files["queries"].with_suffix(".csv")
    files["queries"].write_text("not,a,table\n")                                       # a CSV with the wrong columns
    fails(files, capsys, "popularity", "missing column")
    files["run_weak"].write_text("q01 Q0 T01 one 1.0 x\n")
    fails(files, capsys, "errors", "line 1")


def test_wrong_split_is_still_rejected(files, capsys, tmp_path):
    other = tmp_path / "val.run"
    other.write_text("".join(f"v{i} Q0 T01 1 1.0 s\n" for i in range(1, 6)))
    args = ["errors", "--data-label", "demo", "--qrels", str(files["qrels"]), "--run", f"val={other}",
            "--out", str(files["out"]), *FAST]
    assert main(args) == 2
    assert "wrong split" in capsys.readouterr().err


@pytest.mark.parametrize("extra, text", [
    (["--year-bins", "two,three"], "--year-bins must be whole years"),
    (["--k", "0"], "--k must be >= 1"),
    (["--cutoffs", "x"], "--cutoffs must be whole numbers"),
    (["--min-group-size", "0"], "min_group_size"),
    (["--baseline", "nobody"], "baseline 'nobody'"),
    (["--n-boot", "5"], "at least 100"),
])
def test_bad_options(files, capsys, extra, text):
    assert main(argv(files, "year", "--quiet", "--analysis-date", "2026-01-02", *extra)) == 2
    assert text in capsys.readouterr().err


def test_existing_report_is_not_overwritten_without_permission(files, capsys):
    assert main(argv(files, "errors", *FAST)) == 0
    fails(files, capsys, "errors", "already exists")
    assert main(argv(files, "errors", *FAST, "--overwrite")) == 0


def test_module_entry_point_runs_from_the_repository_root(files):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-B", "-m", "p4.analysis.cli", *argv(files, "errors", *FAST)],
                          capture_output=True, text=True, cwd=REPO, env=env)
    assert done.returncode == 0, done.stderr
    helped = subprocess.run([sys.executable, "-B", "-m", "p4.analysis.cli", "--help"], capture_output=True, text=True, cwd=REPO, env=env)
    assert helped.returncode == 0 and "--data-label" in helped.stdout


# ---- synthetic generator and compatibility with the committed P2 / P3 outputs -------------------------------
P3 = REPO / "p3" / "outputs"


@pytest.fixture(scope="module")
def synthetic_inputs(tmp_path_factory):
    from p4.analysis.synthetic_demo import write_synthetic_inputs
    folder = tmp_path_factory.mktemp("synthetic_inputs")
    counts = write_synthetic_inputs(folder)
    return folder, counts


def test_synthetic_generator_reproduces_p3s_split_and_writes_a_marker(synthetic_inputs):
    folder, counts = synthetic_inputs
    assert counts == {"papers": 330, "queries": 200, "val": 23, "test": 32}              # the split P3's runs were made with
    assert "SYNTHETIC DATA" in (folder / "SYNTHETIC_DATA.txt").read_text()
    assert set(pd.read_parquet(folder / "queries.parquet").columns) >= {"query_id", "cited_paper_id", "cited_citation_count", "text"}
    assert not (folder / "context_functions.parquet").exists()                           # labels would have to be invented


@pytest.mark.skipif(not (P3 / "hybrid_test.run").exists(), reason="P3's committed outputs are not present")
@pytest.mark.parametrize("analysis", ["popularity", "coldstart", "year", "errors"])
def test_all_analyses_run_on_the_committed_p2_and_p3_test_runs(synthetic_inputs, tmp_path, analysis):
    folder, _ = synthetic_inputs
    out = tmp_path / analysis
    code = main([analysis, "--data-label", "synthetic", "--qrels", str(folder / "qrels.test.txt"),
                 "--run", f"abstract_bm25={P3 / 'p2_quick/test/abstract_bm25.run'}",
                 "--run", f"expanded_bm25={P3 / 'p2_quick/test/expanded_bm25.run'}",
                 "--run", f"dense={P3 / 'dense_test.run'}", "--run", f"hybrid={P3 / 'hybrid_test.run'}",
                 "--queries", str(folder / "queries.parquet"), "--papers", str(folder / "papers.parquet"),
                 "--index-contexts", str(folder / "index_contexts.parquet"), "--baseline", "abstract_bm25",
                 "--n-boot", "200", "--min-group-size", "10", "--quiet", "--analysis-date", "2026-01-02", "--out", str(out)])
    assert code == 0
    report = (out / f"{analysis}_report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in report and "NOT real findings" in report and "Matched runs:** yes" in report
    assert "judged queries: 32" in report                                                # all 32 test queries, 4 matched systems
    for csv in out.glob("*.csv"):
        assert pd.read_csv(csv)["data_label"].eq("synthetic").all()


# ---- leakage-safe citation counts (--citation-counts) ---------------------------------------------------------
def write_priors(path, counts):
    pd.DataFrame({"paper_id": list(counts), "citation_count_eval": list(counts.values())}).to_parquet(path, index=False)
    return path


def test_citation_counts_option_replaces_papers_counts_and_says_so(files, tmp_path):
    papers = pd.read_parquet(files["papers"])
    safe = write_priors(tmp_path / "priors.parquet", {p: 1 for p in papers["paper_id"]})                # every paper has exactly 1 safe citation
    assert main(argv(files, "popularity", *FAST, "--citation-counts", str(safe))) == 0
    groups = pd.read_csv(files["out"] / "popularity_query_side_groups.csv")
    sizes = groups[(groups.system == "good") & (groups.metric == "rr@10")].set_index("group")["n_queries"]
    assert sizes["1-2"] == 40 and sizes.drop("1-2").sum() == 0                                           # all 40 queries moved to the 1-2 bucket
    report = (files["out"] / "popularity_report.md").read_text(encoding="utf-8")
    assert "citation_count_eval from" in report and "leakage-safe evaluation graph" in report
    # without the option the report states that the counts are NOT leakage-safe
    files["out"] = tmp_path / "plain"
    assert main(argv(files, "popularity", *FAST)) == 0
    assert "NOT leakage-safe" in (files["out"] / "popularity_report.md").read_text(encoding="utf-8")


def test_citation_counts_option_errors(files, capsys, tmp_path):
    assert main(argv(files, "errors", *FAST, "--citation-counts", str(tmp_path / "nope.parquet"))) == 2
    assert "citation-counts file not found" in capsys.readouterr().err
    pd.DataFrame({"paper_id": ["a"], "other": [1]}).to_parquet(tmp_path / "bad.parquet", index=False)
    assert main(argv(files, "errors", *FAST, "--citation-counts", str(tmp_path / "bad.parquet"))) == 2
    assert "no paper_id / citation_count_eval columns" in capsys.readouterr().err
    args = ["errors", "--data-label", "demo", "--qrels", str(files["qrels"]), "--run", f"g={files['run_good']}",
            "--out", str(files["out"]), "--citation-counts", str(write_priors(tmp_path / "p.parquet", {"a": 1})), *FAST]
    assert main(args) == 2 and "--citation-counts needs --papers" in capsys.readouterr().err
