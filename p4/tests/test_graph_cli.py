"""Tests for p4.graph.cli (PageRank and co-citation command line)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from p4.graph.cli import main

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEMO_EDGES, DEMO_PAPERS = FIXTURES / "demo_edges.csv", FIXTURES / "demo_papers.csv"      # synthetic demo graph (see README)


def test_pagerank_command_prints_a_ranked_table_with_titles(capsys):
    assert main(["pagerank", "--edges", str(DEMO_EDGES), "--papers", str(DEMO_PAPERS), "--top-k", "3"]) == 0
    out = capsys.readouterr().out
    assert "PageRank: 8 papers, 10 edges" in out and "converged" in out and "2 cite nothing" in out
    lines = [ln for ln in out.splitlines() if ln.strip() and ln.split()[0] in ("D", "A", "B", "C", "E")]
    assert [ln.split()[0] for ln in lines] == ["D", "A", "B"]                      # top 3, D (cited by all) first
    assert "Demo paper D (synthetic)" in out


def test_pagerank_writes_a_csv_and_refuses_to_overwrite(tmp_path, capsys):
    target = tmp_path / "pr.csv"
    argv = ["pagerank", "--edges", str(DEMO_EDGES), "--papers", str(DEMO_PAPERS), "--top-k", "100", "--out", str(target)]
    assert main(argv) == 0
    table = pd.read_csv(target)
    assert len(table) == 8 and table["score"].sum() == pytest.approx(1.0, abs=1e-5)
    assert list(table.columns) == ["paper_id", "score", "score_scaled", "rank", "title"]
    capsys.readouterr()
    assert main(argv) == 2 and "already exists" in capsys.readouterr().err
    assert main(argv + ["--overwrite"]) == 0


def test_papers_file_adds_papers_that_have_no_edges(capsys):
    assert main(["pagerank", "--edges", str(DEMO_EDGES), "--top-k", "100"]) == 0
    without = capsys.readouterr().out
    assert main(["pagerank", "--edges", str(DEMO_EDGES), "--papers", str(DEMO_PAPERS), "--top-k", "100"]) == 0
    with_papers = capsys.readouterr().out
    assert "7 papers" in without and "8 papers" in with_papers                    # paper E has no edges


def test_pagerank_options_are_passed_through(capsys):
    assert main(["pagerank", "--edges", str(DEMO_EDGES), "--damping", "0.5", "--tol", "1e-3", "--weighted"]) == 0
    out = capsys.readouterr().out
    assert "damping 0.5" in out and "converged" in out
    assert main(["pagerank", "--edges", str(DEMO_EDGES), "--max-iter", "1"]) == 0   # not converged: warns but still answers
    captured = capsys.readouterr()
    assert "DID NOT CONVERGE" in captured.out and "did not converge" in captured.err


def test_cocitation_command_lists_recommendations(capsys):
    assert main(["cocitation", "--edges", str(DEMO_EDGES), "--papers", str(DEMO_PAPERS), "--paper", "A", "--top-k", "5"]) == 0
    out = capsys.readouterr().out
    assert "3 co-cited pairs" in out and "Recommendations for A" in out
    rows = [ln.split()[:3] for ln in out.splitlines() if ln.strip().startswith(("B ", "C "))]
    assert rows == [["B", "2", "1"], ["C", "2", "2"]]


def test_cocitation_for_a_paper_without_neighbours_says_so(capsys):
    assert main(["cocitation", "--edges", str(DEMO_EDGES), "--papers", str(DEMO_PAPERS), "--paper", "E"]) == 0
    assert "(no rows)" in capsys.readouterr().out                                  # E is known (papers file) but cites/is cited by nobody


def test_cocitation_max_references_and_csv_output(tmp_path, capsys):
    target = tmp_path / "rec.csv"
    assert main(["cocitation", "--edges", str(DEMO_EDGES), "--paper", "B", "--max-references", "2", "--out", str(target)]) == 0
    assert pd.read_csv(target)["paper_id"].tolist() == ["A"]                       # X2 (3 refs) skipped: only X1 co-cites A and B
    assert "1 skipped" in capsys.readouterr().out


# ---- errors: exit code 2 and one line, no traceback ---------------------------------------------------
@pytest.mark.parametrize("argv, text", [
    (["cocitation", "--edges", str(DEMO_EDGES), "--paper", "NOPE"], "'NOPE' is not in the co-citation index"),
    (["pagerank", "--edges", "missing.parquet"], "edge file not found"),
    (["pagerank", "--edges", str(DEMO_EDGES), "--papers", "missing.parquet"], "papers file not found"),
    (["pagerank", "--edges", str(DEMO_EDGES), "--top-k", "0"], "top_k"),
    (["pagerank", "--edges", str(DEMO_EDGES), "--damping", "1.5"], "damping"),
    (["pagerank", "--edges", str(DEMO_EDGES), "--tol", "0"], "tol"),
    (["pagerank", "--edges", str(DEMO_EDGES), "--max-iter", "0"], "max_iter"),
    (["cocitation", "--edges", str(DEMO_EDGES), "--paper", "A", "--top-k", "-1"], "top_k"),
    (["cocitation", "--edges", str(DEMO_EDGES), "--paper", "A", "--max-references", "1"], "max_references"),
])
def test_bad_input_gives_a_one_line_error(argv, text, capsys):
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and text in err and "Traceback" not in err


def test_wrong_columns_in_the_edge_file(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("from,to\nA,B\n")
    assert main(["pagerank", "--edges", str(bad)]) == 2
    assert "no source column 'citing_paper_id'" in capsys.readouterr().err


def test_weighted_option_needs_the_weight_column(tmp_path, capsys):
    no_weights = tmp_path / "e.csv"
    no_weights.write_text("citing_paper_id,cited_paper_id\nA,B\n")
    assert main(["pagerank", "--edges", str(no_weights), "--weighted"]) == 2
    assert "num_contexts" in capsys.readouterr().err


def test_reads_p1_parquet_and_handles_an_empty_graph(tmp_path, capsys):
    schema = pa.schema([("citing_paper_id", pa.string()), ("cited_paper_id", pa.string()), ("num_contexts", pa.int32())])
    empty = tmp_path / "edges.parquet"
    pq.write_table(pa.table({"citing_paper_id": [], "cited_paper_id": [], "num_contexts": []}, schema=schema), empty)
    assert main(["pagerank", "--edges", str(empty)]) == 0                         # no edges and no papers file: nothing to rank
    assert "0 papers, 0 edges" in capsys.readouterr().out
    papers = tmp_path / "papers.parquet"
    pq.write_table(pa.table({"paper_id": ["p1", "p2"], "title": ["one", "two"]}), papers)
    assert main(["pagerank", "--edges", str(empty), "--papers", str(papers)]) == 0
    out = capsys.readouterr().out
    assert "2 papers, 0 edges" in out and "0.500000" in out                         # all papers isolated: uniform


def test_module_entry_point_runs_from_another_directory(tmp_path):
    repo = Path(__file__).resolve().parent.parent.parent
    done = subprocess.run([sys.executable, "-B", "-m", "p4.graph.cli", "cocitation", "--edges", str(DEMO_EDGES), "--paper", "A"],
                          capture_output=True, text=True, cwd=repo)
    assert done.returncode == 0, done.stderr
    assert "Recommendations for A" in done.stdout
    helped = subprocess.run([sys.executable, "-B", "-m", "p4.graph.cli", "--help"], capture_output=True, text=True, cwd=repo)
    assert helped.returncode == 0 and "pagerank" in helped.stdout and "cocitation" in helped.stdout


# ---- --weighted keeps the weights (hand-calculated graph: A->B weight 3, A->C weight 1) -------------------
def write_weighted_csv(path, weights=(3, 1)):
    path.write_text(f"citing_paper_id,cited_paper_id,num_contexts\nA,B,{weights[0]}\nA,C,{weights[1]}\n")
    return path


def scores_from_cli(argv, tmp_path):
    target = tmp_path / "out.csv"
    assert main(argv + ["--top-k", "10", "--out", str(target), "--overwrite"]) == 0
    return pd.read_csv(target).set_index("paper_id")["score"].to_dict()


def test_weighted_flag_uses_the_edge_weights_and_unweighted_ignores_them(tmp_path):
    edges = write_weighted_csv(tmp_path / "w.csv")
    weighted = scores_from_cli(["pagerank", "--edges", str(edges), "--weighted"], tmp_path)
    assert weighted == pytest.approx({"A": 0.259740, "B": 0.425325, "C": 0.314935}, abs=1e-6)
    unweighted = scores_from_cli(["pagerank", "--edges", str(edges)], tmp_path)
    assert unweighted == pytest.approx({"A": 0.259740, "B": 0.370130, "C": 0.370130}, abs=1e-6)


def test_weighted_flag_with_p1s_int32_parquet_weights(tmp_path):
    path = tmp_path / "edges.parquet"
    pq.write_table(pa.table({"citing_paper_id": ["A", "A"], "cited_paper_id": ["B", "C"],
                             "num_contexts": pa.array([3, 1], pa.int32())}), path)
    assert scores_from_cli(["pagerank", "--edges", str(path), "--weighted"], tmp_path) == \
        pytest.approx({"A": 0.259740, "B": 0.425325, "C": 0.314935}, abs=1e-6)


def test_weighted_flag_rejects_zero_and_missing_weights_but_unweighted_runs_still_work(tmp_path, capsys):
    zero = write_weighted_csv(tmp_path / "zero.csv", weights=(3, 0))
    assert main(["pagerank", "--edges", str(zero), "--weighted"]) == 2
    err = capsys.readouterr().err
    assert "1 invalid value(s)" in err and "first at row 1: 0" in err and "Traceback" not in err
    assert main(["pagerank", "--edges", str(zero)]) == 0                     # weights not requested: not looked at
    no_column = tmp_path / "nocol.csv"
    no_column.write_text("citing_paper_id,cited_paper_id\nA,B\n")
    capsys.readouterr()
    assert main(["pagerank", "--edges", str(no_column), "--weighted"]) == 2
    assert "no weight column 'num_contexts'" in capsys.readouterr().err


def test_cli_output_is_identical_across_processes_with_different_hash_seeds(tmp_path):
    import os
    rng = __import__("numpy").random.default_rng(5)
    lines = ["citing_paper_id,cited_paper_id,num_contexts"]
    for _ in range(300):                                                       # a 60-paper random graph with weights, plus noise rows
        a, b = rng.integers(0, 60, 2)
        lines.append(f"{1800 + a % 7}.{10000 + a:05d},{1800 + b % 7}.{10000 + b:05d},{int(rng.integers(1, 6))}")
    edges = tmp_path / "g.csv"
    edges.write_text("\n".join(lines) + "\n")
    repo = Path(__file__).resolve().parent.parent.parent
    commands = {"pagerank": ["pagerank", "--weighted", "--top-k", "60"],
                "cocitation": ["cocitation", "--paper", "1800.10000", "--top-k", "30"]}
    for name, command in commands.items():
        outputs = set()
        for seed in ("0", "1", "4242"):
            env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}
            done = subprocess.run([sys.executable, "-B", "-m", "p4.graph.cli", *command, "--edges", str(edges)],
                                  capture_output=True, text=True, cwd=repo, env=env)
            assert done.returncode == 0, done.stderr
            outputs.add(done.stdout)
        assert len(outputs) == 1, f"{name} output depends on the hash seed"
