"""Tests for p4.graph.eval_graph (the leakage-safe evaluation graph and its priors). All data here is DEMO (hand-made)."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from p4.graph import eval_graph as EG
from p4.graph.pagerank import pagerank
from p4.tests import graph_fixtures as GF


@pytest.fixture
def processed(tmp_path):
    return GF.small_processed(tmp_path / "processed")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_damping_column_names():
    assert [EG.damping_column(d) for d in (0.5, 0.85, 0.95, 0.1, 0.0)] == ["pagerank_d050", "pagerank_d085", "pagerank_d095", "pagerank_d010", "pagerank_d000"]


def test_build_writes_the_expected_files_and_manifest(processed, tmp_path):
    out = tmp_path / "eval"
    manifest = EG.build(processed, out, data_label="demo")
    assert sorted(p.name for p in out.iterdir()) == ["MANIFEST.json", "edges_eval.parquet", "priors.parquet"]
    assert manifest["protocol"] == EG.PROTOCOL and manifest["data_label"] == "demo" and manifest["hold_out_splits"] == ["val", "test"]
    assert manifest["prior_columns"] == ["citation_count_eval", "pagerank_d050", "pagerank_d085", "pagerank_d095"]
    assert manifest["edge_removal"]["edges_removed"] == 4 and manifest["edge_removal"]["edges_kept"] == 3        # see test_leakage_audit
    assert manifest["papers"] == 10 and manifest["queries_per_split"] == {"test": 2, "train": 2, "val": 1}
    assert json.loads((out / "MANIFEST.json").read_text())["data_label"] == "demo"
    assert set(manifest["input_sha256"]) == {"papers.parquet", "queries.parquet", "edges.parquet"}
    assert manifest["input_sha256"]["edges.parquet"] == sha(processed / "edges.parquet")


def test_priors_by_hand(processed, tmp_path):
    out = tmp_path / "eval"
    EG.build(processed, out, data_label="demo")
    priors, _ = EG.load_priors(out)
    counts = priors.set_index("paper_id")["citation_count_eval"]
    # evaluation graph keeps C1->T1, C4->T2, C5->T1 : T1 has 2 citations, T2 has 1, every other paper 0
    assert counts.to_dict() == {"T1": 2, "T2": 1, "T3": 0, "T4": 0, "T5": 0, "C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0}
    assert len(priors) == 10 and priors["paper_id"].is_unique                                         # every paper, isolated ones included
    for column in ("pagerank_d050", "pagerank_d085", "pagerank_d095"):
        assert priors[column].sum() == pytest.approx(1.0) and (priors[column] > 0).all()            # isolated papers are never zero
    assert priors.set_index("paper_id").loc["T1", "pagerank_d085"] > priors.set_index("paper_id").loc["T3", "pagerank_d085"]


def test_pagerank_columns_equal_pagerank_on_the_evaluation_graph(processed, tmp_path):
    out = tmp_path / "eval"
    EG.build(processed, out, data_label="demo", dampings=(0.85,))
    priors, _ = EG.load_priors(out)
    safe = pd.read_parquet(out / "edges_eval.parquet")
    expected = pagerank(safe, nodes=priors["paper_id"], damping=0.85).scores
    assert priors.set_index("paper_id")["pagerank_d085"].to_dict() == pytest.approx(expected.to_dict())
    assert list(priors.columns) == ["paper_id", "citation_count_eval", "pagerank_d085"]


def test_no_held_out_citing_paper_contributes_any_edge(processed, tmp_path):
    out = tmp_path / "eval"
    EG.build(processed, out, data_label="demo")
    safe = pd.read_parquet(out / "edges_eval.parquet")
    queries = pd.read_parquet(processed / "queries.parquet")
    held = set(queries.loc[queries.split.isin(["val", "test"]), "citing_paper_id"])
    assert not set(safe["citing_paper_id"]) & held                                                    # none of C2 or C3 remains


def test_hold_out_choice_changes_the_graph(processed, tmp_path):
    EG.build(processed, tmp_path / "all", data_label="demo")
    manifest = EG.build(processed, tmp_path / "train_only_held_out", data_label="demo", hold_out=("train",))
    # the train citing papers are C1 and C4, one edge each (C1->T1, C4->T2): 2 removed, 5 kept
    assert manifest["hold_out_splits"] == ["train"] and manifest["edge_removal"]["edges_removed"] == 2 and manifest["edge_removal"]["edges_kept"] == 5


def test_strict_index_drops_contexts_written_by_held_out_citing_papers(processed, tmp_path):
    out = tmp_path / "eval"
    manifest = EG.build(processed, out, data_label="demo", strict_index=True)
    strict = pd.read_parquet(out / "index_contexts_strict.parquet")
    assert sorted(strict["context_id"]) == ["x2", "x3"]                                               # x1 (C3) and x4 (C2) were by held-out citers
    assert manifest["strict_index"] == {"index_contexts_total": 4, "removed": 2, "kept": 2}


def test_build_is_deterministic_and_never_touches_its_inputs(processed, tmp_path):
    before = {p.name: sha(p) for p in processed.iterdir()}
    EG.build(processed, tmp_path / "a", data_label="demo")
    EG.build(processed, tmp_path / "b", data_label="demo")
    assert {p.name: sha(p) for p in processed.iterdir()} == before
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a" / "priors.parquet"), pd.read_parquet(tmp_path / "b" / "priors.parquet"))
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a" / "edges_eval.parquet"), pd.read_parquet(tmp_path / "b" / "edges_eval.parquet"))


def test_pagerank_convergence_is_recorded_per_damping(processed, tmp_path):
    manifest = EG.build(processed, tmp_path / "eval", data_label="demo")
    info = manifest["pagerank"]
    assert set(info) == {"pagerank_d050", "pagerank_d085", "pagerank_d095"} and all(v["converged"] for v in info.values())
    assert info["pagerank_d050"]["iterations"] < info["pagerank_d095"]["iterations"]                  # higher damping converges more slowly
    assert all(v["n_nodes"] == 10 and v["n_edges"] == 3 for v in info.values())


@pytest.mark.parametrize("kwargs, message", [
    (dict(data_label="maybe"), "data label"), (dict(data_label="demo", dampings=(1.0,)), "damping"),
    (dict(data_label="demo", dampings=(-0.1,)), "damping"), (dict(data_label="demo", hold_out=("dev",)), "hold-out"),
    (dict(data_label="demo", hold_out=()), "hold-out")])
def test_bad_settings_are_rejected(processed, tmp_path, kwargs, message):
    with pytest.raises(EG.EvalGraphError, match=message):
        EG.build(processed, tmp_path / "eval", **kwargs)
    assert not (tmp_path / "eval").exists()                                                            # nothing was written


def test_existing_output_and_missing_inputs_are_refused(processed, tmp_path):
    out = tmp_path / "eval"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(EG.EvalGraphError, match="refusing to overwrite"):
        EG.build(processed, out, data_label="demo")
    assert (out / "keep.txt").read_text() == "mine"
    (processed / "edges.parquet").unlink()
    with pytest.raises(EG.EvalGraphError, match="edges.parquet not found"):
        EG.build(processed, tmp_path / "other", data_label="demo")


def test_load_priors_rejects_a_foreign_folder(tmp_path):
    with pytest.raises(EG.EvalGraphError, match="does not look like an evaluation-graph folder"):
        EG.load_priors(tmp_path)


def test_cli(processed, tmp_path, capsys):
    out = tmp_path / "eval"
    assert EG.main(["--processed", str(processed), "--out", str(out), "--data-label", "demo", "--dampings", "0.5,0.85"]) == 0
    printed = capsys.readouterr().out
    assert "[demo] evaluation graph: 3 of 7 edges kept" in printed and "pagerank_d050" in printed
    assert EG.main(["--processed", str(processed), "--out", str(out), "--data-label", "demo"]) == 2            # now non-empty
    assert "refusing to overwrite" in capsys.readouterr().err
    assert EG.main(["--processed", str(tmp_path / "nope"), "--out", str(tmp_path / "o2"), "--data-label", "demo"]) == 2
    assert EG.main(["--processed", str(processed), "--out", str(tmp_path / "o3"), "--data-label", "demo", "--dampings", "x"]) == 2
