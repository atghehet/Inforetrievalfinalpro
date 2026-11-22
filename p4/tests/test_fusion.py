"""Tests for p4.graph.fusion (authority-prior fusion). All data here is DEMO (hand-made); nothing is a result."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from p4.eval.harness import evaluate_runs
from p4.eval.trec_io import read_run
from p4.graph import eval_graph as EG
from p4.graph import fusion as FU
from p4.tests import graph_fixtures as GF


# ---- score normalisation -------------------------------------------------------------------------------
def test_minmax_zscore_and_rank_by_hand():
    assert FU.normalise_scores([10, 5, 0], "minmax") == [1.0, 0.5, 0.0]
    z = FU.normalise_scores([10, 5, 0], "zscore")                                     # mean 5, population std sqrt(50/3)
    assert z == pytest.approx([5 / math.sqrt(50 / 3), 0.0, -5 / math.sqrt(50 / 3)])
    assert FU.normalise_scores([10, 5, 0], "rank") == [1.0, 0.5, 0.0]
    assert FU.normalise_scores([9, 1, 1, 0], "rank") == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])   # tied scores keep list order


def test_rank_normalisation_is_stable_for_ties_and_depends_only_on_order():
    assert FU.normalise_scores([5, 5, 1], "rank") == [1.0, 0.5, 0.0]                  # tied scores keep list order
    assert FU.normalise_scores([100, 2, 1], "rank") == FU.normalise_scores([3, 2, 1], "rank")


@pytest.mark.parametrize("method", ("minmax", "zscore", "rank"))                      # "none" keeps raw scores; see the raw-mode tests below
def test_no_spread_empty_and_single_lists(method):
    assert FU.normalise_scores([], method) == []
    expected_single = [1.0] if method == "rank" else [0.0]
    assert FU.normalise_scores([4.2], method) == expected_single
    if method != "rank":
        assert FU.normalise_scores([3, 3, 3], method) == [0.0, 0.0, 0.0]


def test_unknown_normalisation_is_rejected():
    with pytest.raises(FU.FusionError, match="score normalisation"):
        FU.normalise_scores([1, 2], "softmax")


# ---- prior preparation ---------------------------------------------------------------------------------
def test_identity_prior_is_rescaled_to_unit_range():
    p = FU.prepare_prior(pd.Series({"a": 2.0, "b": 4.0, "c": 6.0}), "identity")
    assert p.to_dict() == {"a": 0.0, "b": 0.5, "c": 1.0}


def test_log_prior_with_zeros_puts_zeros_just_below_the_smallest_positive_value():
    p = FU.prepare_prior(pd.Series({"none": 0, "one": 1, "three": 3}), "log")
    low, mid, high = math.log(0.5), math.log(1.5), math.log(3.5)                      # eps = smallest positive / 2 = 0.5
    expected = [(x - low) / (high - low) for x in (low, mid, high)]
    assert [p["none"], p["one"], p["three"]] == pytest.approx(expected)
    assert 0 == p["none"] < p["one"] < p["three"] == 1.0


def test_log_prior_compresses_a_heavy_tailed_prior():
    values = pd.Series({"a": 1e-6, "b": 1e-5, "c": 1e-4, "d": 1e-3})                  # PageRank-like, orders of magnitude apart
    identity, log = FU.prepare_prior(values, "identity"), FU.prepare_prior(values, "log")
    assert identity["c"] < 0.1 and log["c"] == pytest.approx(2 / 3)                    # identity crushes small values, log spreads them


def test_constant_prior_carries_no_information_and_bad_values_are_rejected():
    assert FU.prepare_prior(pd.Series({"a": 5, "b": 5}), "log").tolist() == [0.0, 0.0]
    assert FU.prepare_prior(pd.Series({"a": 0, "b": 0}), "log").tolist() == [0.0, 0.0]
    for bad in (pd.Series({"a": -1.0, "b": 1.0}), pd.Series({"a": np.nan, "b": 1.0}), pd.Series({"a": np.inf, "b": 1.0})):
        with pytest.raises(FU.FusionError, match="finite and non-negative"):
            FU.prepare_prior(bad, "identity")
    with pytest.raises(FU.FusionError, match="prior transform"):
        FU.prepare_prior(pd.Series({"a": 1.0}), "sqrt")


# ---- fusion -----------------------------------------------------------------------------------------------
RUN = {"q1": [("A", 10.0), ("B", 5.0), ("C", 0.0)]}
PRIOR = pd.Series({"A": 0.0, "B": 1.0, "C": 0.5})


def order(run, qid="q1"):
    return [d for d, _ in run[qid]]


def test_convex_fusion_by_hand():
    # minmax scores: A 1, B 0.5, C 0 ; prior: A 0, B 1, C 0.5
    half, _ = FU.fuse_run(RUN, PRIOR, 0.5, mode="convex")                              # 0.5*s + 0.5*p : A .5, B .75, C .25
    assert {d: round(s, 4) for d, s in half["q1"]} == {"B": 0.75, "A": 0.5, "C": 0.25} and order(half) == ["B", "A", "C"]
    assert order(FU.fuse_run(RUN, PRIOR, 0.0)[0]) == ["A", "B", "C"]                   # lambda 0 = the original ranking
    assert order(FU.fuse_run(RUN, PRIOR, 1.0)[0]) == ["B", "C", "A"]                   # lambda 1 = the prior alone


def test_additive_fusion_by_hand():
    fused, _ = FU.fuse_run(RUN, PRIOR, 1.0, mode="additive")                           # s + p : A 1, B 1.5, C 0.5
    assert {d: round(s, 4) for d, s in fused["q1"]} == {"B": 1.5, "A": 1.0, "C": 0.5}
    assert order(FU.fuse_run(RUN, PRIOR, 3.0, mode="additive")[0]) == ["B", "C", "A"]  # lambda above 1 is allowed in additive mode


def test_lambda_zero_reproduces_the_original_ranking_even_with_tied_scores():
    run = {"q": [("X", 1.0), ("Y", 1.0), ("Z", 1.0)]}                                    # no spread: every normalised score is 0
    assert order(FU.fuse_run(run, pd.Series({"X": 0.2, "Y": 0.9, "Z": 0.5}), 0.0)[0], "q") == ["X", "Y", "Z"]
    assert order(FU.fuse_run(run, pd.Series({"X": 0.2, "Y": 0.9, "Z": 0.5}), 0.5)[0], "q") == ["Y", "Z", "X"]       # now the prior decides


def test_ties_break_by_original_rank_then_document_id_and_are_deterministic():
    run = {"q": [("B", 3.0), ("A", 2.0), ("C", 1.0)]}
    flat = pd.Series({"A": 0.5, "B": 0.5, "C": 0.5})                                   # equal priors: only the base scores differ
    assert order(FU.fuse_run(run, flat, 0.5)[0], "q") == ["B", "A", "C"]
    tied = {"q": [("Q", 1.0), ("P", 1.0)]}
    first = FU.fuse_run(tied, pd.Series({"P": 0.0, "Q": 0.0}), 0.5)[0]
    assert first == FU.fuse_run(tied, pd.Series({"P": 0.0, "Q": 0.0}), 0.5)[0] and order(first, "q") == ["Q", "P"]


def test_documents_without_a_prior_get_zero_and_are_counted():
    fused, stats = FU.fuse_run(RUN, pd.Series({"A": 0.4}), 0.5)
    assert stats == {"n_docs": 3, "n_docs_without_prior": 2}
    assert order(fused) == ["A", "B", "C"]                                              # B and C: prior 0 -> 0.5*s only; A: 0.5 + 0.2


def test_queries_are_independent_and_all_kept():
    run = {"q1": [("A", 2.0), ("B", 1.0)], "q2": [("B", 9.0), ("A", 1.0)], "q3": []}
    fused, stats = FU.fuse_run(run, pd.Series({"A": 1.0, "B": 0.0}), 0.5)
    assert set(fused) == {"q1", "q2", "q3"} and fused["q3"] == [] and stats["n_docs"] == 4
    assert order(fused, "q1") == ["A", "B"] and order(fused, "q2") == ["B", "A"]


@pytest.mark.parametrize("lam, mode", [(-0.1, "convex"), (1.5, "convex"), (float("nan"), "convex"), (-1, "additive"), (float("inf"), "additive")])
def test_invalid_lambda_is_rejected(lam, mode):
    with pytest.raises(FU.FusionError, match="lambda"):
        FU.fuse_run(RUN, PRIOR, lam, mode=mode)
    with pytest.raises(FU.FusionError, match="mode"):
        FU.fuse_run(RUN, PRIOR, 0.5, mode="product")


def test_written_run_is_valid_trec_and_never_overwrites(tmp_path):
    fused, _ = FU.fuse_run(RUN, PRIOR, 0.5)
    path = tmp_path / "fused.run"
    FU.write_run(fused, path, "demo_tag")
    assert read_run(path)["q1"] == [(d, pytest.approx(s)) for d, s in fused["q1"]]
    assert path.read_text().splitlines()[0].endswith("demo_tag") and path.read_text().splitlines()[0].startswith("q1 Q0 B 1 ")
    with pytest.raises(FU.FusionError, match="refusing to overwrite"):
        FU.write_run(fused, path, "x")


# ---- sweep and selection ---------------------------------------------------------------------------------
def test_sweep_always_includes_lambda_zero_and_matches_the_harness_baseline():
    run, qrels = GF.leak_run_and_qrels("val")
    prior = pd.Series({f"D{j:02d}": 0.0 for j in range(1, 31)} | {f"Tv{i:02d}": 1.0 for i in range(1, 21)})
    table = FU.sweep(run, qrels, {"p": prior}, [0.5, 1.0], transform="identity")
    assert table["lambda"].tolist() == [0.0, 0.5, 1.0]                                  # 0 added although not requested
    baseline = evaluate_runs({"b": run}, qrels, (10,)).summary.iloc[0]["MRR@10"]
    assert table.loc[0, "MRR@10"] == pytest.approx(baseline) == pytest.approx(0.25)
    assert table.loc[1, "MRR@10"] == pytest.approx(1.0)                                 # a prior that sees the targets lifts them to rank 1


def test_best_setting_prefers_the_smaller_lambda_on_ties_and_flags_no_improvement():
    flat = pd.DataFrame({"prior": ["a"] * 3, "lambda": [0.0, 0.5, 1.0], "mode": "convex", "score_norm": "minmax",
                         "prior_transform": "log", "MRR@10": [0.3, 0.3, 0.2]})
    best = FU.best_setting(flat)
    assert best["lambda"] == 0.0 and not best["improves_on_validation"] and best["baseline_validation_value"] == 0.3
    better = flat.assign(**{"MRR@10": [0.3, 0.4, 0.4]})
    best = FU.best_setting(better)
    assert best["lambda"] == 0.5 and best["improves_on_validation"] and best["validation_value"] == 0.4         # tie 0.5 vs 1.0 -> smaller
    two = pd.concat([better.assign(prior="a"), better.assign(prior="b")], ignore_index=True)
    assert FU.best_setting(two)["prior"] == "a"                                                                  # then the first prior


# ---- the leak, end to end (DEMO scenario with an exact answer) -------------------------------------------------
@pytest.fixture
def scenario(tmp_path):
    processed = GF.leak_scenario(tmp_path / "processed")
    safe, leaky = tmp_path / "safe", tmp_path / "leaky"
    EG.build(processed, safe, data_label="demo")                                       # val and test citing papers held out
    EG.build(processed, leaky, data_label="demo", hold_out=("train",))                 # nothing held out: the queries' own edges stay
    return {"safe": safe, "leaky": leaky}


def test_the_scenario_has_the_expected_priors(scenario):
    safe, _ = EG.load_priors(scenario["safe"])
    leaky, _ = EG.load_priors(scenario["leaky"])
    assert (safe["citation_count_eval"] == 0).all()                                      # no paper is cited once the held-out edges are gone
    targets = leaky[leaky["paper_id"].str.startswith("T")]
    distractors = leaky[leaky["paper_id"].str.startswith("D")]
    assert (targets["citation_count_eval"] == 1).all() and (distractors["citation_count_eval"] == 0).all()


def test_a_safe_prior_cannot_help_but_a_leaky_prior_lifts_every_target_to_rank_one(scenario):
    run, qrels = GF.leak_run_and_qrels("val")
    _, safe_best = FU.tune_fusion(run, qrels, scenario["safe"], data_label="demo", prior_columns=["citation_count_eval"])
    assert safe_best["lambda"] == 0.0 and not safe_best["improves_on_validation"] and safe_best["validation_value"] == pytest.approx(0.25)
    _, leaky_best = FU.tune_fusion(run, qrels, scenario["leaky"], data_label="demo", prior_columns=["citation_count_eval"],
                                   allow_leaky_prior=True)
    assert leaky_best["improves_on_validation"] and leaky_best["lambda"] > 0 and leaky_best["validation_value"] == pytest.approx(1.0)
    assert any("LEAKY PRIOR ALLOWED" in w for w in leaky_best["warnings"])               # the output is stamped


def test_the_leaky_prior_is_refused_without_the_explicit_flag(scenario):
    run, qrels = GF.leak_run_and_qrels("val")
    with pytest.raises(FU.FusionError, match="leakage"):
        FU.tune_fusion(run, qrels, scenario["leaky"], data_label="demo")


def test_tune_on_validation_then_apply_to_test_end_to_end(scenario):
    val_run, val_qrels = GF.leak_run_and_qrels("val")
    test_run, test_qrels = GF.leak_run_and_qrels("test")
    table, best = FU.tune_fusion(val_run, val_qrels, scenario["safe"], data_label="demo")
    assert set(table["prior"]) == {"citation_count_eval", "pagerank_d050", "pagerank_d085", "pagerank_d095"} and table["data_label"].eq("demo").all()
    fused, stats = FU.apply_fusion(test_run, scenario["safe"], best, data_label="demo", split="test")
    mrr = evaluate_runs({"f": fused}, test_qrels, (10,)).summary.iloc[0]["MRR@10"]
    base = evaluate_runs({"b": test_run}, test_qrels, (10,)).summary.iloc[0]["MRR@10"]
    assert base == pytest.approx(0.25) and mrr == pytest.approx(0.25) and stats["warnings"] == []   # no gain: nothing to learn from a safe graph


def test_apply_checks_split_label_and_column(scenario):
    run, qrels = GF.leak_run_and_qrels("val")
    _, best = FU.tune_fusion(run, qrels, scenario["safe"], data_label="demo", prior_columns=["pagerank_d085"])
    test_run, _ = GF.leak_run_and_qrels("test")
    with pytest.raises(FU.FusionError, match="data label mismatch"):
        FU.apply_fusion(test_run, scenario["safe"], best, data_label="real")
    with pytest.raises(FU.FusionError, match="leakage"):
        FU.apply_fusion(test_run, scenario["leaky"], best, data_label="demo", split="test")
    with pytest.raises(FU.FusionError, match="not in the prior folder"):
        FU.apply_fusion(test_run, scenario["safe"], {**best, "prior": "nope"}, data_label="demo")
    with pytest.raises(FU.FusionError, match="prior column"):
        FU.tune_fusion(run, qrels, scenario["safe"], data_label="demo", prior_columns=["nope"])


def test_data_label_must_match_the_prior_folder(tmp_path, scenario):
    run, qrels = GF.leak_run_and_qrels("val")
    with pytest.raises(FU.FusionError, match="data label mismatch"):
        FU.tune_fusion(run, qrels, scenario["safe"], data_label="synthetic")


def test_tuning_has_no_access_to_test_data():
    import inspect
    parameters = inspect.signature(FU.tune_fusion).parameters
    assert "qrels_val" in parameters and not any("test" in name for name in parameters)                # validation only, by construction


# ---- command line --------------------------------------------------------------------------------------------
def test_cli_tune_then_apply(tmp_path, scenario, capsys):
    val_run, val_qrels = GF.leak_run_and_qrels("val")
    test_run, _ = GF.leak_run_and_qrels("test")
    GF.write_run_file(tmp_path / "val.run", val_run)
    GF.write_run_file(tmp_path / "test.run", test_run)
    GF.write_qrels_file(tmp_path / "val.qrels", val_qrels)
    out = tmp_path / "tuning"
    assert FU.main(["tune", "--data-label", "demo", "--run", str(tmp_path / "val.run"), "--qrels", str(tmp_path / "val.qrels"),
                    "--prior-dir", str(scenario["safe"]), "--out", str(out), "--prior", "citation_count_eval", "--lambdas", "0,0.5,1"]) == 0
    printed = capsys.readouterr().out
    assert "[demo] best on validation" in printed and "does NOT improve on" in printed
    table = pd.read_csv(out / "fusion_tuning.csv")
    assert list(table["lambda"]) == [0.0, 0.5, 1.0] and table["data_label"].eq("demo").all()
    best = json.loads((out / "fusion_best.json").read_text())
    assert best["data_label"] == "demo" and best["prior_hold_out_splits"] == ["val", "test"] and best["n_validation_queries"] == 20
    fused = tmp_path / "fused_test.run"
    assert FU.main(["apply", "--data-label", "demo", "--run", str(tmp_path / "test.run"), "--prior-dir", str(scenario["safe"]),
                    "--best", str(out / "fusion_best.json"), "--out", str(fused), "--tag", "demo_fusion"]) == 0
    assert read_run(fused).keys() == test_run.keys() and fused.read_text().splitlines()[0].endswith("demo_fusion")


def test_cli_errors_are_one_line_and_exit_2(tmp_path, scenario, capsys):
    val_run, val_qrels = GF.leak_run_and_qrels("val")
    GF.write_run_file(tmp_path / "val.run", val_run)
    GF.write_qrels_file(tmp_path / "val.qrels", val_qrels)
    base = ["tune", "--run", str(tmp_path / "val.run"), "--qrels", str(tmp_path / "val.qrels"), "--out", str(tmp_path / "o")]
    assert FU.main(base + ["--data-label", "demo", "--prior-dir", str(scenario["leaky"])]) == 2
    assert "leakage" in capsys.readouterr().err
    assert FU.main(base + ["--data-label", "real", "--prior-dir", str(scenario["safe"])]) == 2
    assert "data label mismatch" in capsys.readouterr().err
    assert FU.main(base + ["--data-label", "demo", "--prior-dir", str(tmp_path / "nope")]) == 2
    assert FU.main(base + ["--data-label", "demo", "--prior-dir", str(scenario["safe"]), "--lambdas", "2"]) == 2
    assert "lambda" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        FU.main(["tune", "--run", "x", "--qrels", "y", "--prior-dir", "z", "--out", "o"])           # --data-label is required


def test_apply_uses_exactly_the_frozen_lambda_mode_normalisation_and_prior(scenario):
    # The safe graph of this scenario has no edges, so its prior is constant and lambda cannot matter. To check that lambda IS applied the
    # prior must carry information: use the leaky graph, allowed explicitly (this is a mechanics test on DEMO data, not a result).
    run, _ = GF.leak_run_and_qrels("test")
    priors, _ = EG.load_priors(scenario["leaky"])
    best = {"data_label": "demo", "prior": "citation_count_eval", "lambda": 0.2, "mode": "additive", "score_norm": "zscore",
            "prior_transform": "identity"}
    apply = lambda b: FU.apply_fusion(run, scenario["leaky"], b, data_label="demo", split="test", allow_leaky_prior=True)[0]
    prior01 = FU.prepare_prior(priors.set_index("paper_id")["citation_count_eval"], "identity")
    expected, _ = FU.fuse_run(run, prior01, 0.2, mode="additive", score_norm="zscore")
    assert apply(best) == expected                                                              # identical scores, not just the same order
    assert apply({**best, "lambda": 0.5}) != apply(best)                                        # a different lambda really gives different scores
    assert apply({**best, "mode": "convex"}) != apply(best) and apply({**best, "score_norm": "minmax"}) != apply(best)


# ---- raw mode: score' = score + lambda * log(prior), the formula in docs/P4_HANDOFF.md (toy numbers only) -----------
RAW_RUN = {"q1": [("A", 3.0), ("B", 2.9)]}                                              # BM25-like raw scores, A first
RAW_PRIOR = pd.Series({"A": 1e-5, "B": 1e-3})                                            # PageRank-like values; B is the better connected paper


def raw_prior(values=RAW_PRIOR):
    return FU.prepare_prior(values, "log_raw")


def test_raw_mode_is_exactly_score_plus_lambda_times_log_prior():
    prior = raw_prior()
    assert prior["A"] == pytest.approx(math.log(1e-5)) and prior["B"] == pytest.approx(math.log(1e-3))   # log, and NOT rescaled to [0, 1]
    for lam, expected_order in ((0.0, ["A", "B"]), (0.01, ["A", "B"]), (0.1, ["B", "A"]), (3.0, ["B", "A"])):
        fused, _ = FU.fuse_run(RAW_RUN, prior, lam, mode="raw", score_norm="none")
        scores = dict(fused["q1"])
        assert scores["A"] == pytest.approx(3.0 + lam * math.log(1e-5)) and scores["B"] == pytest.approx(2.9 + lam * math.log(1e-3))
        assert order(fused) == expected_order
    # lambda = 0 returns the system's own scores unchanged (no per-query normalisation is applied in raw mode)
    assert dict(FU.fuse_run(RAW_RUN, prior, 0.0, mode="raw", score_norm="none")[0]["q1"]) == {"A": 3.0, "B": 2.9}
    # by hand at lambda 0.1: A = 3.0 - 1.1513 = 1.8487, B = 2.9 - 0.6908 = 2.2092, so B moves first
    fused, _ = FU.fuse_run(RAW_RUN, prior, 0.1, mode="raw", score_norm="none")
    assert dict(fused["q1"])["A"] == pytest.approx(1.8487, abs=1e-4) and dict(fused["q1"])["B"] == pytest.approx(2.2092, abs=1e-4)


def test_raw_mode_zero_prior_uses_the_documented_epsilon_and_never_overflows():
    assert FU.RAW_LOG_EPSILON == 1e-12
    prior = raw_prior(pd.Series({"A": 0.0, "B": 1e-3, "C": 1e-15}))                       # an exact zero, and a value below epsilon
    assert prior["A"] == pytest.approx(math.log(1e-12)) and prior["C"] == pytest.approx(math.log(1e-12))
    assert prior["B"] == pytest.approx(math.log(1e-3)) and np.isfinite(prior).all()
    fused, _ = FU.fuse_run({"q1": [("A", 1.0), ("B", 1.0)]}, prior, 0.5, mode="raw", score_norm="none")
    assert order(fused) == ["B", "A"] and all(math.isfinite(s) for _, s in fused["q1"])  # zero prior ranks below a positive one, no -inf
    with pytest.raises(FU.FusionError, match="non-negative"):
        FU.prepare_prior(pd.Series({"A": -1.0}), "log_raw")


def test_raw_mode_document_without_a_prior_gets_the_lowest_value_not_zero():
    # In raw mode log values are negative, so a default of 0.0 would be the HIGHEST prior (log 1). The missing document must get log(eps).
    prior = raw_prior(pd.Series({"A": 1e-5}))                                            # B has no prior at all
    fused, stats = FU.fuse_run({"q1": [("B", 3.0), ("A", 2.9)]}, prior, 0.1, mode="raw", score_norm="none")
    assert stats == {"n_docs": 2, "n_docs_without_prior": 1}
    assert dict(fused["q1"])["B"] == pytest.approx(3.0 + 0.1 * math.log(1e-12)) and order(fused) == ["A", "B"]


def test_raw_mode_settings_are_consistent_and_existing_modes_keep_their_defaults():
    assert FU.resolve_settings("raw", None, None) == ("none", "log_raw")
    assert FU.resolve_settings("convex", None, None) == ("minmax", "log") and FU.resolve_settings("additive", None, None) == ("minmax", "log")
    assert FU.resolve_settings("convex", "zscore", "identity") == ("zscore", "identity")      # explicit choices are untouched
    for bad in (("raw", "minmax", None), ("raw", None, "log"), ("convex", "none", None), ("additive", None, "log_raw")):
        with pytest.raises(FU.FusionError):
            FU.resolve_settings(*bad)
    with pytest.raises(FU.FusionError, match="only with it"):
        FU.fuse_run(RAW_RUN, raw_prior(), 0.1, mode="raw", score_norm="minmax")
    with pytest.raises(FU.FusionError, match="only with it"):
        FU.fuse_run(RAW_RUN, PRIOR, 0.1, mode="convex", score_norm="none")
    assert FU.fuse_run(RAW_RUN, raw_prior(), 5.0, mode="raw", score_norm="none")                # lambda above 1 is allowed
    with pytest.raises(FU.FusionError, match="lambda"):
        FU.fuse_run(RAW_RUN, raw_prior(), -0.1, mode="raw", score_norm="none")


def test_raw_mode_sweep_picks_the_smallest_lambda_that_helps_and_keeps_lambda_zero_as_baseline():
    # Toy validation set: the relevant paper is B, ranked second by the system, but with the higher prior (so a prior can help).
    # B overtakes A when lambda * (log 1e-3 - log 1e-5) = lambda * 4.605 exceeds the score gap 0.1, i.e. lambda > 0.0217.
    run = {f"q{i}": [("A", 3.0), ("B", 2.9)] for i in range(4)}
    qrels = {f"q{i}": {"B": 1} for i in range(4)}
    table = FU.sweep(run, qrels, {"toy": RAW_PRIOR}, FU.DEFAULT_RAW_LAMBDAS, mode="raw")
    assert set(table["mode"]) == {"raw"} and set(table["score_norm"]) == {"none"} and set(table["prior_transform"]) == {"log_raw"}
    mrr = dict(zip(table["lambda"], table["MRR@10"]))
    assert mrr[0.0] == pytest.approx(0.5) and mrr[0.01] == pytest.approx(0.5)                # B second -> 0.5, 0.01 is too small to flip
    assert mrr[0.03] == pytest.approx(1.0) and mrr[3.0] == pytest.approx(1.0)                # 0.03 is the first grid value above 0.0217
    best = FU.best_setting(table)
    assert best["lambda"] == 0.03 and best["improves_on_validation"] and best["validation_value"] == pytest.approx(1.0)


def test_raw_mode_tune_and_apply_end_to_end_and_cli(tmp_path, scenario, capsys):
    # Mechanics on the DEMO scenario (not a result): the safe graph has no edges, so PageRank is constant and cannot reorder anything.
    val_run, val_qrels = GF.leak_run_and_qrels("val")
    test_run, _ = GF.leak_run_and_qrels("test")
    table, best = FU.tune_fusion(val_run, val_qrels, scenario["safe"], data_label="demo", prior_columns=["pagerank_d085"], mode="raw")
    assert (best["mode"], best["score_norm"], best["prior_transform"]) == ("raw", "none", "log_raw")
    assert best["lambda"] == 0.0 and not best["improves_on_validation"]                      # no gain, so no prior wins (ties go to lambda 0)
    assert sorted(table["lambda"].unique()) == sorted(FU.DEFAULT_RAW_LAMBDAS)                # the wider raw grid is the default in raw mode
    fused, _ = FU.apply_fusion(test_run, scenario["safe"], best, data_label="demo", split="test")
    assert {q: [d for d, _ in r] for q, r in fused.items()} == {q: [d for d, _ in r] for q, r in test_run.items()}
    # command line: --mode raw fills in score_norm / prior_transform; a conflicting choice is a one-line error
    GF.write_run_file(tmp_path / "val.run", val_run)
    GF.write_qrels_file(tmp_path / "val.qrels", val_qrels)
    args = ["tune", "--data-label", "demo", "--run", str(tmp_path / "val.run"), "--qrels", str(tmp_path / "val.qrels"),
            "--prior-dir", str(scenario["safe"]), "--prior", "pagerank_d085"]
    assert FU.main(args + ["--out", str(tmp_path / "raw_out"), "--mode", "raw"]) == 0
    saved = json.loads((tmp_path / "raw_out" / "fusion_best.json").read_text())
    assert (saved["mode"], saved["score_norm"], saved["prior_transform"]) == ("raw", "none", "log_raw")
    assert FU.main(args + ["--out", str(tmp_path / "bad_out"), "--mode", "raw", "--score-norm", "minmax"]) == 2
    assert "needs score_norm 'none'" in capsys.readouterr().err
