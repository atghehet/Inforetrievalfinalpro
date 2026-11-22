"""Tests for p4.experiments.key_comparisons, eval_config and crosscheck_check_run. TOY data only (generated scores, DEMO scenario): no project results."""
import json
import textwrap

import numpy as np
import pandas as pd
import pytest

from p4.experiments import crosscheck_check_run as CC
from p4.experiments import eval_config as EC
from p4.experiments import key_comparisons as KC
from p4.tests import graph_fixtures as GF

NAMES = sorted({n for _, s, b in KC.COMPARISONS for n in (s, b)})


def toy_per_query(n=80, lift=None):
    """Every system gets the same noise-free score 0.5 except those in `lift`, which are +0.3 for every query (a clear, uniform improvement)."""
    rng = np.random.default_rng(7)
    base = np.clip(rng.normal(0.4, 0.2, n), 0, 1)
    rows = []
    for name in NAMES:
        score = np.clip(base + (lift or {}).get(name, 0.0), 0, 1)
        for i, v in enumerate(score):
            rows.append({"system": name, "query_id": f"q{i:03d}", "rr@10": v, "recall@10": float(v > 0.45), "ndcg@10": v})
    return pd.DataFrame(rows)


def test_one_row_per_comparison_and_metric_with_holm_over_the_family():
    table = KC.key_comparisons(toy_per_query(lift={"p2_expanded_bm25": 0.3}), n_boot=200)
    assert len(table) == len(KC.COMPARISONS) * 3 == 48 and set(table["metric"]) == {"MRR@10", "Recall@10", "nDCG@10"}
    assert list(table[table.metric == "MRR@10"]["question"]) == [q for q, _, _ in KC.COMPARISONS]
    mrr = table[table.metric == "MRR@10"].set_index(["system", "baseline"])
    row = mrr.loc[("p2_expanded_bm25", "p2_abstract_bm25")]
    assert row["mean_diff"] == pytest.approx(row["mean_system"] - row["mean_baseline"]) and row["mean_diff"] > 0.1 and row["significant_holm"]
    assert row["ci_low"] > 0 and row["p_holm"] <= 1                                              # a uniform lift: the interval excludes zero
    identical = mrr.loc[("p3_minilm_contexts", "p3_minilm_abstract")]
    assert identical["direction"] == "tie" and not identical["significant_holm"]                  # identical systems are never significant
    # Holm cannot lower a p-value: p_holm >= p within the family
    assert (table["p_holm"] >= table["p_value"] - 1e-12).all()


def test_a_small_inconsistent_difference_is_not_called_significant():
    pq = toy_per_query()
    rng = np.random.default_rng(1)
    mask = (pq["system"] == "p3_hybrid_rrf")
    pq.loc[mask, "rr@10"] = np.clip(pq.loc[mask, "rr@10"].to_numpy() + rng.normal(0.0, 0.05, mask.sum()), 0, 1)   # noise, no real effect
    row = KC.key_comparisons(pq, n_boot=200).query("metric == 'MRR@10' and system == 'p3_hybrid_rrf' and baseline == 'p2_expanded_bm25'").iloc[0]
    assert not row["significant_holm"] and row["ci_low"] < 0 < row["ci_high"]


def test_markdown_and_cli(tmp_path, capsys):
    path = tmp_path / "per_query.csv"
    toy_per_query(lift={"p3_hybrid_rrf": 0.3}).to_csv(path, index=False)
    out = tmp_path / "out"
    assert KC.main(["--per-query", str(path), "--out", str(out), "--n-boot", "100", "--note", "TOY"]) == 0
    text = (out / "key_comparisons.md").read_text()
    assert text.startswith("# Pre-specified comparisons") and "TOY" in text and "significant, better" in text and "not significant" in text
    assert len(pd.read_csv(out / "key_comparisons.csv")) == 48 and "significant after Holm" in capsys.readouterr().out
    assert KC.main(["--per-query", str(path), "--out", str(out)]) == 2                              # not empty
    trimmed = tmp_path / "trimmed.csv"
    pd.read_csv(path).query("system != 'p3_hybrid_rrf'").to_csv(trimmed, index=False)
    assert KC.main(["--per-query", str(trimmed), "--out", str(tmp_path / "o2")]) == 2
    assert "p3_hybrid_rrf" in capsys.readouterr().err


# ---- eval_config -----------------------------------------------------------------------------------------------
def test_eval_config_scores_every_system_on_the_chosen_split(tmp_path):
    processed = tmp_path / "processed"
    (processed / "qrels").mkdir(parents=True)
    runs = tmp_path / "runs"
    runs.mkdir()
    for split in ("val", "test"):
        run, qrels = GF.leak_run_and_qrels(split)
        GF.write_run_file(runs / f"a.{split}.run", run, "a")
        GF.write_run_file(runs / f"b.{split}.run", run, "b")
        GF.write_qrels_file(processed / "qrels" / f"qrels.{split}.txt", qrels)
    cfg = {"systems": {s: {"val": str(runs / f"{s}.val.run"), "test": str(runs / f"{s}.test.run")} for s in ("a", "b")}, "baseline": "b"}
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    assert EC.main(["--config", str(path), "--processed", str(processed), "--split", "val", "--out", str(out), "--n-boot", "100", "--note", "DEMO."]) == 0
    summary = pd.read_csv(out / "summary.csv")
    assert list(summary["system"]) == ["a", "b"] and (summary["n_queries"] == 20).all()                # the 20 validation queries of the scenario
    assert "DEMO." in (out / "report.md").read_text() and "split: val" in (out / "report.md").read_text()
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert EC.main(["--config", str(bad), "--processed", str(processed), "--split", "val", "--out", str(tmp_path / "o2")]) == 2
    assert EC.main(["--config", str(tmp_path / "nope.json"), "--processed", str(processed), "--split", "val", "--out", str(tmp_path / "o3")]) == 2


# ---- cross-check against check_run.py (a stand-in script with the same output format; the real one needs the team's package) -------------
FAKE_CHECK_RUN = textwrap.dedent('''
    import sys
    from pathlib import Path
    rows = [l.split() for l in Path(sys.argv[1]).read_text().splitlines()]
    {body}
''')


def write_fake(tmp_path, mrr="0.2500", r10="0.5000", r100="1.0000", ndcg="0.3000", ok=True):
    body = (f"print('run: x  split: val'); print('MRR@10 {mrr}  Recall@10 {r10}  Recall@100 {r100}  nDCG@10 {ndcg}'); print('format OK')"
            if ok else "print('3 FORMAT ERRORS'); sys.exit(1)")
    path = tmp_path / "fake_check_run.py"
    path.write_text(FAKE_CHECK_RUN.format(body=body))
    return path


def demo_run_and_numbers(tmp_path):
    run, qrels = GF.leak_run_and_qrels("val")
    GF.write_run_file(tmp_path / "x.val.run", run, "x")
    GF.write_qrels_file(tmp_path / "qrels.val.txt", qrels)
    from p4.eval.harness import evaluate_runs
    row = evaluate_runs({"x": run}, qrels, (10, 100)).summary.iloc[0]
    return tmp_path / "x.val.run", tmp_path / "qrels.val.txt", row


def test_crosscheck_agrees_when_the_numbers_match_to_four_decimals(tmp_path):
    run, qrels, row = demo_run_and_numbers(tmp_path)
    fake = write_fake(tmp_path, *(f"{row[c]:.4f}" for c in ("MRR@10", "Recall@10", "Recall@100", "nDCG@10")))
    table = CC.crosscheck({"x": run}, qrels, "val", fake, tmp_path, tmp_path)
    assert table[[f"{c} diff" for c in CC.COLUMNS]].to_numpy().max() <= CC.TOLERANCE


def test_crosscheck_reports_a_discrepancy_and_a_failing_script(tmp_path):
    run, qrels, row = demo_run_and_numbers(tmp_path)
    fake = write_fake(tmp_path, mrr=f"{row['MRR@10'] + 0.01:.4f}", r10=f"{row['Recall@10']:.4f}", r100=f"{row['Recall@100']:.4f}", ndcg=f"{row['nDCG@10']:.4f}")
    table = CC.crosscheck({"x": run}, qrels, "val", fake, tmp_path, tmp_path)
    assert table["MRR@10 diff"].iloc[0] == pytest.approx(0.01, abs=1e-4)
    with pytest.raises(RuntimeError, match="check_run.py failed"):
        CC.crosscheck({"x": run}, qrels, "val", write_fake(tmp_path, ok=False), tmp_path, tmp_path)


def test_crosscheck_cli_exit_codes(tmp_path):
    run, qrels, row = demo_run_and_numbers(tmp_path)
    processed = tmp_path / "data" / "processed"
    (processed / "qrels").mkdir(parents=True)
    (processed / "qrels" / "qrels.val.txt").write_text(qrels.read_text())
    cfg = tmp_path / "runs.json"
    cfg.write_text(json.dumps({"systems": {"x": {"val": str(run), "test": str(run)}}}))
    good = write_fake(tmp_path, *(f"{row[c]:.4f}" for c in ("MRR@10", "Recall@10", "Recall@100", "nDCG@10")))
    args = ["--config", str(cfg), "--processed", str(processed), "--check-run", str(good), "--integration-src", str(tmp_path), "--split", "val"]
    assert CC.main(args + ["--out", str(tmp_path / "ok.csv")]) == 0 and (tmp_path / "ok.csv").exists()
    assert CC.main(args + ["--out", str(tmp_path / "ok.csv")]) == 2                                     # never overwrites
    bad = write_fake(tmp_path, mrr="0.9999", r10="0.9999", r100="0.9999", ndcg="0.9999")
    args[args.index(str(good))] = str(bad)
    assert CC.main(args + ["--out", str(tmp_path / "bad.csv")]) == 2


# ---- graph_facts (DEMO leak scenario) -------------------------------------------------------------------------------
def test_graph_facts_on_the_demo_scenario(tmp_path):
    from p4.experiments import graph_facts as GFX
    from p4.graph import eval_graph as EG
    processed = GF.leak_scenario(tmp_path / "processed")
    graph = tmp_path / "graph"
    EG.build(processed, graph, data_label="demo")
    facts = GFX.graph_facts(processed, graph)
    papers = pd.read_parquet(processed / "papers.parquet")
    assert facts["nodes"] == len(papers) and facts["data_label"] == "demo"
    assert facts["spearman_pagerank_d085_vs_safe_count"] is None                      # both columns are constant in this scenario: undefined, not NaN
    assert facts["edge_removal"]["edges_kept"] + facts["edge_removal"]["edges_removed"] == facts["edges_full"]
    assert facts["test_targets"]["full_count_zero"] == 0.0 and facts["test_targets"]["safe_count_zero"] == 1.0   # every target is cited only by its own query's paper
    assert facts["papers_never_cited_safe_graph"] == 1.0 and len(facts["top10_pagerank_d085"]) == 10
    assert facts["direction_check"] == {"edges_with_both_years": 0, "citing_year_>=_cited_year": None, "citing_year_<_cited_year": None}   # no years in the fixture
    out = tmp_path / "facts.json"
    assert GFX.main(["--processed", str(processed), "--graph-dir", str(graph), "--out", str(out)]) == 0 and json.loads(out.read_text())["nodes"] == facts["nodes"]
    assert GFX.main(["--processed", str(processed), "--graph-dir", str(graph), "--out", str(out)]) == 2                # never overwrites
    assert GFX.main(["--processed", str(tmp_path / "none"), "--graph-dir", str(graph), "--out", str(tmp_path / "f2.json")]) == 2


# ---- leak B diagnostic and the --only-queries subset (TOY tables) ----------------------------------------------------
def toy_processed(tmp_path):
    from p4.experiments import leak_b_diagnostic  # noqa: F401  (import check)
    processed = tmp_path / "processed"
    processed.mkdir()
    # 4 test queries: q1/q2 have another sentence of the same paper citing the same target; q3 has a same-paper sentence citing ANOTHER target; q4 has none
    queries = pd.DataFrame({"query_id": ["q1", "q2", "q3", "q4", "q5"], "split": ["test", "test", "test", "test", "val"],
                            "citing_paper_id": ["c1", "c2", "c3", "c4", "c5"], "cited_paper_id": ["t1", "t2", "t3", "t4", "t5"],
                            "context_id": ["x1", "x2", "x3", "x4", "x5"]})
    index = pd.DataFrame({"context_id": ["y1", "y2", "y3", "y4"], "citing_paper_id": ["c1", "c2", "c3", "c9"], "cited_paper_id": ["t1", "t2", "t99", "t4"]})
    queries.to_parquet(processed / "queries.parquet", index=False)
    index.to_parquet(processed / "index_contexts.parquet", index=False)
    return processed


def test_query_flags_distinguish_same_target_same_paper_from_same_paper_only(tmp_path):
    from p4.experiments import leak_b_diagnostic as LB
    flags = LB.query_flags(toy_processed(tmp_path), "test").set_index("query_id")
    assert flags["same_paper_same_target"].to_dict() == {"q1": True, "q2": True, "q3": False, "q4": False}
    assert flags["same_paper_any"].to_dict() == {"q1": True, "q2": True, "q3": True, "q4": False}                 # q4's paper only appears with another citing paper c9
    assert "q5" not in flags.index                                                                             # the validation query is not in the test flags


def test_diagnostic_compares_flagged_and_unflagged_queries_and_skips_tiny_groups():
    from p4.experiments import leak_b_diagnostic as LB
    rng = np.random.default_rng(3)
    qids = [f"q{i:03d}" for i in range(60)]
    flags = pd.DataFrame({"query_id": qids, "same_paper_same_target": [i < 30 for i in range(60)], "same_paper_any": [i < 55 for i in range(60)]})
    rows = []
    for system, bonus in (("uses_contexts", 0.3), ("abstract_only", 0.0)):
        for i, q in enumerate(qids):
            v = float(np.clip(0.3 + (bonus if i < 30 else 0.0) + rng.normal(0, 0.05), 0, 1))
            rows.append({"system": system, "query_id": q, "rr@10": v, "recall@10": float(v > 0.4)})
    table = LB.diagnostic(pd.DataFrame(rows), flags, ["uses_contexts", "abstract_only"], n_boot=200)
    cell = lambda s, f, m="MRR@10": table[(table.system == s) & (table.flag == f) & (table.metric == m)].iloc[0]
    assert cell("uses_contexts", "same_paper_same_target")["diff"] == pytest.approx(0.3, abs=0.05) and cell("uses_contexts", "same_paper_same_target")["ci_low"] > 0.2
    assert abs(cell("abstract_only", "same_paper_same_target")["diff"]) < 0.05                                   # the control does not move with the flag
    small = cell("uses_contexts", "same_paper_any")                                                              # only 5 queries without the flag: not compared
    assert small["n_without"] == 5 and np.isnan(small["diff"]) and not np.isnan(small["mean_without"])


def test_leak_b_cli_writes_the_id_lists_and_key_comparisons_can_use_them(tmp_path, capsys):
    from p4.experiments import leak_b_diagnostic as LB
    processed = toy_processed(tmp_path)
    pq = toy_per_query(n=4)
    pq["query_id"] = pq["query_id"].map({"q000": "q1", "q001": "q2", "q002": "q3", "q003": "q4"})
    pq_path = tmp_path / "per_query.csv"
    pq.to_csv(pq_path, index=False)
    out = tmp_path / "lb"
    assert LB.main(["--processed", str(processed), "--per-query", str(pq_path), "--out", str(out), "--n-boot", "100"]) == 0
    assert (out / "queries_with_same_paper_same_target.txt").read_text().split() == ["q1", "q2"]
    assert (out / "queries_without_same_paper_same_target.txt").read_text().split() == ["q3", "q4"]
    assert "2 with another sentence of the same paper citing the same target" in capsys.readouterr().out
    assert LB.main(["--processed", str(processed), "--per-query", str(pq_path), "--out", str(out)]) == 2             # not empty
    # key_comparisons restricted to a list of query ids
    big = tmp_path / "big.csv"
    toy_per_query(n=60, lift={"p2_expanded_bm25": 0.3}).to_csv(big, index=False)
    keep = tmp_path / "keep.txt"
    keep.write_text("".join(f"q{i:03d}" + chr(10) for i in range(40)))
    res = tmp_path / "sub"
    assert KC.main(["--per-query", str(big), "--only-queries", str(keep), "--out", str(res), "--n-boot", "100"]) == 0
    assert "Subset: 40 queries" in (res / "key_comparisons.md").read_text()
    assert set(pd.read_csv(res / "key_comparisons.csv")["n_queries"]) == {40}
    keep.write_text("q000" + chr(10) + "q-not-there" + chr(10))
    assert KC.main(["--per-query", str(big), "--only-queries", str(keep), "--out", str(tmp_path / "bad")]) == 2
    assert "must list queries present" in capsys.readouterr().err


def test_by_bucket_reports_exposure_and_scores_per_citation_bucket(tmp_path):
    from p4.experiments import leak_b_diagnostic as LB
    processed = toy_processed(tmp_path)                                                                    # test queries q1..q4 target t1..t4
    priors = pd.DataFrame({"paper_id": ["t1", "t2", "t3", "t4"], "citation_count_eval": [0, 0, 5, 1]})   # buckets: 0, 0, 3-10, 1-2
    pq = pd.DataFrame({"system": ["s"] * 4, "query_id": ["q1", "q2", "q3", "q4"], "rr@10": [1.0, 0.5, 0.2, 0.1], "recall@10": [1.0] * 4})
    flags = LB.query_flags(processed, "test")
    exposure, cells = LB.by_bucket(processed, "test", flags, priors, pq, ["s"])
    e = exposure.set_index("bucket")
    assert list(e.index) == ["0", "1-2", "3-10", "11-50", ">50"] and e.loc["0", "n_queries"] == 2 and e.loc["0", "share_same_paper_same_target"] == 1.0   # q1, q2 both exposed
    assert e.loc["3-10", "n_same_paper_same_target"] == 0 and e.loc["1-2", "n_queries"] == 1 and e.loc["11-50", "n_queries"] == 0
    zero = cells[(cells["bucket"] == "0") & cells["same_paper_same_target_in_index"]].iloc[0]
    assert zero["n_queries"] == 2 and zero["MRR@10"] == pytest.approx(0.75)
    out = tmp_path / "lb2"
    pq_path = tmp_path / "pq.csv"
    pq.to_csv(pq_path, index=False)
    priors_path = tmp_path / "priors.parquet"
    priors.to_parquet(priors_path, index=False)
    assert LB.main(["--processed", str(processed), "--per-query", str(pq_path), "--priors", str(priors_path), "--out", str(out), "--n-boot", "100"]) == 0
    assert (out / "exposure_by_bucket.csv").exists() and (out / "mrr_by_bucket_and_exposure.csv").exists()
