"""Tests for p4.eval.report (Markdown generation)."""
import pandas as pd

from p4.eval.efficiency import summarise
from p4.eval.harness import evaluate_runs, per_query_column, summary_name
from p4.eval.report import build_report, markdown_table
from p4.eval.significance import compare_result
import pytest

QRELS = {f"q{i}": {f"D{i}": 1} for i in range(1, 9)}
GOOD = {q: [(f"D{q[1:]}", 1.0)] for q in QRELS}                              # relevant paper at rank 1
PARTIAL = {q: v for q, v in GOOD.items() if int(q[1:]) <= 5}                 # misses 3 judged queries


def build(note=None, efficiency=None):
    result = evaluate_runs({"good": GOOD, "partial": PARTIAL}, QRELS, cutoffs=(10,))
    comparisons = compare_result(result, ["rr@10"], baseline="good", n_boot=200)
    return build_report(result=result, comparisons=comparisons, settings={"baseline": "good", "seed": 42},
                        headline_k=10, efficiency=efficiency, note=note)


def test_markdown_table_formats_numbers_nan_booleans_and_pipes():
    df = pd.DataFrame({"name": ["a|b", "c"], "score": [0.123456, float("nan")], "ok": [True, False], "n": [3, 4]})
    lines = markdown_table(df).splitlines()
    assert lines[0] == "| name | score | ok | n |"
    assert lines[1] == "|---|---|---|---|"
    assert lines[2] == "| a/b | 0.1235 | yes | 3 |"                           # pipe replaced, 4 decimals
    assert lines[3] == "| c | n/a | no | 4 |"


def test_metric_name_helpers_accept_both_styles():
    assert per_query_column("MRR@10") == "rr@10" and per_query_column("rr@10") == "rr@10"
    assert per_query_column("nDCG@5") == "ndcg@5" and per_query_column("Recall@100") == "recall@100"
    assert per_query_column("MAP") == "ap" and per_query_column("ap") == "ap"
    assert summary_name("rr@10") == "MRR@10" and summary_name("ap") == "MAP"
    for bad in ("MRR", "foo@10", "MRR@ten", ""):
        with pytest.raises(ValueError, match="unknown metric"):
            per_query_column(bad)


def test_report_contains_every_section_and_the_numbers():
    text = build(note="SYNTHETIC data, not real results")
    for heading in ("# Evaluation report", "## Inputs", "## Summary metrics (cutoff 10)",
                    "## Query coverage warnings", "## Statistical comparisons", "### MRR@10",
                    "## Efficiency (from P2's logs)", "## How to read this report"):
        assert heading in text
    assert "> **Note:** SYNTHETIC data, not real results" in text
    assert "| good | 8 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |" in text   # perfect system
    assert "| partial | 8 | 3 |" in text and "3 judged queries had no results" in text
    assert "never from the averages alone" in text                            # the significance caveat is always printed


def test_report_says_so_when_there_is_nothing_to_compare_or_no_efficiency_data():
    result = evaluate_runs({"only": GOOD}, QRELS, cutoffs=(10,))
    comparisons = compare_result(result, [], baseline="only")
    text = build_report(result=result, comparisons=comparisons, settings={"baseline": "only"}, headline_k=10)
    assert "Not computed (needs at least two systems)" in text and "Not requested" in text


def test_report_includes_efficiency_tables_and_the_not_recorded_notice(tmp_path):
    (tmp_path / "e.csv").write_text("index,build_seconds,bytes\nctx,1.5,1048576\n")
    (tmp_path / "t.csv").write_text("run,variant,model,parameters,validation_mrr_at_10,median_ms,mean_ms\n"
                                    "ctx-bm25-0,context,bm25,{},0.3,2.0,2.5\n")
    text = build(efficiency=summarise(tmp_path / "e.csv", tmp_path / "t.csv"))
    assert "Index build cost" in text and "| ctx | 1.5000 | 1.0000 |" in text
    assert "memory usage" in text and "Not recorded" in text
    assert "| context | bm25 | 0.3000 | 2.0000 | 2.5000 |" in text
