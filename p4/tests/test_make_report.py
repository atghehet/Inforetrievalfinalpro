"""Tests for p4.experiments.make_report and analysis_tables. TOY tables only: no project results."""
import json

import numpy as np
import pandas as pd
import pytest

from p4.experiments import analysis_tables as AT
from p4.experiments import make_report as MR


def test_df_to_md_formats_numbers_text_and_missing_values():
    df = pd.DataFrame({"system": ["a", "b"], "n": [3, 4], "score": [0.123456, np.nan]})
    assert MR.df_to_md(df) == "| system | n | score |\n|---|---:|---:|\n| a | 3 | 0.1235 |\n| b | 4 |  |\n"
    assert MR.df_to_md(df, "{:.1f}").splitlines()[2].endswith("| 0.1 |")


def test_fill_replaces_every_placeholder_and_refuses_unknown_ones():
    text = "before" + chr(10) + "{{table:one}}" + chr(10) + "middle" + chr(10) + "{{table:two}}" + chr(10) + "after" + chr(10)     # tables sit on their own lines
    filled = MR.fill(text, {"one": "| a |" + chr(10), "two": "X"})
    assert filled == chr(10).join(["before", "| a |", "", "middle", "X", "", "after", ""])                      # a table, then the text's own line break: one blank line follows it
    with pytest.raises(MR.ReportError, match=r"\['missing'\]"):
        MR.fill("{{table:one}} {{table:missing}}", {"one": "x"})
    assert MR.fill("no placeholders", {}) == "no placeholders"


def test_main_errors_are_one_line_and_never_overwrite(tmp_path, capsys):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (out / "old.md").write_text("keep")
    assert MR.main(["--real-dir", str(tmp_path), "--src", str(src), "--out", str(out)]) == 2 and "not empty" in capsys.readouterr().err
    assert (out / "old.md").read_text() == "keep"
    assert MR.main(["--real-dir", str(tmp_path / "nothing"), "--src", str(src), "--out", str(tmp_path / "new")]) == 2
    assert "missing input file" in capsys.readouterr().err                                         # the real-dir does not exist: reported, not a traceback


def test_group_table_orders_buckets_and_marks_small_groups():
    rows = []
    for system, base in (("p2_abstract_bm25", 0.2), ("p2_expanded_bm25", 0.4)):
        for group, n in (("0", 100), ("1-2", 50), (">50", 5)):
            rows.append({"system": system, "group": group, "metric": "rr@10", "n_queries": n, "small_group": n < 10, "mean": base + n / 1000})
    text = AT.group_table(pd.DataFrame(rows), "rr@10", AT.BUCKETS)
    header = text.splitlines()[0]
    assert header.index("0 (n=100)") < header.index("1-2 (n=50)") < header.index(">50 (n=5, small)") and "3-10" not in header   # absent bucket: no column
    assert "| p2_abstract_bm25 | 0.300 | 0.250 | 0.205 |" in text and "p3_hybrid_rrf" not in text                          # only systems that exist


def test_tables_are_skipped_when_their_inputs_are_absent(tmp_path):
    assert AT.build(tmp_path) == {}                                                                  # nothing to build: no tables, no crash
    folder = tmp_path / "sensitivity_extended_lambda_val" / "p2_abstract_bm25"
    folder.mkdir(parents=True)
    pd.DataFrame({"prior": ["citation_count_eval"] * 3, "lambda": [0.0, 0.3, 100.0], "MRR@10": [0.25, 0.28, 0.09]}).to_csv(folder / "fusion_tuning.csv", index=False)
    assert AT.build(tmp_path) == {}                                                                  # the second system's file is missing: skipped as a whole
