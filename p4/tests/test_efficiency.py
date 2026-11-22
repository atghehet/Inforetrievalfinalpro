"""Tests for p4.eval.efficiency (reading and joining P2's efficiency.csv and tuning.csv)."""
from pathlib import Path

import pytest

from p4.eval.efficiency import EfficiencyFormatError, read_efficiency, read_tuning, summarise

REPO = Path(__file__).resolve().parent.parent.parent
P2_QUICK = REPO / "p3" / "outputs" / "p2_quick"          # P2's logs as committed by P3

EFFICIENCY = "index,build_seconds,bytes\nabstract-w0-plain-all,0.5,1048576\ncontext-w0-plain-all,1.5,2097152\n"
TUNING = (
    "run,variant,model,parameters,validation_mrr_at_10,median_ms,mean_ms\n"
    'abstract-w0-plain-all-bm25-0,abstract,bm25,"{""k1"": 0.8, ""b"": 0.4}",0.10,2.0,2.5\n'
    'abstract-w0-plain-all-bm25-1,abstract,bm25,"{""k1"": 1.2, ""b"": 0.4}",0.30,3.0,3.5\n'
    "abstract-w0-plain-all-tfidf-0,abstract,tfidf,{},0.20,1.0,1.5\n"
    'context-w0-plain-all-rm3-0,context,rm3,"{""feedback_docs"": 3}",0.40,9.0,9.5\n')


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_read_efficiency_adds_size_in_mib(tmp_path):
    df = read_efficiency(write(tmp_path, "efficiency.csv", EFFICIENCY))
    assert list(df.columns) == ["index", "build_seconds", "bytes", "size_mib"]
    assert df["size_mib"].tolist() == [1.0, 2.0]                          # 1048576 bytes = exactly 1 MiB
    assert df["build_seconds"].tolist() == [0.5, 1.5]


def test_read_tuning_derives_the_index_name_from_the_run_name(tmp_path):
    df = read_tuning(write(tmp_path, "tuning.csv", TUNING))
    assert df["index"].tolist() == ["abstract-w0-plain-all"] * 3 + ["context-w0-plain-all"]
    assert df.loc[0, "parameters"] == '{"k1": 0.8, "b": 0.4}'              # JSON with commas survives the CSV parsing


def test_summarise_joins_build_cost_to_each_tuned_run(tmp_path):
    s = summarise(write(tmp_path, "e.csv", EFFICIENCY), write(tmp_path, "t.csv", TUNING))
    runs = s.runs.set_index("run")
    assert runs.loc["abstract-w0-plain-all-bm25-1", "build_seconds"] == 0.5
    assert runs.loc["context-w0-plain-all-rm3-0", "size_mib"] == 2.0
    assert runs.loc["context-w0-plain-all-rm3-0", "median_ms"] == 9.0      # query time is copied, not recomputed
    assert len(s.runs) == 4


def test_best_configuration_per_variant_and_model_is_chosen_by_validation_mrr(tmp_path):
    best = summarise(write(tmp_path, "e.csv", EFFICIENCY), write(tmp_path, "t.csv", TUNING)).best_per_variant
    picked = {(r.variant, r.model): r.run for r in best.itertuples()}
    assert picked[("abstract", "bm25")] == "abstract-w0-plain-all-bm25-1"  # 0.30 beats 0.10
    assert len(best) == 3                                                   # (abstract,bm25) (abstract,tfidf) (context,rm3)


def test_summary_states_what_is_not_recorded(tmp_path):
    s = summarise(write(tmp_path, "e.csv", EFFICIENCY))
    assert s.runs is None and s.best_per_variant is None                   # nothing invented without tuning.csv
    text = " ".join(s.notes)
    assert "memory usage" in text and "Not recorded" in text
    assert "machine" in text


def test_runs_without_a_matching_index_are_kept_with_blank_cost_and_a_note(tmp_path):
    only_abstract = "index,build_seconds,bytes\nabstract-w0-plain-all,0.5,1048576\n"
    s = summarise(write(tmp_path, "e.csv", only_abstract), write(tmp_path, "t.csv", TUNING))
    row = s.runs.set_index("run").loc["context-w0-plain-all-rm3-0"]
    assert row[["build_seconds", "bytes", "size_mib"]].isna().all()
    assert any("context-w0-plain-all" in n for n in s.notes)


def test_missing_file_gives_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="efficiency file not found"):
        read_efficiency(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="tuning file not found"):
        read_tuning(tmp_path / "nope.csv")


@pytest.mark.parametrize("text, message", [
    ("index,build_seconds\na,1.0\n", r"missing column\(s\) \['bytes'\]"),     # a column P2 always writes is gone
    ("index,build_seconds,bytes\n", "no data rows"),
    ("", "cannot read as CSV"),
    ("index,build_seconds,bytes\na,fast,100\n", "non-negative numbers"),
    ("index,build_seconds,bytes\na,-1.0,100\n", "non-negative numbers"),
    ("index,build_seconds,bytes\na,1.0,100\na,2.0,200\n", "duplicate index"),
])
def test_malformed_efficiency_csv_gives_a_clear_error(tmp_path, text, message):
    with pytest.raises(EfficiencyFormatError, match=message):
        read_efficiency(write(tmp_path, "bad.csv", text))


def test_tuning_csv_with_wrong_columns_is_rejected(tmp_path):
    with pytest.raises(EfficiencyFormatError, match="missing column"):
        read_tuning(write(tmp_path, "bad.csv", "run,variant\na,b\n"))


@pytest.mark.skipif(not (P2_QUICK / "efficiency.csv").exists(), reason="P2's committed logs are not present")
def test_real_p2_logs_from_the_repository_are_readable_and_fully_joined():
    s = summarise(P2_QUICK / "efficiency.csv", P2_QUICK / "tuning.csv")
    assert len(s.index_cost) == 4 and len(s.runs) == 13                    # the four index variants, 13 tuned configs
    assert s.runs["build_seconds"].notna().all()                           # every run found its index
    assert not any("not in efficiency.csv" in n for n in s.notes)
    best = {(r.variant, r.model): r.run for r in s.best_per_variant.itertuples()}
    assert best[("context", "bm25")] == "context-w0-plain-all-bm25-0"
