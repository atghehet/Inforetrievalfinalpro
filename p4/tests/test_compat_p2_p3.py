"""Compatibility regression: the P4 tools must reproduce the numbers P2 and P3 already published.

The committed P2/P3 results were produced on P1's SYNTHETIC sample (random text from scripts/make_sample.py),
so nothing here says anything about real retrieval quality. What it proves is that P4's readers, metrics,
harness and CLI agree with P2's and P3's own code on real pipeline output files.

The qrels for that synthetic sample are not committed, so they are regenerated here with P1's deterministic
code, read-only (`make_sample.make(400, seed=0)`, `max_queries=200`). That recipe gives P3's exact split
(330 papers, 145/23/32 train/val/test queries); the test skips if the committed files are absent.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

from p4.eval.cli import main
from p4.eval.harness import evaluate_runs
from p4.eval.trec_io import read_qrels, read_run

REPO = Path(__file__).resolve().parent.parent.parent
P3 = REPO / "p3" / "outputs"
P2_QUICK = P3 / "p2_quick"

pytestmark = pytest.mark.skipif(not (P2_QUICK / "tuning.csv").exists() or not (P3 / "hybrid_test.run").exists(),
                                reason="P2/P3 committed outputs are not present")


@pytest.fixture(scope="module")
def synthetic_qrels(tmp_path_factory):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import make_sample
        from citeds import config
        from citeds.leakage import build_index_contexts
        from citeds.splits import assign_query_ids, make_queries
        from citeds.tables import build_tables
        from citeds.unarxive import parse_record
    except ImportError as err:                                   # pragma: no cover
        pytest.skip(f"P1 code not importable: {err}")
    records = make_sample.make(400, seed=0)
    parsed = [parse_record(r) for r in records if "cs.CL" in r["metadata"]["categories"].split()]
    papers, contexts, _edges, _log = build_tables(parsed)
    queries = make_queries(contexts, papers, seed=config.SEED, max_queries=200)
    remaining = build_index_contexts(contexts, queries).groupby("cited_paper_id").size()
    keep = queries["cited_paper_id"].map(remaining).fillna(0) >= config.MIN_REMAINING_CONTEXTS
    queries = assign_query_ids(queries[keep])
    assert queries["split"].value_counts().to_dict() == {"train": 145, "test": 32, "val": 23}   # P3's split
    folder = tmp_path_factory.mktemp("synthetic_qrels")
    paths = {}
    for split in ("val", "test"):
        part = queries[queries["split"] == split]
        paths[split] = folder / f"qrels.{split}.txt"
        paths[split].write_text("".join(f"{q} 0 {d} 1\n" for q, d in zip(part["query_id"], part["cited_paper_id"])))
    return paths


def test_reproduces_p3_published_test_metrics(synthetic_qrels):
    published = pd.read_csv(P3 / "hybrid_test_metrics.csv").set_index("system")
    runs = {"Dense": read_run(P3 / "dense_test.run"), "BM25": read_run(P2_QUICK / "fusion" / "bm25.test.run"),
            "Hybrid RRF k=20": read_run(P3 / "hybrid_test.run")}
    ours = evaluate_runs(runs, read_qrels(synthetic_qrels["test"]), cutoffs=(10,)).summary.set_index("system")
    for system in runs:
        assert ours.loc[system, "n_queries"] == published.loc[system, "queries"] == 32
        for ours_col, p3_col in (("Recall@10", "Recall@10"), ("MRR@10", "MRR@10"), ("nDCG@10", "nDCG@10")):
            assert round(ours.loc[system, ours_col], 4) == published.loc[system, p3_col], (system, ours_col)


def test_reproduces_p3_published_validation_metrics(synthetic_qrels):
    qrels = read_qrels(synthetic_qrels["val"])
    dense = evaluate_runs({"d": read_run(P3 / "dense_val.run")}, qrels, cutoffs=(10,)).summary.iloc[0]
    published = pd.read_csv(P3 / "dense_metrics.csv").set_index("split").loc["val"]
    assert [round(dense[c], 4) for c in ("Recall@10", "MRR@10", "nDCG@10")] == \
        [published["Recall@10"], published["MRR@10"], published["nDCG@10"]]
    hybrid = evaluate_runs({"h": read_run(P3 / "hybrid_validation.run")}, qrels, cutoffs=(10,)).summary.iloc[0]
    k20 = pd.read_csv(P3 / "hybrid_validation_metrics.csv").set_index("k").loc[20]
    assert [round(hybrid[c], 4) for c in ("Recall@10", "MRR@10", "nDCG@10")] == \
        [k20["Recall@10"], k20["MRR@10"], k20["nDCG@10"]]


def test_reproduces_every_p2_validation_mrr_in_tuning_csv(synthetic_qrels):
    qrels = read_qrels(synthetic_qrels["val"])
    tuning = pd.read_csv(P2_QUICK / "tuning.csv")
    assert len(tuning) == 13
    for row in tuning.itertuples():
        run = read_run(P2_QUICK / "validation" / f"{row.run}.run")
        mrr = evaluate_runs({"x": run}, qrels, cutoffs=(10,)).summary.loc[0, "MRR@10"]
        assert mrr == pytest.approx(row.validation_mrr_at_10, abs=1e-12), row.run


def test_cli_evaluates_every_committed_test_run_together(synthetic_qrels, tmp_path):
    out = tmp_path / "out"
    code = main(["--qrels", str(synthetic_qrels["test"]), "--out", str(out), "--quiet", "--n-boot", "300",
                 "--run", f"dense={P3 / 'dense_test.run'}", "--run", f"hybrid={P3 / 'hybrid_test.run'}",
                 "--runs-glob", str(P2_QUICK / "test" / "*.run"),
                 "--baseline", "abstract_bm25", "--efficiency", str(P2_QUICK / "efficiency.csv"),
                 "--tuning", str(P2_QUICK / "tuning.csv"), "--note", "SYNTHETIC sample data"])
    assert code == 0
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 15 and (summary["n_queries"] == 32).all()           # 2 + 13 systems, same 32 queries each
    assert (summary["n_no_results"] == 0).all()                                # no system returned nothing for any query
    comparisons = pd.read_csv(out / "comparisons.csv")
    assert len(comparisons) == 14 * 3 and set(comparisons["baseline"]) == {"abstract_bm25"}
    assert comparisons["p_value"].between(0, 1).all() and comparisons["p_holm"].between(0, 1).all()
    assert (out / "efficiency_runs.csv").exists() and "SYNTHETIC sample data" in (out / "report.md").read_text(encoding="utf-8")
