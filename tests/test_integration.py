"""Integration tests: P1 -> P2 export, and P3's shared evaluation/fusion helpers."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "p3" / "src"))


def _build_sample(tmp_path, monkeypatch):
    import make_sample
    from citeds import build, config
    raw = tmp_path / "raw"
    raw.mkdir()
    with open(raw / "s.jsonl", "w") as fh:
        for r in make_sample.make(150, seed=3):
            fh.write(json.dumps(r) + "\n")
    proc = tmp_path / "processed"
    for name in ("PAPERS", "CONTEXTS", "INDEX_CONTEXTS", "EDGES", "QUERIES", "CONTEXT_FUNCTIONS"):
        monkeypatch.setattr(config, f"{name}_PATH", proc / getattr(config, f"{name}_PATH").name)
    for attr, val in (("PROCESSED_DIR", proc), ("QRELS_DIR", proc / "qrels"), ("TOPICS_DIR", proc / "topics"),
                      ("STATS_PATH", tmp_path / "stats.md"), ("CRAWL_CACHE_DIR", tmp_path / "cache"),
                      ("RAW_DIR", raw), ("EXTERNAL_DIR", tmp_path / "ext"), ("DATA_DIR", tmp_path)):
        monkeypatch.setattr(config, attr, val)
    build.run(raw, "cs.CL", 10_000, 42, 300)
    return proc


def test_export_p2_format(tmp_path, monkeypatch):
    import pandas as pd
    from citeds import config, export_p2, schema
    proc = _build_sample(tmp_path, monkeypatch)
    ctx = pd.read_parquet(config.CONTEXTS_PATH, columns=["context_id"])
    labels = pd.DataFrame({"context_id": ctx["context_id"], "function": "method", "confidence": 0.9})
    schema.conform(labels, schema.CONTEXT_FUNCTIONS)
    labels.to_parquet(config.CONTEXT_FUNCTIONS_PATH)
    out = tmp_path / "p2_input"
    counts = export_p2.export(out)
    rows = [json.loads(l) for l in open(out / "contexts.jsonl")]
    assert counts["contexts"] == len(rows) > 0
    r = rows[0]
    assert set(r) >= {"context_id", "cited_paper_id", "sentence", "before", "after", "label"}
    assert r["label"] == "method" and isinstance(r["before"], list) and len(r["before"]) <= 2
    assert any(len(x["before"]) == 2 for x in rows)          # two-sentence windows are present
    papers = {json.loads(l)["paper_id"] for l in open(out / "papers.jsonl")}
    assert {x["cited_paper_id"] for x in rows} <= papers
    for split in ("validation", "test"):
        qids = {json.loads(l)["query_id"] for l in open(out / f"{split}.jsonl")}
        judged = {l.split()[0] for l in open(out / f"{split}.qrels")}
        assert judged == qids


def test_p3_evaluate_and_rrf():
    from common import evaluate, rrf
    rel = {"q1": {"A"}, "q2": {"B"}, "q3": {"C"}}
    run = {"q1": {"A": 1, "X": 2}, "q2": {"X": 1, "B": 2}}      # q3 missing -> counts as 0
    m = evaluate(run, rel)
    assert abs(m["MRR@10"] - (1 + 0.5) / 3) < 1e-9 and m["Recall@10"] == 2 / 3
    fused = rrf([{"q": {"A": 1, "B": 2}}, {"q": {"B": 1, "C": 2}}], k=60)
    assert [d for d, _ in fused["q"]][0] == "B"               # B ranks well in both runs
