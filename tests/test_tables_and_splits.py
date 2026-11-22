import pandas as pd

from citeds import config, schema
from citeds.clean import num_tokens
from citeds.leakage import build_index_contexts
from citeds.splits import make_queries


def test_tables_match_schema(sample_tables):
    papers, contexts, edges, _ = sample_tables
    for df, sch in ((papers, schema.PAPERS), (contexts, schema.CONTEXTS), (edges, schema.EDGES)):
        schema.conform(df, sch)  # raises if columns missing / wrong type


def test_cleaning_rules(sample_tables):
    papers, contexts, edges, _ = sample_tables
    corpus = set(papers["paper_id"])
    assert papers["paper_id"].is_unique and contexts["context_id"].is_unique
    assert (papers["abstract"].map(num_tokens) >= config.MIN_ABSTRACT_TOKENS).all()
    assert contexts["cited_paper_id"].isin(corpus).all()
    assert (contexts["citing_paper_id"] != contexts["cited_paper_id"]).all()
    assert (contexts["text"].map(num_tokens) >= config.MIN_SENTENCE_TOKENS).all()
    assert not contexts["text"].str.contains(r"\{\{|\\[a-z]+|\$", regex=True).any()
    assert not contexts.duplicated(["citing_paper_id", "cited_paper_id", "text"]).any()
    # citation_count == number of distinct citing papers
    cc = edges.groupby("cited_paper_id").size()
    assert (papers.set_index("paper_id")["citation_count"] == cc.reindex(papers["paper_id"]).fillna(0).values).all()


def test_queries_deterministic_and_well_formed(sample_tables):
    papers, contexts, _, _ = sample_tables
    q1 = make_queries(contexts, papers, seed=7, max_queries=400)
    q2 = make_queries(contexts, papers, seed=7, max_queries=400)
    q3 = make_queries(contexts, papers, seed=8, max_queries=400)
    pd.testing.assert_frame_equal(q1, q2)
    assert not q1["context_id"].equals(q3["context_id"])
    schema.conform(q1, schema.QUERIES)
    assert q1["query_id"].is_unique and q1["sentence_id"].is_unique
    assert set(q1["split"]) <= set(config.SPLIT_SIZES)
    # one relevant doc per query
    assert (contexts.set_index("context_id").loc[q1["context_id"], "num_cited_in_sentence"] == 1).all()
    # a citing paper's sentences live in exactly one split
    assert (q1.groupby("citing_paper_id")["split"].nunique() == 1).all()
    assert q1.groupby("cited_paper_id").size().max() <= config.MAX_QUERIES_PER_CITED_PAPER


def test_every_query_target_keeps_index_contexts(sample_tables):
    papers, contexts, _, _ = sample_tables
    q = make_queries(contexts, papers, max_queries=400)
    idx = build_index_contexts(contexts, q)
    remaining = idx.groupby("cited_paper_id").size()
    # build.py drops the (rare) queries failing this after near-dup removal; check most pass.
    ok = q["cited_paper_id"].map(remaining).fillna(0) >= config.MIN_REMAINING_CONTEXTS
    assert ok.mean() > 0.95


def test_build_end_to_end(tmp_path, monkeypatch):
    import make_sample, json
    raw = tmp_path / "raw"
    raw.mkdir()
    with open(raw / "s.jsonl", "w") as fh:
        for r in make_sample.make(150, seed=3):
            fh.write(json.dumps(r) + "\n")
    proc = tmp_path / "processed"
    for name in ("PAPERS", "CONTEXTS", "INDEX_CONTEXTS", "EDGES", "QUERIES"):
        monkeypatch.setattr(config, f"{name}_PATH", proc / getattr(config, f"{name}_PATH").name)
    monkeypatch.setattr(config, "PROCESSED_DIR", proc)
    monkeypatch.setattr(config, "QRELS_DIR", proc / "qrels")
    monkeypatch.setattr(config, "TOPICS_DIR", proc / "topics")
    monkeypatch.setattr(config, "STATS_PATH", tmp_path / "stats.md")
    monkeypatch.setattr(config, "CRAWL_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "EXTERNAL_DIR", tmp_path / "ext")
    from citeds import build, io
    info = build.run(raw, "cs.CL", 10_000, 42, 300)
    assert info["counts"]["queries"] > 0
    qrels = (proc / "qrels" / "qrels.test.txt").read_text().splitlines()
    assert all(len(l.split()) == 4 for l in qrels)
    docs = io.expanded_documents()
    assert len(docs) == info["counts"]["papers"] and docs["text"].str.len().gt(0).all()
    assert "Dataset statistics" in (tmp_path / "stats.md").read_text()
