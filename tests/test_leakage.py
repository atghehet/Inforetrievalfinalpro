import pandas as pd
import pytest

from citeds.leakage import assert_no_leakage, build_index_contexts, find_leaking_rows
from citeds.splits import make_queries


def ctx(cid, citing, cited, text, prev="", nxt="", sent=None):
    sent = sent or cid.split(">")[0]
    return {"context_id": cid, "sentence_id": sent, "citing_paper_id": citing, "cited_paper_id": cited,
            "text": text, "prev_text": prev, "next_text": nxt, "section": "", "num_cited_in_sentence": 1}


QUERY = "Adapter layers add small bottleneck modules to pretrained transformers for efficient tuning."


@pytest.fixture
def toy():
    contexts = pd.DataFrame([
        ctx("A#s1>P", "A", "P", QUERY),                                        # the query itself
        ctx("A#s1>Q", "A", "Q", QUERY, sent="A#s1"),                           # same sentence, other cited paper
        ctx("B#s5>P", "B", "P", QUERY.upper().replace(".", "!")),              # exact copy (after normalising)
        ctx("C#s2>P", "C", "P", QUERY.replace("efficient", "cheap")),          # near duplicate
        ctx("D#s3>P", "D", "P", "Adapters are a parameter efficient alternative to full fine tuning."),  # keep
        ctx("E#s9>P", "E", "P", "We follow the adapter recipe of prior work for our experiments.",
            prev=QUERY),                                                       # query leaks via prev_text
        ctx("F#s1>R", "F", "R", QUERY.replace("efficient", "cheap")),          # near dup but cites OTHER paper: keep
    ])
    queries = pd.DataFrame([{"query_id": "q000000", "split": "test", "context_id": "A#s1>P",
                             "sentence_id": "A#s1", "citing_paper_id": "A", "cited_paper_id": "P",
                             "text": QUERY, "cited_citation_count": 5}])
    return contexts, queries


def test_leakage_removes_query_copies_and_near_duplicates(toy):
    contexts, queries = toy
    idx = build_index_contexts(contexts, queries)
    kept = set(idx["context_id"])
    assert kept == {"D#s3>P", "E#s9>P", "F#s1>R"}
    assert idx.loc[idx["context_id"] == "E#s9>P", "prev_text"].item() == ""
    assert_no_leakage(idx, queries)


def test_assert_no_leakage_fails_on_unfiltered_index(toy):
    contexts, queries = toy
    with pytest.raises(AssertionError):
        assert_no_leakage(contexts, queries)


def test_unrelated_text_not_flagged(toy):
    contexts, queries = toy
    leak = find_leaking_rows(contexts, queries)
    assert not leak[contexts["context_id"] == "D#s3>P"].item()


def test_full_pipeline_no_leakage(sample_tables):
    papers, contexts, _, _ = sample_tables
    q = make_queries(contexts, papers, max_queries=500)
    idx = build_index_contexts(contexts, q)
    assert_no_leakage(idx, q)
    assert not set(q["sentence_id"]) & set(idx["sentence_id"])
