"""Build the core tables (papers, contexts, edges) from parsed unarXive papers."""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from . import config
from .clean import clean_text, normalise_for_match, num_tokens
from .unarxive import RawPaper

log = logging.getLogger(__name__)


def build_tables(papers: Iterable[RawPaper], max_papers: int = config.MAX_PAPERS
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return (papers_df, contexts_df, edges_df, cleaning_log).

    The corpus is the set of in-field papers with a usable abstract. A citation
    context is kept only if its cited paper is in the corpus (otherwise it has
    no document to retrieve).
    """
    clog = {
        "papers_seen": 0, "papers_no_abstract": 0, "papers_kept": 0,
        "citing_sentences_seen": 0, "contexts_cited_outside_corpus": 0,
        "contexts_self_citation": 0, "contexts_too_short": 0, "contexts_too_long": 0,
        "contexts_duplicate": 0, "contexts_kept": 0,
    }
    paper_rows = []
    raw_ctx = []  # (citing, sentence_number, cited, text, prev, next, section, n_cited_raw)
    for p in papers:
        clog["papers_seen"] += 1
        abstract = clean_text(p.abstract)
        if num_tokens(abstract) < config.MIN_ABSTRACT_TOKENS or not p.title:
            clog["papers_no_abstract"] += 1
            continue
        if len(paper_rows) >= max_papers:
            break
        paper_rows.append({
            "paper_id": p.paper_id, "title": clean_text(p.title), "abstract": abstract,
            "year": p.year, "categories": p.categories,
        })
        for para in p.paragraphs:
            cleaned = [clean_text(s.text) for s in para]
            for i, s in enumerate(para):
                if not s.cited_arxiv_ids:
                    continue
                clog["citing_sentences_seen"] += 1
                prev_t = cleaned[i - 1] if i > 0 else ""
                next_t = cleaned[i + 1] if i + 1 < len(para) else ""
                for cited in s.cited_arxiv_ids:
                    raw_ctx.append((p.paper_id, s.sentence_number, cited, cleaned[i],
                                    prev_t, next_t, s.section))
    papers_df = pd.DataFrame(paper_rows).drop_duplicates("paper_id").reset_index(drop=True)
    clog["papers_kept"] = len(papers_df)
    corpus = set(papers_df["paper_id"])

    ctx = pd.DataFrame(raw_ctx, columns=["citing_paper_id", "sentence_number", "cited_paper_id",
                                         "text", "prev_text", "next_text", "section"])
    n0 = len(ctx)
    ctx = ctx[ctx["cited_paper_id"].isin(corpus)]
    clog["contexts_cited_outside_corpus"] = n0 - len(ctx)
    n0 = len(ctx)
    ctx = ctx[ctx["citing_paper_id"] != ctx["cited_paper_id"]]
    clog["contexts_self_citation"] = n0 - len(ctx)
    ntok = ctx["text"].map(num_tokens)
    clog["contexts_too_short"] = int((ntok < config.MIN_SENTENCE_TOKENS).sum())
    clog["contexts_too_long"] = int((ntok > config.MAX_SENTENCE_TOKENS).sum())
    ctx = ctx[(ntok >= config.MIN_SENTENCE_TOKENS) & (ntok <= config.MAX_SENTENCE_TOKENS)]
    # Deduplicate: the same sentence repeated within a citing paper for the same cited paper.
    n0 = len(ctx)
    ctx = ctx.assign(_norm=ctx["text"].map(normalise_for_match))
    ctx = ctx.drop_duplicates(["citing_paper_id", "cited_paper_id", "_norm"]).drop(columns="_norm")
    clog["contexts_duplicate"] = n0 - len(ctx)
    if ctx.empty:
        raise ValueError(
            f"No citation contexts left: none of the {clog['citing_sentences_seen']} citing sentences "
            f"cites another paper in this corpus of {len(papers_df)} papers. The corpus is too small "
            f"(a sample or a single shard) - add more unarXive files. Cleaning log: {clog}")

    ctx["sentence_id"] = ctx["citing_paper_id"] + "#s" + ctx["sentence_number"].map("{:04d}".format)
    ctx["context_id"] = ctx["sentence_id"] + ">" + ctx["cited_paper_id"]
    ctx["num_cited_in_sentence"] = ctx.groupby("sentence_id")["cited_paper_id"].transform("size").astype("int32")
    ctx = ctx.drop(columns="sentence_number").sort_values("context_id").reset_index(drop=True)
    clog["contexts_kept"] = len(ctx)

    edges = (ctx.groupby(["citing_paper_id", "cited_paper_id"]).size()
             .rename("num_contexts").reset_index())
    edges["num_contexts"] = edges["num_contexts"].astype("int32")

    cit = edges.groupby("cited_paper_id").size()
    nctx = ctx.groupby("cited_paper_id").size()
    papers_df["citation_count"] = papers_df["paper_id"].map(cit).fillna(0).astype("int32")
    papers_df["num_contexts"] = papers_df["paper_id"].map(nctx).fillna(0).astype("int32")
    papers_df["year"] = papers_df["year"].astype("Int32")
    log.info("cleaning log: %s", clog)
    return papers_df, ctx, edges, clog
