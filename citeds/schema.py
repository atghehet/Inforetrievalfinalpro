"""Shared table schemas. THIS IS THE CONTRACT BETWEEN TEAM MEMBERS.

All tables are Parquet files in data/processed/ (paths in citeds.config).
Column names below are fixed: if you need a new column, add it here in a PR.

ID scheme
---------
paper_id    Normalised arXiv ID without version, e.g. "2106.09685" or
            "cs/0112017" (old style). This is the `docid` in every run file.
sentence_id "<citing_paper_id>#<sentence_number>", e.g. "2106.09685#s0042".
            One sentence can cite several papers, so it is NOT unique per row.
context_id  "<sentence_id>><cited_paper_id>", e.g. "2106.09685#s0042>1902.00751".
            Unique per (citing sentence, cited paper) pair.
query_id    "q" + 6-digit number, e.g. "q000123". This is the `qid` in run files.
"""
from __future__ import annotations

import pyarrow as pa

PAPERS = pa.schema([
    ("paper_id", pa.string()),         # normalised arXiv ID
    ("title", pa.string()),
    ("abstract", pa.string()),         # cleaned abstract text
    ("year", pa.int32()),              # first-version year (may be null)
    ("categories", pa.string()),       # space-separated arXiv categories
    ("citation_count", pa.int32()),    # in-corpus citations (number of citing papers)
    ("num_contexts", pa.int32()),      # number of cleaned contexts that cite it
])

CONTEXTS = pa.schema([
    ("context_id", pa.string()),
    ("sentence_id", pa.string()),
    ("citing_paper_id", pa.string()),
    ("cited_paper_id", pa.string()),
    ("text", pa.string()),             # cleaned citing sentence (markers removed)
    ("prev_text", pa.string()),        # cleaned previous sentence ("" if none)
    ("next_text", pa.string()),        # cleaned next sentence ("" if none)
    ("section", pa.string()),          # section title as in the source, lower-cased
    ("num_cited_in_sentence", pa.int32()),  # how many corpus papers this sentence cites
])

EDGES = pa.schema([
    ("citing_paper_id", pa.string()),
    ("cited_paper_id", pa.string()),
    ("num_contexts", pa.int32()),      # how many sentences carry this citation
])

QUERIES = pa.schema([
    ("query_id", pa.string()),
    ("split", pa.string()),            # "train" | "val" | "test"
    ("context_id", pa.string()),       # the context the query was taken from
    ("sentence_id", pa.string()),
    ("citing_paper_id", pa.string()),
    ("cited_paper_id", pa.string()),   # the relevant document (one per query)
    ("text", pa.string()),             # query text = cleaned citing sentence
    ("cited_citation_count", pa.int32()),  # for cold-start / popularity bucketing
])

CONTEXT_FUNCTIONS = pa.schema([
    ("context_id", pa.string()),
    ("function", pa.string()),         # "background" | "method" | "result"
    ("confidence", pa.float32()),      # probability of the predicted class
])

CITATION_FUNCTIONS = ("background", "method", "result")

# Optional crawler output: PAPERS columns + OpenAlex metadata.
FRESH_PAPERS = pa.schema(list(PAPERS) + [
    ("openalex_id", pa.string()),
    ("openalex_cited_by_count", pa.int32()),  # global citations per OpenAlex
    ("referenced_works", pa.string()),         # "|"-joined OpenAlex work IDs
    ("publication_date", pa.string()),         # YYYY-MM-DD
])

ALL = {
    "papers": PAPERS,
    "contexts": CONTEXTS,
    "index_contexts": CONTEXTS,
    "edges": EDGES,
    "queries": QUERIES,
    "context_functions": CONTEXT_FUNCTIONS,
}


def conform(df, schema: pa.Schema):
    """Return a pyarrow Table with exactly `schema`'s columns, in order and type."""
    missing = [n for n in schema.names if n not in df.columns]
    if missing:
        raise ValueError(f"missing columns for schema: {missing}")
    return pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
