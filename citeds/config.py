"""Shared configuration for the data pipeline (Member 1).

Every path and constant other members need lives here, so nobody hard-codes
file names. Import as:  from citeds import config
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("CITEDS_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("CITEDS_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"              # unarXive JSONL dumps go here (not committed)
PROCESSED_DIR = DATA_DIR / "processed"  # outputs every other member reads
EXTERNAL_DIR = DATA_DIR / "external"    # SciCite / ACL-ARC, crawler cache

PAPERS_PATH = PROCESSED_DIR / "papers.parquet"
CONTEXTS_PATH = PROCESSED_DIR / "contexts.parquet"          # ALL cleaned contexts
INDEX_CONTEXTS_PATH = PROCESSED_DIR / "index_contexts.parquet"  # leakage-controlled
EDGES_PATH = PROCESSED_DIR / "edges.parquet"
QUERIES_PATH = PROCESSED_DIR / "queries.parquet"
QRELS_DIR = PROCESSED_DIR / "qrels"                         # qrels.<split>.txt (TREC)
TOPICS_DIR = PROCESSED_DIR / "topics"                       # topics.<split>.tsv
CONTEXT_FUNCTIONS_PATH = PROCESSED_DIR / "context_functions.parquet"
FRESH_PAPERS_PATH = PROCESSED_DIR / "fresh_papers.parquet"  # optional crawler output
STATS_PATH = ROOT / "docs" / "dataset_stats.md"

# ---------------------------------------------------------------------------
# Corpus selection
# ---------------------------------------------------------------------------
FIELD = "cs.CL"            # arXiv category the corpus is restricted to
MAX_PAPERS = 200_000       # hard cap on corpus size

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
MIN_SENTENCE_TOKENS = 6    # contexts shorter than this (after cleaning) are dropped
MAX_SENTENCE_TOKENS = 120  # very long "sentences" are usually parsing errors
MIN_ABSTRACT_TOKENS = 20

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
SEED = 42
SPLIT_SIZES = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_QUERIES = 20_000                # total queries sampled across all splits
MAX_QUERIES_PER_CITED_PAPER = 5     # stops very popular papers dominating the query set
MIN_REMAINING_CONTEXTS = 1          # cited paper must keep >= this many contexts in the index

# ---------------------------------------------------------------------------
# Leakage control
# ---------------------------------------------------------------------------
NEAR_DUP_JACCARD = 0.5     # word-3-gram Jaccard >= this (vs a query citing the same paper) = near duplicate.
                           # 0.5 catches one- or two-word edits of a ~15-word sentence; lower = stricter.
SHINGLE_SIZE = 3

# ---------------------------------------------------------------------------
# Crawler (optional)
# ---------------------------------------------------------------------------
# !!! PLACEHOLDER — REPLACE BEFORE RUNNING THE CRAWLER !!!
# OpenAlex and Semantic Scholar ask automated clients for a contact email.
# Set the CITEDS_CONTACT_EMAIL environment variable (preferred, keeps it out
# of Git) or edit this line. The crawler refuses to run while it is the placeholder.
CONTACT_EMAIL = os.environ.get("CITEDS_CONTACT_EMAIL", "YOUR_EMAIL@example.com")
CRAWL_CACHE_DIR = EXTERNAL_DIR / "crawl_cache"
CRAWL_MIN_INTERVAL_S = 0.2  # <= 5 requests/second; OpenAlex polite-pool limit is 10/s


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, QRELS_DIR, TOPICS_DIR, CRAWL_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
