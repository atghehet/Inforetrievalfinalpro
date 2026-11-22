# Shared data format (team contract)

Owner: Member 1. Source of truth in code: `citeds/schema.py`. If you need a
new column, change `schema.py` and this file in the same pull request.

All tables are **Parquet** files in `data/processed/`. Read them with
`citeds.io` (recommended) or `pandas.read_parquet`.

## IDs

| ID | Format | Example | Notes |
|---|---|---|---|
| `paper_id` | arXiv ID, no version | `2106.09685`, `cs/0112017` | Use this as `docid` in run files |
| `sentence_id` | `<citing paper_id>#s<4-digit sentence number>` | `2106.09685#s0042` | Not unique per row: one sentence can cite several papers |
| `context_id` | `<sentence_id>><cited paper_id>` | `2106.09685#s0042>1902.00751` | Unique per row |
| `query_id` | `q` + 6 digits | `q000123` | Use this as `qid` in run files |

## Tables

### `papers.parquet` — the documents

| Column | Type | Meaning |
|---|---|---|
| `paper_id` | string | arXiv ID |
| `title` | string | cleaned title |
| `abstract` | string | cleaned abstract |
| `year` | int32 (nullable) | year of first arXiv version |
| `categories` | string | space-separated arXiv categories |
| `citation_count` | int32 | number of **in-corpus** papers citing it |
| `num_contexts` | int32 | number of cleaned citing sentences (before leakage control) |

### `index_contexts.parquet` — citation contexts SAFE to index

Same columns as `contexts.parquet`, with every query sentence (all splits),
its exact copies, and near-duplicates removed. **Use only this file for
document expansion.**

### `contexts.parquet` — all cleaned citation contexts (NOT safe to index)

| Column | Type | Meaning |
|---|---|---|
| `context_id` | string | see IDs |
| `sentence_id` | string | see IDs |
| `citing_paper_id` | string | paper containing the sentence |
| `cited_paper_id` | string | paper being cited (always in `papers`) |
| `text` | string | cleaned citing sentence, citation markers removed |
| `prev_text` | string | previous sentence in the paragraph (`""` if none) |
| `next_text` | string | next sentence in the paragraph (`""` if none) |
| `section` | string | section heading, lower-cased |
| `num_cited_in_sentence` | int32 | how many corpus papers this sentence cites |

### `edges.parquet` — citation graph (Member 4: PageRank)

| Column | Type | Meaning |
|---|---|---|
| `citing_paper_id` | string | source node |
| `cited_paper_id` | string | target node |
| `num_contexts` | int32 | sentences carrying this citation (edge weight, optional) |

Built from cleaned contexts, so both endpoints are in `papers` and there are
no self-loops.

### `queries.parquet` — evaluation queries

| Column | Type | Meaning |
|---|---|---|
| `query_id` | string | see IDs |
| `split` | string | `train`, `val` or `test` |
| `context_id`, `sentence_id`, `citing_paper_id` | string | where the query came from |
| `cited_paper_id` | string | the single relevant document |
| `text` | string | query text = cleaned citing sentence |
| `cited_citation_count` | int32 | relevant paper's `citation_count` (for cold-start buckets) |

Also written as TREC files:

* `qrels/qrels.<split>.txt` — `qid 0 docid 1`
* `topics/topics.<split>.tsv` — `qid<TAB>query text`

### `context_functions.parquet` — citation-function labels (Member 2, Member 4)

| Column | Type | Meaning |
|---|---|---|
| `context_id` | string | joins to `contexts` / `index_contexts` / `queries` |
| `function` | string | `background`, `method` or `result` |
| `confidence` | float32 | predicted-class probability |

Covers every row of `contexts.parquet`, including query sentences.

### `fresh_papers.parquet` (optional crawler output)

`papers` columns plus `openalex_id`, `openalex_cited_by_count`,
`referenced_works` (`|`-joined OpenAlex IDs) and `publication_date`.

## Run files (all members)

TREC format, one line per retrieved document, space separated:

```
qid Q0 docid rank score run_name
q000123 Q0 1902.00751 1 17.43 bm25_abstract
```

## Which split for what

* `train`: anything learned (fusion weights, learning-to-rank, citation-function weighting).
* `val`: hyper-parameter tuning (BM25 k1/b, RM3, fusion alpha, PageRank weight).
* `test`: final numbers only, once per system.
