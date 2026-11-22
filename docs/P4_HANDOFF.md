# P4 handoff: Graph, Evaluation and Analysis

Everything P4 needs from P1, P2 and P3, where it is, and the conventions it
follows. The inputs are produced by `python scripts/run_all.py` (see the root
README) and shared on the team drive as `data/processed` and `runs/`.

## Inputs

| What | Where | Notes |
|---|---|---|
| Qrels (ground truth) | `data/processed/qrels/qrels.{train,val,test}.txt` | `qid 0 docid 1`, exactly one relevant paper per query |
| Queries | `data/processed/queries.parquet` | `query_id`, `split`, `text`, `cited_paper_id`, `cited_citation_count`, `context_id` |
| Papers | `data/processed/papers.parquet` | `paper_id`, `title`, `abstract`, `year`, `citation_count`, `num_contexts` |
| Citation graph | `data/processed/edges.parquet` | `citing_paper_id → cited_paper_id`, `num_contexts` (edge weight). Both ends are in `papers`; no self-loops |
| Citation-function labels | `data/processed/context_functions.parquet` | Join `queries.context_id = context_functions.context_id` for per-function analysis |
| Index of all run files | `runs/MANIFEST.csv` | `member, system, split, path, queries, lines, six_columns` |
| P2 lexical runs | `runs/p2/validation/*.run` (tuning grid), `runs/p2/test/*.run` (frozen, one per representation and model), `runs/p2/tuning.csv`, `best.json`, `efficiency.csv` | P2 selected on validation MRR@10 |
| P3 dense runs | `runs/p3/dense/minilm_<variant>.{val,test}.run` | MiniLM only; variants `abstract`, `contexts`, `expanded` |
| P3 hybrid runs | `runs/p3/hybrid/hybrid_rrf.{val,test}.run`, `selection.json` | Dense system and RRF k frozen on validation |

Read the data with `citeds.io` (for example `io.load_edges()`, `io.load_queries("test")`). Column definitions are in `docs/SCHEMA.md`.

## What was run (scope)

* **P2 lexical: middle grid** (`--p2 middle`). It covers every model (TF-IDF, BM25, query likelihood, RM3, BM25F) on every representation (abstract, context, expanded, fielded), window 0, no stemming, method filter on and off. BM25 is tuned over k1 ∈ {0.8, 1.2, 1.6} × b ∈ {0.4, 0.75} on abstract, context and expanded. Other models use one setting. Stemming, context-window (1, 2) and BM25F-weight comparisons were **not run**; `python scripts/run_all.py --p2 full` reproduces them (hours). P2 runs keep the top 100 papers per query.
* **P3 dense: MiniLM only** (`all-MiniLM-L6-v2`), abstract, contexts and expanded. SPECTER is future work (`--models ... sentence-transformers/allenai-specter`).
* **P3 hybrid:** RRF of the best MiniLM run and P2's tuned BM25 fusion run, with k chosen on validation (`runs/p3/hybrid/selection.json`).

## Conventions

* Run format: `qid Q0 docid rank score tag`, ranks from 1. P2 runs hold up to 100 papers per query and P3 runs up to 1,000, so compare Recall@k only for k ≤ 100.
* IDs: `query_id` such as `q002635`, and `paper_id` = arXiv ID without version, such as `2106.09685`.
* Tuning is on `val` only. `test` runs are frozen and meant to be scored once per system.
* A query missing from a run counts as a miss: score it as 0, don't drop it.
* `scripts/check_run.py <run> --split val|test` validates the format and prints MRR@10, Recall@10/100 and nDCG@10. P4's harness should reproduce these numbers on the same file.

## Before reporting any result

```python
from citeds import io
from citeds.leakage import assert_no_leakage
assert_no_leakage(io.load_index_contexts(), io.load_queries())
```

## P4 tasks and the inputs each one uses

| Task | Inputs |
|---|---|
| Evaluation harness (MRR, Recall@k, nDCG, MAP) | Qrels + any run in `MANIFEST.csv` |
| Significance tests (for example a paired t-test or randomisation test on per-query scores) | Two runs on the same split, per-query metrics from the harness |
| PageRank | `edges.parquet` (directed citing → cited; `num_contexts` as an optional weight) |
| PageRank prior experiment | Rerank a frozen run with `score' = score + w * log(PageRank)` or a rank fusion. Tune `w` on val and apply it once on test |
| Popularity bias and cold start | `queries.cited_citation_count` buckets 1–2, 3–10, 11–50, >50 (same as `docs/dataset_stats.md`) |
| Per-citation-function error analysis | `context_functions.function` for the query's `context_id` |
| Co-citation recommender | `edges.parquet` (papers cited together by the same citing paper) and P3 embeddings in `cache/p3/<model>/papers.npy` (row order = `papers.parquet`) |

## Dataset facts to state in the report

The corpus is 8,843 cs.CL papers from the unarXive 2022 open subset (CC BY-SA
4.0). There are 3,742 queries: 2,635 train, 549 val and 558 test. 76% of papers
are never cited inside the corpus, so they can only be found through their
abstract, and `contexts`-only systems cannot retrieve them. Every query's
target paper is cited at least once by construction. Full numbers are in
`docs/dataset_stats.md`.
