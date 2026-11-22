# Results (data v3, middle grid, MiniLM)

Corpus: 8,843 cs.CL papers. Queries: 549 validation and 558 test, one relevant
paper each. All choices were made on validation MRR@10; test was scored once
with those choices frozen. Produced by `python scripts/run_all.py` (P2 middle
grid, MiniLM dense). P4's harness gives the official numbers and significance tests.

## Validation MRR@10, every system

Best setting per representation and model. P2 used window 0 and no
stemming. BM25 was tuned over 6 settings; other models use one setting.

| Representation | TF-IDF | BM25 (tuned) | Query likelihood | RM3 | BM25F | Dense MiniLM |
|---|---:|---:|---:|---:|---:|---:|
| Abstract only | 0.211 | 0.250 | 0.245 | 0.245 | | 0.218 |
| Citation contexts only | 0.054 | 0.263 | 0.286 | 0.259 | | 0.228 |
| Expanded (abstract + contexts) | 0.183 | **0.367** | 0.335 | 0.360 | | 0.287 |
| Fielded (title, abstract, contexts) | | | | | 0.343 | |
| **Hybrid RRF** (BM25 expanded + MiniLM expanded, k = 10) | | | | | | **0.387** |

BM25 settings selected: abstract k1 = 0.8, b = 0.75; contexts k1 = 0.8, b = 0.4;
expanded k1 = 1.2, b = 0.4. The two TF-IDF cells for contexts and expanded
come from the method-only index, which won on validation. Everywhere else,
all contexts beat method-only contexts.

### Method filter (all contexts vs method-only), validation MRR@10, best setting

| Representation | Model | All contexts | Method only |
|---|---|---:|---:|
| Contexts | BM25 | 0.263 | 0.183 |
| Contexts | Query likelihood | 0.286 | 0.216 |
| Expanded | BM25 | 0.367 | 0.331 |
| Expanded | RM3 | 0.360 | 0.320 |
| Fielded | BM25F | 0.343 | 0.299 |

### Hybrid RRF constant (validation)

| k | 10 | 20 | 40 | 60 | 100 |
|---|---:|---:|---:|---:|---:|
| MRR@10 | **0.387** | 0.383 | 0.380 | 0.377 | 0.376 |

k = 10 is the smallest value tried, so an even smaller k might do slightly better.

## Test, frozen systems

| System | MRR@10 | Recall@10 | Recall@100 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense MiniLM, expanded | 0.302 | 0.473 | 0.724 | 0.342 |
| BM25, expanded (k1 = 1.2, b = 0.4) | 0.391 | 0.525 | 0.722 | 0.424 |
| **Hybrid RRF, k = 10** | **0.399** | **0.599** | **0.780** | **0.447** |

## What these show (preliminary, before significance testing)

* **Research question 1 (document expansion):** adding citation contexts to the abstract helps every model. BM25 goes from 0.250 to 0.367 on validation, and MiniLM from 0.218 to 0.287.
* **Research question 2 (which approach):** tuned BM25 beats MiniLM. Hybrid fusion is best, with the largest gain in recall (Recall@10 0.599 vs 0.525 for BM25 on test).
* **Method filter:** keeping only method-labelled contexts hurts. The background contexts carry useful descriptions too.

## Not run

Stemming, context windows 1 and 2, BM25F field weights, and the full parameter
grids for query likelihood and RM3 were not run; `--p2 full` runs them. SPECTER
was not run either; add it with `--models ... sentence-transformers/allenai-specter`.
