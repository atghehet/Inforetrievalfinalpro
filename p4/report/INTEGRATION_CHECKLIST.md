# Integration checklist (P4)

**Workflow agreed with the team lead (2026-11-22):** P4 prepares code, tests, protocol and report structure now. **No real experiment or real finding
is produced until P1, P2 and P3 have synced their final files and the integrated project is verified.** The cs.CL baseline build
(`IRdata\builds\cscl_baseline\`) stays as baseline dataset statistics only; it is not a retrieval or graph-fusion result.
Synthetic and demo results are allowed during preparation and are always labelled SYNTHETIC / DEMO.

## 1. Files P4 expects

### From P1 (`data/processed/`, built on the agreed dataset)
| File | Columns (names are fixed in `citeds/schema.py`) |
|---|---|
| `papers.parquet` | `paper_id`, `title`, `abstract`, `year`, `categories`, `citation_count`, `num_contexts` |
| `contexts.parquet`, `index_contexts.parquet` | `context_id`, `sentence_id`, `citing_paper_id`, `cited_paper_id`, `text`, `prev_text`, `next_text`, `section`, `num_cited_in_sentence` |
| `edges.parquet` | `citing_paper_id`, `cited_paper_id`, `num_contexts` (edge points citing -> cited) |
| `queries.parquet` | `query_id`, `split`, `context_id`, `sentence_id`, `citing_paper_id`, `cited_paper_id`, `text`, `cited_citation_count` |
| `qrels/qrels.{train,val,test}.txt` | `qid 0 docid 1` (docid = `paper_id`) |
| `context_functions.parquet` | `context_id`, `function` (background / method / result), `confidence`; must cover every context |
| `build_info.json` | counts and cleaning log (provenance) |

P1 code changes proposed (not yet approved): raw-text reference fallback (`proposals/p1_raw_text_reference_fallback.md`) and `encoding="utf-8"` on file writes
(`splits.py` 86 and 89, `build.py` 64 and 71, `citation_function.py` 141 and 143). **P4 needs to know which build (with or without them) produced the data**, because
counts differ (baseline without the fallback: 2,720 edges, 5,014 contexts, 1,918 queries).

### From P2
| File | Notes |
|---|---|
| `test/*.run`, `validation/*.run` | six-column TREC runs; docid = `paper_id`; qid = P1's `query_id`; the sixth column holds the winning configuration, so P4 uses `--name-from-tag` |
| `best.json`, `tuning.csv`, `efficiency.csv` | for the efficiency tables and the reproduction check (`p2_check` in the verifier) |
| `fusion/bm25.validation.run`, `fusion/bm25.test.run` | the BM25 runs P3 fuses |
| the JSONL files indexed (`papers.jsonl`, `contexts.jsonl`, `validation.jsonl`, `test.jsonl`) | **gate:** every `context_id` must be in `index_contexts.parquet`; none may be a query context or query sentence |
| method-filter outputs (`apply_function_labels.py`, commit `4ffbafd`) | needs P1's `context_functions.parquet`; validation runs exist for every configuration, test runs only for the winner of each representation/model |

### From P3
| File | Notes |
|---|---|
| `dense_{val,test}.run`, `hybrid_{validation,test}.run` (and any additional systems) | same TREC format; **portable scripts** (the current ones hard-code `~/Desktop/CitationSearchProject`) |
| timing and index-size logs for dense and hybrid | for the efficiency table; none exist yet |
| embeddings (for the content-based recommender) | optional, P4 recommender add-on |

## 2. Known compatibility issues (to settle at integration)

| # | Issue | Owner | Status |
|---|---|---|---|
| 1 | P1's build crashes on Windows without UTF-8 mode (non-ASCII query text) | P1 | proposal written |
| 2 | Only 9.6% of references resolve (graph and contexts about half of what the fallback gives) | P1 | proposal written |
| 3 | `edges.parquet` and `papers.citation_count` include the query's own citation (leak A) | P4 + team | protocol proposed, needs approval |
| 4 | P1's index keeps same-citing-paper contexts for 65% of test queries (leak B, baseline numbers) | team | decision needed: primary index setting |
| 5 | P2's frozen test-run file names hide the winning configuration; only the best per representation/model is exported on test | P2 / P4 | `--name-from-tag` added; a test-set "with vs without method filter" comparison needs P2 to export both |
| 6 | P3 converter once wrote section names into `label`; P2's `apply_function_labels.py` replaces them with predicted functions | P2, P3 | resolved in code, verify on delivery |
| 7 | P3's converter passes one neighbour sentence per side; P2's window experiment expects up to two | P3 | open |
| 8 | Splits are `val` in P1 and `validation` in P2's folders | P2, P3 | handled by P3's converter; verify the qrels content equals P1's |
| 9 | Run depths differ (P2 up to 100, P3 dense/hybrid every paper), so MAP and deep ranks are not comparable across systems | P4 | documented in reports |
| 10 | The synthetic `p3/outputs` runs are the only runs available now; the compatibility tests depend on that folder | P4 | tests skip if it is absent; re-point at the final layout |
| 11 | Python packages: P3 needs `sentence-transformers` and `faiss-cpu`; P4 needs numpy, pandas, scipy, pyarrow, pytest (plus matplotlib for figures, not yet installed) | team | install approval needed |

## 3. Order of work when the integrated project arrives

1. **Freeze:** record the commit of each teammate's code and the exact data build (`RECORD.md`). Do not mix data builds.
2. **Verify the integration (read-only):**
   `python -m p4.experiments.verify_integration --processed <P1 processed> --config runs.json --out verify.json`
   (schemas, ID consistency, qrels versus queries, run files, leakage gate, reproduction of P2's logged validation MRR@10). **Every failure must be
   fixed by the owning teammate before anything else.**
3. **Run P4's own tests** (section 4) on the integrated tree.
4. **Team decisions recorded in `RECORD.md`:** leakage protocol approved (graph rule, index setting), dataset build version.
5. **Run the experiments once, in one command:**
   `python -m p4.experiments.run_all --data-label real --processed <P1 processed> --config runs.json --out <NEW folder> --confirm-integrated-project`
   (leakage gate, evaluation graph, test-split evaluation with paired tests and efficiency, fusion tuned on validation and applied to test,
   popularity / cold-start / year / errors analyses with leakage-safe counts; the function analysis only if `context_functions.parquet` exists).
6. **Sensitivity:** repeat the expansion comparison with the strict index (needs P2/P3 to rebuild from `index_contexts_strict.parquet`).
7. **Write findings** from `EXPERIMENT_MANIFEST.json` and the outputs, labelled REAL, into `RECORD.md`, then the report.

## 4. Tests that must pass after integration

| Suite | Command (from the repository root) |
|---|---|
| P4 | `python -m pytest p4 -q` (all tests; the compatibility tests re-run P4's harness on the committed P2/P3 outputs) |
| P1 | `python -m pytest tests -q` |
| P2 | `cd lexical; python -m unittest discover` |
| Integration | `python -m p4.experiments.verify_integration ...` reports 0 failed |
| Leakage | `python -m p4.graph.leakage_audit --processed <P1 processed> --out audit.json --context-file <each file P2/P3 index>` (the orchestrator does this itself and stops on failure) |

## 5. Outstanding experiments and results (nothing below exists as a real result yet)

| Research question | Experiment | Needs |
|---|---|---|
| RQ1 expansion vs abstract-only | abstract / context / expanded / fielded on the test split, paired tests; strict-index sensitivity | P2 runs, protocol decision |
| RQ2 lexical vs dense vs hybrid | all systems on the test split, paired tests, efficiency | P2 and P3 runs |
| RQ3 PageRank prior | tune prior and lambda on validation, apply to test; damping 0.5 / 0.85 / 0.95; popularity analysis with and without the prior | leakage-safe graph approved; runs |
| RQ4 cold start and citation function | cold-start and year analyses (ready); function analysis | P1's labels |
| Recommender add-on | co-citation and bibliographic coupling, hide-references Recall@k (**not yet implemented**) | evaluation protocol to specify |
| Report figures and final tables | **not yet implemented** (matplotlib) | install approval |
| Error analysis by sentence length and query field; 10-15 curated failure cases | **not yet implemented** | next P4 batch |
