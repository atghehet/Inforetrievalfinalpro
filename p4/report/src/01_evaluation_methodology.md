# 1. Evaluation Methodology

*P4 section. All numbers in this section come from the real project data (data label REAL); the tables are generated from the output files in `p4/outputs/real/` by `p4/experiments/make_report.py`.*

## 1.1 Task, data and splits

A query is one citing sentence from a cs.CL paper (citation markers removed); the single relevant document is the paper that sentence cites. The corpus
holds 8,843 cs.CL papers from the unarXive open subset (CC BY-SA 4.0), built by P1 (`data/processed`). There are 3,742 queries, split **by citing paper** into
2,635 train, 549 validation and 558 test queries (1,299 / 254 / 280 citing papers). Every query has exactly one relevant paper, so the qrels carry one
judgement per query. P4 verified that no citing paper appears in more than one split (0 shared of 1,833).

Only 24% of the corpus papers are cited by another corpus paper, so 76% can only be found through their abstract; a system that indexes citation contexts
alone cannot retrieve them. Every query's target is cited at least once in the full graph by construction.

## 1.2 Systems evaluated

Seventeen systems were available, all as TREC run files (`qid Q0 docid rank score tag`):

* **P2, lexical (13):** the representations *abstract*, *context* (citing sentences only) and *expanded* (abstract plus citing sentences) with TF-IDF, BM25,
  query likelihood (QL) and RM3, plus a fielded BM25F system. Every configuration was tuned on validation by P2; the frozen test run of each is evaluated.
* **P3, dense (3):** MiniLM sentence embeddings with exact FAISS search for the *abstract*, *contexts* (max over a paper's citing sentences) and *expanded* representations.
* **P3, hybrid (1):** reciprocal rank fusion of the dense system `minilm_expanded` and P2's BM25 run (`expanded-w0-plain-all-bm25-2`), with the RRF constant k = 10 chosen on validation.

SPECTER runs were not delivered, so no SPECTER result is reported.

## 1.3 Metrics

All metrics are computed per query and averaged over **all judged queries**; a query for which a run returns nothing scores 0 (no query was dropped; every run answers all 549 validation and 558 test queries).
With one relevant paper per query: **Recall@K** is the share of queries whose paper is in the top K (a hit rate); **MRR@10** is the mean reciprocal rank, 0 beyond rank 10; **nDCG@10** uses binary gain
(`1/log2(rank+1)`); **MAP** is average precision over the whole returned list, which with one relevant paper equals the reciprocal rank at the run's depth.
We report Recall@5, Recall@10, Recall@100, MRR@10, nDCG@10 and MAP.

*Run depth.* P2 runs return 100 papers per query, P3 runs 1,000. Metrics up to rank 100 are comparable across systems; **MAP is not exactly comparable** (a P2 relevant paper ranked below 100 scores 0, a P3 one still scores
`1/rank`), and P2 systems cannot be re-ranked beyond rank 100. We therefore use MRR@10 as the primary metric and treat MAP as secondary.

## 1.4 Protocol and discipline

1. **Validation for choices, test once.** P2 and P3 selected their configurations on validation MRR@10 and froze them. P4's own choice (the authority-prior weight, section 2) was tuned on validation only.
   The frozen test runs were evaluated by one command (`run_all`); the list of comparisons below was fixed from the research questions before any test number was inspected, and nothing was re-tuned afterwards.
2. **Input checks before any result** (`p4/outputs/real/stage1/`):
   * `verify_integration`: 0 failed, 0 warnings. Tables match the schema, IDs are consistent, qrels agree with the queries (549 / 558 / 2,635), every run answers every judged query and uses only known papers,
     splits are disjoint by citing paper, and P4's harness reproduces P2's own logged validation MRR@10 for all 47 tuning runs (largest difference 5.6e-17).
   * P1's leakage assertion (`assert_no_leakage`) passes on the real index (6,598 contexts, 3,742 queries). The P2/P3 input file was regenerated with P1's own export code
     (the teammates' indexed file was not shared with P4); its 6,598 contexts are all index contexts, with no query context and no query sentence.
   * Cross-check against the team's `scripts/check_run.py` on all 17 validation runs: MRR@10, Recall@10, Recall@100 and nDCG@10 agree to the four decimals `check_run.py` prints
     (largest difference 4.99e-5, i.e. rounding).
3. **Significance testing.** For a pair of systems we take the per-query score difference on the same 558 test queries and report the mean difference with a **95% bootstrap confidence interval**
   (10,000 resamples, seed 42) and a **paired Wilcoxon signed-rank test** (zero differences dropped; exact distribution for small samples). Because many pairs are compared, p-values are
   **Holm-corrected** within each metric over the 16 pre-specified comparisons; we call a difference significant only if the corrected p-value is below 0.05 and the sign of the rank evidence agrees with the sign of the mean difference.
   A difference whose confidence interval includes zero is not reported as an improvement, even if its mean is positive.
4. **Pre-specified comparisons (16 per metric).** RQ1 (does expansion help): expanded vs abstract and vs context for BM25, and for dense. RQ2 (lexical vs dense vs hybrid): dense vs BM25, hybrid vs the best lexical and vs dense.
   Within P2: expanded BM25 vs RM3, QL and BM25F. RQ3 (authority prior): each of four systems vs itself with the prior.

## 1.5 Leakage

P1 removes every query sentence, its exact copies and near-duplicates from the indexed contexts. Two residual issues remain and are handled as follows (details: `p4/report/proposals/leakage_safe_graph_protocol.md`):

* **Citation graph (leak A).** `edges.parquet` and `papers.citation_count` include the citation that produced each held-out query: every held-out query's own edge is in the full graph (100%). Any authority score computed from
  the full graph would therefore know that a query's target is cited. P4 never uses it for evaluation: it builds an *evaluation graph* without the citing papers of validation and test queries (section 2).
* **Same-paper contexts in the index (leak B).** For 55.0% of validation and 54.1% of test queries (302 of 549 and 302 of 558), the index still contains *other* sentences written by the same citing paper.
  P1 does not remove them, and they were indexed by P2 and P3. This is a property of the delivered runs and **could not be corrected by P4**; it may favour context-based systems (contexts, expanded, hybrid) and is
  reported as a limitation. A strict-index rerun (`index_contexts_strict.parquet`) was not possible because it requires rerunning P2 and P3.

## 1.6 Efficiency

Only P2 delivered timing and index-size logs (median and mean milliseconds per query, index build seconds and bytes). No timing or index-size log exists for the P3 dense and hybrid systems, so none is reported or estimated.

## 1.7 Reproduction

The exact commands are in `p4/report/final/REPRODUCE.md`.
