# 2. Graph-based Ranking

*P4 section. REAL data. Tables are generated from `p4/outputs/real/`.*

## 2.1 The citation graph and its direction

P1's `edges.parquet` holds one edge per (citing paper, cited paper) pair, **pointing from the citing paper to the cited paper**, with the number of citing sentences as an optional weight; both ends are corpus papers and there are no self-loops.
We checked the direction against publication years: in 99.86% of the 5,869 edges the citing paper is not older than the cited paper, so the orientation is the intended one. PageRank therefore
lets a paper pass authority to the papers it cites, and a paper's score reflects being cited by papers that are themselves cited. We use the **unweighted** graph (one edge per pair).

## 2.2 Leakage risk and the evaluation graph

The full graph contains the citation behind every held-out query: for 100% of validation and test queries the query's own edge is present, so `papers.citation_count` and a PageRank computed from the full graph both "know" that the target is cited.
For example, no test target has zero citations in the full graph, while 25.6% of test targets have none once the held-out edges are removed.

**Protocol (leakage-safe-v1).** Remove every edge whose *citing* paper has a validation or test query (534 held-out citing papers; 1,198 of 5,869 edges, 20.4%). P1 assigns splits per citing paper, so this removes every edge that could have produced a held-out
query, together with all other citations those papers make. From the remaining 4,671 edges we compute, for **all** 8,843 papers, (a) `citation_count_eval`, the number of remaining citing papers, and (b) PageRank with damping 0.5, 0.85 and 0.95.
The fusion tool refuses a prior whose manifest does not show that the evaluated split was held out. Because validation and test are both held out, the same graph is used for tuning on validation and evaluation on test, and the test queries' citations never enter any prior.

| quantity | value |
|---|---|
| papers (nodes) | 8,843 |
| citation edges, full graph | 5,869 |
| edges removed (citing papers of val/test queries held out) | 1,198 (20.4%) |
| edges kept in the evaluation graph | 4,671 |
| held-out citing papers | 534 |
| edges whose citing paper is not older than the cited paper | 99.86% |
| papers never cited, full graph | 76.0% |
| papers never cited, evaluation graph | 78.6% |
| test targets with 0 citations, full graph / evaluation graph | 0.0% / 25.6% |
| mean citations of a test target, full / evaluation graph | 5.47 / 3.60 |
| Spearman: PageRank (d=0.85) vs evaluation citation count | 0.994 |
| PageRank damping 0.50: iterations / converged / final change | 25 / True / 5.7e-11 |
| PageRank damping 0.85: iterations / converged / final change | 100 / False / 1.5e-10 |
| PageRank damping 0.95: iterations / converged / final change | 100 / False / 1.0e-05 |


PageRank with damping 0.5 converged; with 0.85 it reached a final change of 1.5e-10 and with 0.95 a final change of 1.0e-5 after the 100-iteration cap, so the 0.95 scores are slightly unconverged (this did not affect the selection, see below).
PageRank is almost a monotone function of the evaluation citation count here (Spearman 0.994), because the graph is shallow: 78.6% of papers have no citations at all. The highest-ranked papers are well-known (for example XLNet, Sentence-BERT, byte-pair-encoding NMT), a basic sanity check.

## 2.3 Authority-prior fusion

For a system's ranked list (up to 1,000 papers for P3, 100 for P2) we re-rank with the formula given in the project handoff:

`score'(d) = score(d) + lambda * log(prior(d))`

with the system's own scores (no normalisation) and the prior taken from {evaluation citation count, PageRank 0.5 / 0.85 / 0.95}. A prior value of 0 is replaced by 1e-12 so the logarithm is finite (a paper that is never cited receives the strongest penalty).
The prior and **lambda are tuned on the validation queries only** (grid 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3; ties go to the smaller lambda, so "no prior" wins unless the prior strictly helps on validation), then frozen and applied once to the test run.
Fusion was applied to four systems: abstract-only BM25 (the lexical baseline for the abstract-only question), expanded BM25 (the best lexical system on validation), dense `minilm_expanded`, and the hybrid.
A prior can only reorder papers a system already returned; P2 lists have 100 papers, so P2 Recall@100 cannot change.

**Validation (tuning).** Best validation MRR@10 reached by each prior (over lambda):

| system | prior | val MRR@10 (lambda 0) | best val MRR@10 | at lambda |
|---|---|---:|---:|---:|
| p2_abstract_bm25 | citation_count_eval | 0.2502 | 0.2835 | 0.3000 |
| p2_abstract_bm25 | pagerank_d050 | 0.2502 | 0.2694 | 3.0000 |
| p2_abstract_bm25 | pagerank_d085 | 0.2502 | 0.2724 | 3.0000 |
| p2_abstract_bm25 | pagerank_d095 | 0.2502 | 0.2729 | 3.0000 |
| p2_expanded_bm25 | citation_count_eval | 0.3669 | 0.3844 | 0.1000 |
| p2_expanded_bm25 | pagerank_d050 | 0.3669 | 0.3692 | 1.0000 |
| p2_expanded_bm25 | pagerank_d085 | 0.3669 | 0.3708 | 0.3000 |
| p2_expanded_bm25 | pagerank_d095 | 0.3669 | 0.3706 | 0.3000 |
| p3_minilm_expanded | citation_count_eval | 0.2866 | 0.2932 | 0.0010 |
| p3_minilm_expanded | pagerank_d050 | 0.2866 | 0.2866 | 0.0000 |
| p3_minilm_expanded | pagerank_d085 | 0.2866 | 0.2866 | 0.0000 |
| p3_minilm_expanded | pagerank_d095 | 0.2866 | 0.2866 | 0.0000 |
| p3_hybrid_rrf | citation_count_eval | 0.3865 | 0.3911 | 0.0010 |
| p3_hybrid_rrf | pagerank_d050 | 0.3865 | 0.3865 | 0.0000 |
| p3_hybrid_rrf | pagerank_d085 | 0.3865 | 0.3865 | 0.0000 |
| p3_hybrid_rrf | pagerank_d095 | 0.3865 | 0.3865 | 0.0000 |


On validation the leakage-safe citation count beat every PageRank variant for every system, and it was the prior frozen in all four cases. For the P2 abstract BM25 system the PageRank variants peaked at the largest weight of the grid (lambda 3), which raised
the question whether the grid was too narrow for them. A **validation-only** check with a wider grid (lambda up to 100; no test file is read; `p4/outputs/real/sensitivity_extended_lambda_val/`) answers it: PageRank's best value stays at lambda 3 and then falls sharply, so the
frozen choice was not an artefact of the grid, and the citation count remains the best prior.

| system | prior | val MRR@10 (lambda 0) | best val MRR@10, lambda 0-100 | at lambda | val MRR@10 at lambda 100 |
|---|---|---:|---:|---:|---:|
| p2_abstract_bm25 | citation_count_eval | 0.2502 | 0.2835 | 0.3000 | 0.0970 |
| p2_abstract_bm25 | pagerank_d050 | 0.2502 | 0.2694 | 3.0000 | 0.1073 |
| p2_abstract_bm25 | pagerank_d085 | 0.2502 | 0.2724 | 3.0000 | 0.0997 |
| p2_abstract_bm25 | pagerank_d095 | 0.2502 | 0.2729 | 3.0000 | 0.0983 |
| p2_expanded_bm25 | citation_count_eval | 0.3669 | 0.3844 | 0.1000 | 0.0321 |
| p2_expanded_bm25 | pagerank_d050 | 0.3669 | 0.3692 | 1.0000 | 0.0444 |
| p2_expanded_bm25 | pagerank_d085 | 0.3669 | 0.3708 | 0.3000 | 0.0420 |
| p2_expanded_bm25 | pagerank_d095 | 0.3669 | 0.3706 | 0.3000 | 0.0448 |


The frozen configuration and the test evaluation were made before this check and were not changed by it.

**Test (frozen configuration, evaluated once).**

| system | prior chosen on val | lambda | val MRR@10 base | val MRR@10 fused | test MRR@10 base | test MRR@10 fused | test nDCG@10 base | test nDCG@10 fused | test Recall@10 base | test Recall@10 fused |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p2_abstract_bm25 | citation_count_eval | 0.3000 | 0.2502 | 0.2835 | 0.2695 | 0.3127 | 0.3028 | 0.3453 | 0.4086 | 0.4480 |
| p2_expanded_bm25 | citation_count_eval | 0.1000 | 0.3669 | 0.3844 | 0.3912 | 0.3991 | 0.4237 | 0.4343 | 0.5251 | 0.5466 |
| p3_minilm_expanded | citation_count_eval | 0.0010 | 0.2866 | 0.2932 | 0.3015 | 0.2994 | 0.3421 | 0.3414 | 0.4731 | 0.4767 |
| p3_hybrid_rrf | citation_count_eval | 0.0010 | 0.3865 | 0.3911 | 0.3990 | 0.3944 | 0.4468 | 0.4425 | 0.5986 | 0.5950 |


What the test results support (paired tests, Holm-corrected, see section 3.2):

* **Abstract-only BM25:** the prior raises MRR@10 from 0.2695 to 0.3127 (+0.0432, 95% CI [+0.0250, +0.0616]) and nDCG@10 by +0.0426; both survive the Holm correction. Recall@10 rises by +0.0394 (CI [+0.0125, +0.0681]) but the corrected p-value (0.068) does not reach 0.05, so we do not claim a Recall@10 improvement.
* **Expanded BM25:** +0.0079 MRR@10 (CI [-0.0008, +0.0166]; corrected p = 0.27), +0.0215 Recall@10 (corrected p = 0.15): **no statistically supported gain**, although the point estimates are positive.
* **Dense `minilm_expanded` and hybrid:** MRR@10 changes of -0.0020 and -0.0047 with intervals that include zero: **no effect**. On validation their best weight was the smallest non-zero weight (0.001), i.e. the prior was nearly switched off.

So the leakage-safe authority prior helps only the system that has no citation information of its own (abstract-only retrieval). Once a representation already contains the citing sentences, or dense/hybrid evidence, it adds nothing measurable.
A plausible reading, which we did not test directly, is that context expansion already favours well-cited papers (they have more citing sentences to match), so the prior is largely redundant there; the popularity analysis (section 3.4) looks at this from the retrieval side.
Using the prior alone is not a retrieval system: ranking all papers by evaluation citation count, ignoring the query, gives MRR 0.010 on the test queries.

## 2.4 Limitations of the graph experiment

* The graph is small and shallow (5,869 edges; 76% of papers uncited), which limits what PageRank can add beyond a citation count.
* The benchmark samples queries from citing sentences, so its targets are papers that *are* cited, and popular papers are cited more often. A citation-based prior is therefore informative about the benchmark's targets by design; this is part of the task
  (people cite well-known papers) but means the gain should not be assumed to transfer to a setting where queries target uncited or new papers. 25.6% of test targets have no citations in the evaluation graph and are penalised by the prior.
* Held-out papers' own citations are excluded, but papers citing a test target from the *training* split remain in the graph; this is legitimate prior knowledge (it does not come from a test query) and is the intended design.
* The protocol is P4's proposal and was not separately signed off by the other team members.
* Only the unweighted graph and the additive-log fusion were evaluated; weighted PageRank, HITS and rank-based fusion were not run.
