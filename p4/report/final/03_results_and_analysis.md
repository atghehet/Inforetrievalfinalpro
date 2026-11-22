# 3. Results and Analysis

*P4 section. **All results are REAL**: 17 systems from P2 and P3 on the real cs.CL corpus, evaluated with P4's harness. Tables are generated from the files in `p4/outputs/real/` by `p4/experiments/make_report.py`. Test numbers come from the frozen
test runs (558 queries), evaluated once; validation numbers (549 queries) were used for analysis and tuning only.*

## 3.1 Main results on the test queries

| system | n_queries | n_no_results | Recall@5 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | MAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p2_abstract_tfidf | 558 | 0 | 0.3280 | 0.3710 | 0.5645 | 0.2469 | 0.2769 | 0.2550 |
| p2_abstract_bm25 | 558 | 0 | 0.3566 | 0.4086 | 0.5914 | 0.2695 | 0.3028 | 0.2766 |
| p2_abstract_ql | 558 | 0 | 0.3513 | 0.4194 | 0.5878 | 0.2640 | 0.3010 | 0.2704 |
| p2_abstract_rm3 | 558 | 0 | 0.3620 | 0.4032 | 0.5932 | 0.2726 | 0.3042 | 0.2800 |
| p2_context_tfidf | 558 | 0 | 0.0771 | 0.0968 | 0.3620 | 0.0485 | 0.0599 | 0.0570 |
| p2_context_bm25 | 558 | 0 | 0.3333 | 0.3817 | 0.5735 | 0.2624 | 0.2908 | 0.2696 |
| p2_context_ql | 558 | 0 | 0.3692 | 0.4265 | 0.5699 | 0.2914 | 0.3237 | 0.2966 |
| p2_context_rm3 | 558 | 0 | 0.3190 | 0.3584 | 0.5233 | 0.2653 | 0.2876 | 0.2707 |
| p2_expanded_tfidf | 558 | 0 | 0.2885 | 0.3692 | 0.6541 | 0.2058 | 0.2445 | 0.2179 |
| p2_expanded_bm25 | 558 | 0 | 0.4767 | 0.5251 | 0.7222 | 0.3912 | 0.4237 | 0.3990 |
| p2_expanded_ql | 558 | 0 | 0.4498 | 0.4875 | 0.6487 | 0.3570 | 0.3888 | 0.3630 |
| p2_expanded_rm3 | 558 | 0 | 0.4659 | 0.5305 | 0.6989 | 0.3796 | 0.4158 | 0.3857 |
| p2_fielded_bm25f | 558 | 0 | 0.4552 | 0.5090 | 0.6900 | 0.3568 | 0.3936 | 0.3635 |
| p3_minilm_abstract | 558 | 0 | 0.3029 | 0.3781 | 0.6201 | 0.2317 | 0.2663 | 0.2421 |
| p3_minilm_contexts | 558 | 0 | 0.3029 | 0.3674 | 0.5914 | 0.2235 | 0.2577 | 0.2326 |
| p3_minilm_expanded | 558 | 0 | 0.3943 | 0.4731 | 0.7240 | 0.3015 | 0.3421 | 0.3117 |
| p3_hybrid_rrf | 558 | 0 | 0.5197 | 0.5986 | 0.7796 | 0.3990 | 0.4468 | 0.4066 |


How to read it: every system answers all 558 queries (`n_no_results` is 0), so no query is scored as a missing result. Recall@100 is comparable across systems; MAP is not exactly comparable between P2 (100 results per query) and P3 (1,000), see section 1.3.

Main observations (MRR@10 unless stated):

* **Highest point estimates: the hybrid (RRF of dense `minilm_expanded` and BM25 expanded), MRR@10 0.3990** (not significantly above expanded BM25 on MRR@10, section 3.2), with the highest Recall@10 (0.5986), nDCG@10 (0.4468) and Recall@100 (0.7796). The best purely lexical system is **expanded BM25 (0.3912)**; the best dense system is `minilm_expanded` (0.3015).
* **Representation (lexical):** expanded (abstract plus citing sentences) is the strongest for BM25 (0.3912 vs 0.2695 abstract-only and 0.2624 context-only) and for RM3 and QL; context-only is weak except with query likelihood (0.2914) and
  nearly useless with TF-IDF (0.0485). Expanded TF-IDF (0.2058) is *worse* than abstract-only TF-IDF (0.2469), i.e. expansion does not help every scoring model.
* **Representation (dense):** expanded 0.3015 > abstract 0.2317 > contexts 0.2235; context-only embeddings are no better than abstracts.
* **Lexical vs dense:** with the same representation BM25 beats the MiniLM dense system for the expanded representation (0.3912 vs 0.3015) and is slightly ahead for abstracts (0.2695 vs 0.2317).
* **Validation and test agree on the main ordering:** on the 549 validation queries the hybrid (0.3865) is ahead of expanded BM25 (0.3669), dense expanded (0.2866) and abstract-only BM25 (0.2502), as on test; the selection on validation did not visibly overfit (smaller differences between neighbouring systems can change places).

Validation results (analysis only):

| system | n_queries | n_no_results | Recall@5 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | MAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p2_abstract_tfidf | 549 | 0 | 0.3005 | 0.3716 | 0.5774 | 0.2112 | 0.2494 | 0.2186 |
| p2_abstract_bm25 | 549 | 0 | 0.3352 | 0.4007 | 0.6029 | 0.2502 | 0.2858 | 0.2572 |
| p2_abstract_ql | 549 | 0 | 0.3297 | 0.3989 | 0.5902 | 0.2445 | 0.2813 | 0.2514 |
| p2_abstract_rm3 | 549 | 0 | 0.3279 | 0.3789 | 0.5974 | 0.2451 | 0.2773 | 0.2529 |
| p2_context_tfidf | 549 | 0 | 0.0856 | 0.1093 | 0.3916 | 0.0536 | 0.0669 | 0.0626 |
| p2_context_bm25 | 549 | 0 | 0.3242 | 0.3825 | 0.5829 | 0.2629 | 0.2910 | 0.2699 |
| p2_context_ql | 549 | 0 | 0.3479 | 0.3880 | 0.5902 | 0.2860 | 0.3105 | 0.2934 |
| p2_context_rm3 | 549 | 0 | 0.3060 | 0.3479 | 0.5301 | 0.2588 | 0.2802 | 0.2644 |
| p2_expanded_tfidf | 549 | 0 | 0.2678 | 0.3607 | 0.6375 | 0.1825 | 0.2243 | 0.1940 |
| p2_expanded_bm25 | 549 | 0 | 0.4627 | 0.5137 | 0.7122 | 0.3669 | 0.4023 | 0.3749 |
| p2_expanded_ql | 549 | 0 | 0.4244 | 0.4754 | 0.6594 | 0.3349 | 0.3690 | 0.3423 |
| p2_expanded_rm3 | 549 | 0 | 0.4645 | 0.5118 | 0.7086 | 0.3597 | 0.3967 | 0.3665 |
| p2_fielded_bm25f | 549 | 0 | 0.4499 | 0.5137 | 0.6976 | 0.3426 | 0.3840 | 0.3494 |
| p3_minilm_abstract | 549 | 0 | 0.3097 | 0.3807 | 0.6029 | 0.2175 | 0.2561 | 0.2273 |
| p3_minilm_contexts | 549 | 0 | 0.3097 | 0.3698 | 0.5738 | 0.2280 | 0.2617 | 0.2366 |
| p3_minilm_expanded | 549 | 0 | 0.3862 | 0.4718 | 0.7031 | 0.2866 | 0.3306 | 0.2969 |
| p3_hybrid_rrf | 549 | 0 | 0.5155 | 0.5756 | 0.7668 | 0.3865 | 0.4321 | 0.3944 |


## 3.2 Significance of the pre-specified comparisons

Paired Wilcoxon signed-rank tests on the same 558 test queries, 95% bootstrap confidence intervals for the mean difference, Holm correction over the 16 comparisons of each metric (list fixed from the research questions, see section 1.4). MRR@10:

| Q | Metric | System | vs | mean sys | mean base | diff | 95% CI | p (Holm) | verdict |
|---|---|---|---|---:|---:|---:|---|---:|---|
| RQ1 | MRR@10 | p2_expanded_bm25 | p2_abstract_bm25 | 0.3912 | 0.2695 | +0.1217 | [+0.0960, +0.1481] | 2.76e-17 | significant, better |
| RQ1 | MRR@10 | p2_context_bm25 | p2_abstract_bm25 | 0.2624 | 0.2695 | -0.0071 | [-0.0482, +0.0349] | 1 | not significant |
| RQ1 | MRR@10 | p2_expanded_bm25 | p2_context_bm25 | 0.3912 | 0.2624 | +0.1289 | [+0.0958, +0.1618] | 6.42e-12 | significant, better |
| RQ1 | MRR@10 | p3_minilm_expanded | p3_minilm_abstract | 0.3015 | 0.2317 | +0.0698 | [+0.0412, +0.0984] | 4.76e-05 | significant, better |
| RQ1 | MRR@10 | p3_minilm_contexts | p3_minilm_abstract | 0.2235 | 0.2317 | -0.0082 | [-0.0475, +0.0311] | 1 | not significant |
| RQ2 | MRR@10 | p3_minilm_abstract | p2_abstract_bm25 | 0.2317 | 0.2695 | -0.0378 | [-0.0677, -0.0075] | 0.0811 | not significant |
| RQ2 | MRR@10 | p3_minilm_expanded | p2_expanded_bm25 | 0.3015 | 0.3912 | -0.0897 | [-0.1219, -0.0588] | 1.7e-06 | significant, worse |
| RQ2 | MRR@10 | p3_hybrid_rrf | p2_expanded_bm25 | 0.3990 | 0.3912 | +0.0078 | [-0.0137, +0.0295] | 1 | not significant |
| RQ2 | MRR@10 | p3_hybrid_rrf | p3_minilm_expanded | 0.3990 | 0.3015 | +0.0975 | [+0.0762, +0.1192] | 3.34e-17 | significant, better |
| P2 | MRR@10 | p2_expanded_bm25 | p2_expanded_rm3 | 0.3912 | 0.3796 | +0.0116 | [-0.0044, +0.0275] | 1 | not significant |
| P2 | MRR@10 | p2_expanded_bm25 | p2_expanded_ql | 0.3912 | 0.3570 | +0.0342 | [+0.0212, +0.0480] | 7.82e-06 | significant, better |
| P2 | MRR@10 | p2_expanded_bm25 | p2_fielded_bm25f | 0.3912 | 0.3568 | +0.0344 | [+0.0146, +0.0551] | 0.00759 | significant, better |
| RQ3 | MRR@10 | p2_abstract_bm25+authority | p2_abstract_bm25 | 0.3127 | 0.2695 | +0.0432 | [+0.0250, +0.0616] | 6.98e-07 | significant, better |
| RQ3 | MRR@10 | p2_expanded_bm25+authority | p2_expanded_bm25 | 0.3991 | 0.3912 | +0.0079 | [-0.0008, +0.0166] | 0.266 | not significant |
| RQ3 | MRR@10 | p3_minilm_expanded+authority | p3_minilm_expanded | 0.2994 | 0.3015 | -0.0020 | [-0.0098, +0.0054] | 1 | not significant |
| RQ3 | MRR@10 | p3_hybrid_rrf+authority | p3_hybrid_rrf | 0.3944 | 0.3990 | -0.0047 | [-0.0141, +0.0045] | 1 | not significant |


What is, and is not, established (Holm-corrected, MRR@10; Recall@10 and nDCG@10 in section 3.10 give the same picture except where noted):

* **RQ1, expansion helps lexical retrieval: yes.** Expanded BM25 beats abstract-only BM25 by +0.122 (CI [+0.096, +0.148]) and context-only BM25 by +0.129. The same holds for nDCG@10 and Recall@10.
* **RQ1, dense: expansion helps over the abstract (+0.070, CI [+0.041, +0.098]) but context-only dense is not better than abstract-only** (-0.008, not significant). Adding contexts to BM25 *context-only* does not beat the abstract either (-0.007, not significant).
* **RQ2, lexical vs dense:** dense expanded is significantly *worse* than BM25 expanded (-0.090, CI [-0.122, -0.059]); dense abstract vs BM25 abstract (-0.038) is **not** significant after correction.
* **RQ2, hybrid:** the hybrid is significantly better than dense expanded (+0.098) but **not significantly better than expanded BM25 on MRR@10 (+0.008, CI [-0.014, +0.030])**. For Recall@10 the hybrid is significantly better than expanded BM25 (+0.074, CI [+0.047, +0.100]); for nDCG@10 (+0.023) it is not significant after correction.
  So the hybrid's advantage is at deeper cut-offs (more relevant papers in the top 10), not in the rank of the first hit.
* **Within P2:** BM25 beats QL (+0.034) and BM25F (+0.034) significantly; BM25 vs RM3 (+0.012) is not significant.
* **RQ3, authority prior:** see section 2.3: significant only for abstract-only BM25 (+0.043 MRR@10); not for expanded BM25 (+0.008, p = 0.27), dense (-0.002) or hybrid (-0.005).

Recall@10 and nDCG@10 versions of this table are in section 3.10. In total 22 of the 48 (16 comparisons x 3 metrics) are significant after Holm correction; we report only those as differences, and call the others "no evidence of a difference".

## 3.3 Residual leakage inside the index (leak B) and a robustness analysis

P1 keeps other sentences of the query's own citing paper in the retrieval index (section 1.5): 302 of the 558 test queries (54.1%) have another sentence from the same paper citing the same target in the index, and 445 (79.7%) have some other sentence from that paper.
This is exposure P4 could not remove, but it can be measured. For each system we compare the mean MRR@10 of queries with and without a same-paper, same-target sentence in the index (observational: groups may differ in other ways):

| system | n with | n without | MRR@10 with | MRR@10 without | difference (95% CI) |
|---|---:|---:|---:|---:|---|
| p2_abstract_bm25 | 302 | 256 | 0.284 | 0.253 | +0.031 [-0.036, +0.097] |
| p2_context_bm25 | 302 | 256 | 0.367 | 0.139 | +0.228 [+0.165, +0.294] |
| p2_expanded_bm25 | 302 | 256 | 0.465 | 0.305 | +0.160 [+0.086, +0.234] |
| p3_minilm_abstract | 302 | 256 | 0.245 | 0.215 | +0.030 [-0.032, +0.092] |
| p3_minilm_contexts | 302 | 256 | 0.306 | 0.127 | +0.179 [+0.120, +0.240] |
| p3_minilm_expanded | 302 | 256 | 0.377 | 0.213 | +0.164 [+0.099, +0.230] |
| p3_hybrid_rrf | 302 | 256 | 0.475 | 0.309 | +0.166 [+0.095, +0.236] |


Systems that use citation contexts (context-only, expanded, hybrid) score **+0.16 to +0.23 MRR@10 higher** on exposed queries, with confidence intervals far from zero, whereas the abstract-only systems, which cannot use another sentence of the citing paper, show
only +0.03 with intervals that include zero. This is strong evidence that part of the advantage of the context-based systems on the full test set is due to the residual same-paper exposure, not only to genuinely better retrieval.

**Exploratory robustness check (defined after the main test results were known, therefore not a pre-specified analysis).** We repeated the 16 comparisons on the 256 test queries that have *no* same-paper, same-target sentence in the index (Holm over those 16):

| Q | Metric | System | vs | mean sys | mean base | diff | 95% CI | p (Holm) | verdict |
|---|---|---|---|---:|---:|---:|---|---:|---|
| RQ1 | MRR@10 | p2_expanded_bm25 | p2_abstract_bm25 | 0.3046 | 0.2525 | +0.0520 | [+0.0244, +0.0807] | 0.00703 | significant, better |
| RQ1 | MRR@10 | p2_context_bm25 | p2_abstract_bm25 | 0.1387 | 0.2525 | -0.1138 | [-0.1658, -0.0630] | 0.000293 | significant, worse |
| RQ1 | MRR@10 | p2_expanded_bm25 | p2_context_bm25 | 0.3046 | 0.1387 | +0.1659 | [+0.1205, +0.2113] | 4.06e-10 | significant, better |
| RQ1 | MRR@10 | p3_minilm_expanded | p3_minilm_abstract | 0.2128 | 0.2155 | -0.0027 | [-0.0398, +0.0351] | 1 | not significant |
| RQ1 | MRR@10 | p3_minilm_contexts | p3_minilm_abstract | 0.1265 | 0.2155 | -0.0890 | [-0.1428, -0.0350] | 0.0109 | significant, worse |
| RQ2 | MRR@10 | p3_minilm_abstract | p2_abstract_bm25 | 0.2155 | 0.2525 | -0.0370 | [-0.0809, +0.0058] | 0.324 | not significant |
| RQ2 | MRR@10 | p3_minilm_expanded | p2_expanded_bm25 | 0.2128 | 0.3046 | -0.0918 | [-0.1365, -0.0474] | 0.00135 | significant, worse |
| RQ2 | MRR@10 | p3_hybrid_rrf | p2_expanded_bm25 | 0.3092 | 0.3046 | +0.0047 | [-0.0259, +0.0353] | 1 | not significant |
| RQ2 | MRR@10 | p3_hybrid_rrf | p3_minilm_expanded | 0.3092 | 0.2128 | +0.0965 | [+0.0681, +0.1260] | 2.28e-09 | significant, better |
| P2 | MRR@10 | p2_expanded_bm25 | p2_expanded_rm3 | 0.3046 | 0.2814 | +0.0231 | [-0.0003, +0.0473] | 0.36 | not significant |
| P2 | MRR@10 | p2_expanded_bm25 | p2_expanded_ql | 0.3046 | 0.2740 | +0.0305 | [+0.0107, +0.0521] | 0.0412 | significant, better |
| P2 | MRR@10 | p2_expanded_bm25 | p2_fielded_bm25f | 0.3046 | 0.3122 | -0.0076 | [-0.0322, +0.0169] | 1 | not significant |
| RQ3 | MRR@10 | p2_abstract_bm25+authority | p2_abstract_bm25 | 0.3333 | 0.2525 | +0.0808 | [+0.0573, +0.1050] | 1.96e-09 | significant, better |
| RQ3 | MRR@10 | p2_expanded_bm25+authority | p2_expanded_bm25 | 0.3265 | 0.3046 | +0.0219 | [+0.0095, +0.0353] | 0.00029 | significant, better |
| RQ3 | MRR@10 | p3_minilm_expanded+authority | p3_minilm_expanded | 0.2199 | 0.2128 | +0.0072 | [-0.0001, +0.0151] | 0.0206 | significant, better |
| RQ3 | MRR@10 | p3_hybrid_rrf+authority | p3_hybrid_rrf | 0.3218 | 0.3092 | +0.0126 | [+0.0053, +0.0210] | 0.00476 | significant, better |


On these queries: expanded BM25 is still significantly better than abstract-only BM25, but by **+0.052 instead of +0.122**, so more than half of the full-set gain disappears; **dense expansion no longer helps** (-0.003, not significant) and **context-only systems are significantly worse than abstract-only**
(BM25: -0.114; dense: -0.089). The hybrid still beats dense expanded but is not better than expanded BM25 (+0.005, not significant). The authority prior is significant after correction for all four fused systems on this subset (largest for abstract-only, +0.081; for dense expanded the interval just touches zero, [-0.0001, +0.0151]);
unexposed queries are those the same-paper context cannot rescue, so a popularity signal has more room to matter. The ranking of the systems is therefore more fragile than the headline test table suggests, and conclusions about *citation contexts* should be read with the unexposed numbers in mind.
Rerunning P2 and P3 with a strict index (no sentence of any held-out citing paper) would settle this; it was out of reach for P4.

## 3.4 Popularity bias and cold start

Popularity is the number of citing papers inside the corpus, taken from the leakage-safe evaluation graph (the full-graph count would include each query's own citation). Buckets are those of the handoff and `dataset_stats.md`
(1-2, 3-10, 11-50, >50), plus a bucket **0** for papers with no citation in the evaluation graph. Groups with fewer than 10 queries (>50: 6 queries) are shown but not interpreted.

### Performance by target popularity

MRR@10 on the test queries, by the number of citations of the relevant paper:

| system \ citations of the target | 0 (n=143) | 1-2 (n=267) | 3-10 (n=123) | 11-50 (n=19) | >50 (n=6, small) |
|---|---:|---:|---:|---:|---:|
| p2_abstract_bm25 | 0.368 | 0.252 | 0.208 | 0.226 | 0.102 |
| p2_context_bm25 | 0.281 | 0.191 | 0.374 | 0.377 | 0.339 |
| p2_expanded_bm25 | 0.448 | 0.335 | 0.410 | 0.583 | 0.556 |
| p3_minilm_abstract | 0.310 | 0.213 | 0.216 | 0.083 | 0.000 |
| p3_minilm_contexts | 0.249 | 0.143 | 0.309 | 0.501 | 0.589 |
| p3_minilm_expanded | 0.361 | 0.231 | 0.341 | 0.492 | 0.575 |
| p3_hybrid_rrf | 0.453 | 0.328 | 0.444 | 0.583 | 0.742 |
| p2_abstract_bm25+authority | 0.268 | 0.332 | 0.314 | 0.404 | 0.208 |
| p2_expanded_bm25+authority | 0.408 | 0.355 | 0.444 | 0.608 | 0.561 |
| p3_minilm_expanded+authority | 0.322 | 0.246 | 0.344 | 0.493 | 0.575 |
| p3_hybrid_rrf+authority | 0.376 | 0.349 | 0.466 | 0.596 | 0.750 |


* **Abstract-only systems do not benefit from popularity** (BM25: 0.252 for 1-2 citations, 0.208 for 3-10, 0.226 for 11-50; the trend over popularity is not significant after correction). **Context-using systems do**: context-only BM25 rises from 0.191 (1-2 citations) to 0.374 (3-10) and 0.377 (11-50); expanded BM25 from 0.335 to 0.410 and 0.583; the hybrid from 0.328 to 0.444 and 0.583. A paper that is cited more has more sentences that can match a query.
* **The bucket with 0 evaluation citations looks easy but is not a cold-start test.** These 143 test queries (25.6%) target papers cited only by validation or test papers, so every index sentence about them comes from a held-out citing paper.
  For 134 of the 143 (93.7%) the index contains another sentence of the *same* citing paper that cites the same target (leak B, section 3.3); only 9 queries are free of it:

| citations of the target (evaluation graph) | test queries | with a same-paper, same-target sentence in the index | share |
|---|---:|---:|---|
| 0 | 143 | 134 | 93.7% |
| 1-2 | 267 | 93 | 34.8% |
| 3-10 | 123 | 60 | 48.8% |
| 11-50 | 19 | 12 | 63.2% |
| >50 | 6 | 3 | 50.0% |


  Expanded BM25 scores 0.462 on the exposed queries of this bucket (0.236 on the 9 unexposed ones, too few to interpret), the same level as exposed queries in the other buckets, so the index holds sentences about these papers and they are not "cold" for a context-based system. Exposure is probably not the only difference: abstract-only BM25, which cannot use the citing paper's other sentences, also scores higher
  here (0.368) than for 1-2 citations (0.252), and we did not investigate why. Either way, the benchmark cannot support a clean cold-start estimate for papers without citing context:

| system | citations of target | n exposed | MRR@10 exposed | n not exposed | MRR@10 not exposed |
|---|---|---:|---:|---:|---:|
| p2_abstract_bm25 | 0 | 134 | 0.376 | 9 | 0.241 |
| p2_abstract_bm25 | 1-2 | 93 | 0.247 | 174 | 0.255 |
| p2_abstract_bm25 | 3-10 | 60 | 0.169 | 63 | 0.245 |
| p2_abstract_bm25 | 11-50 | 12 | 0.134 | 7 | 0.383 |
| p2_abstract_bm25 | >50 | 3 | 0.204 | 3 | 0.000 |
| p2_expanded_bm25 | 0 | 134 | 0.462 | 9 | 0.236 |
| p2_expanded_bm25 | 1-2 | 93 | 0.442 | 174 | 0.278 |
| p2_expanded_bm25 | 3-10 | 60 | 0.502 | 63 | 0.323 |
| p2_expanded_bm25 | 11-50 | 12 | 0.438 | 7 | 0.833 |
| p2_expanded_bm25 | >50 | 3 | 0.667 | 3 | 0.444 |
| p3_hybrid_rrf | 0 | 134 | 0.471 | 9 | 0.196 |
| p3_hybrid_rrf | 1-2 | 93 | 0.412 | 174 | 0.283 |
| p3_hybrid_rrf | 3-10 | 60 | 0.550 | 63 | 0.344 |
| p3_hybrid_rrf | 11-50 | 12 | 0.576 | 7 | 0.595 |
| p3_hybrid_rrf | >50 | 3 | 0.733 | 3 | 0.750 |


### Do the systems retrieve popular papers?

For the top-10 list of each system, the share of retrieved papers that are cited at least once (21.4% of the corpus; with 79% of papers uncited the "most-cited 10%" cut-off collapses to this), the median citation count of a retrieved paper, and how often a wrong top-1 result is more cited than the true target:

| system | share of top-10 papers cited at least once (corpus 21.4%) | median citations of a top-10 paper | queries with a wrong top-1 | wrong top-1 more cited than the target |
|---|---:|---:|---:|---:|
| p2_abstract_bm25 | 0.263 | 0.000 | 442 | 0.090 |
| p2_context_bm25 | 0.910 | 1.500 | 440 | 0.452 |
| p2_expanded_bm25 | 0.465 | 0.000 | 378 | 0.270 |
| p3_minilm_abstract | 0.264 | 0.000 | 462 | 0.091 |
| p3_minilm_contexts | 0.944 | 4.000 | 466 | 0.727 |
| p3_minilm_expanded | 0.697 | 2.000 | 429 | 0.562 |
| p3_hybrid_rrf | 0.608 | 1.000 | 384 | 0.521 |
| p2_abstract_bm25+authority | 0.875 | 1.000 | 420 | 0.367 |
| p2_expanded_bm25+authority | 0.699 | 1.000 | 374 | 0.447 |
| p3_minilm_expanded+authority | 0.825 | 2.000 | 431 | 0.657 |
| p3_hybrid_rrf+authority | 0.810 | 2.000 | 388 | 0.662 |


* **Context-using systems retrieve popular papers**: 91-94% of the top-10 of context-only systems are cited at least once (corpus 21%), 47-70% for expanded BM25 and dense, versus 26% for the abstract-only systems, whose lists reflect the corpus (and can therefore reach uncited papers). Context-only systems can only return papers that have at least one indexed context.
* **The authority prior strengthens this bias.** Fusing with the leakage-safe prior raises the share of top-10 papers that are cited from 26% to 88% (abstract-only BM25), 47% to 70% (expanded BM25), 70% to 83% (dense) and 61% to 81% (hybrid), and the rate at which a wrong top-1 is more cited than the target
  rises by 28 points for abstract-only BM25 (9% to 37%) and by 14 points for the hybrid (52% to 66%). Section 2.3 showed that this buys a significant accuracy gain only for abstract-only BM25; for expanded BM25, dense and hybrid it moves retrieval toward popular papers **without a measurable accuracy gain**.

### Cold start: how many contexts does the target have?

The number of indexed citing sentences of the target paper is the direct cold-start variable for a context-based system. The benchmark keeps only queries whose target has at least one indexed context (16 queries were dropped by P1 for having none), so **targets with zero indexed contexts cannot occur**; the
experiment below compares targets with 1, 2-5, 6-20 and more contexts. Note that these counts include same-paper sentences (section 3.3).

| system \ indexed contexts of the target | 0 (n=0, small) | 1 (n=290) | 2-5 (n=127) | 6-20 (n=106) | 21-100 (n=29) | >100 (n=6, small) |
|---|---:|---:|---:|---:|---:|---:|
| p2_abstract_bm25 |  | 0.303 | 0.238 | 0.246 | 0.191 | 0.102 |
| p2_context_bm25 |  | 0.174 | 0.272 | 0.453 | 0.392 | 0.339 |
| p2_expanded_bm25 |  | 0.343 | 0.356 | 0.523 | 0.511 | 0.556 |
| p3_minilm_abstract |  | 0.271 | 0.171 | 0.253 | 0.075 | 0.000 |
| p3_minilm_contexts |  | 0.132 | 0.190 | 0.430 | 0.454 | 0.589 |
| p3_minilm_expanded |  | 0.261 | 0.234 | 0.442 | 0.434 | 0.575 |
| p3_hybrid_rrf |  | 0.357 | 0.329 | 0.545 | 0.526 | 0.742 |
| p2_abstract_bm25+authority |  | 0.306 | 0.303 | 0.343 | 0.333 | 0.208 |
| p2_expanded_bm25+authority |  | 0.337 | 0.388 | 0.536 | 0.534 | 0.561 |
| p3_minilm_expanded+authority |  | 0.255 | 0.240 | 0.439 | 0.435 | 0.575 |
| p3_hybrid_rrf+authority |  | 0.338 | 0.341 | 0.552 | 0.538 | 0.750 |


| system | Spearman rho (95% CI) | p (Holm) | significant (Holm) |
|---|---|---:|---|
| p2_abstract_bm25 | -0.057 [-0.139, +0.023] | 0.467 | no |
| p2_context_bm25 | +0.311 [+0.230, +0.389] | 7.18e-13 | yes |
| p2_expanded_bm25 | +0.152 [+0.067, +0.231] | 0.00133 | yes |
| p3_minilm_abstract | -0.061 [-0.140, +0.024] | 0.467 | no |
| p3_minilm_contexts | +0.347 [+0.266, +0.426] | 5.14e-16 | yes |
| p3_minilm_expanded | +0.182 [+0.097, +0.264] | 9.64e-05 | yes |
| p3_hybrid_rrf | +0.170 [+0.088, +0.251] | 0.000296 | yes |
| p2_abstract_bm25+authority | +0.046 [-0.037, +0.124] | 0.467 | no |
| p2_expanded_bm25+authority | +0.192 [+0.108, +0.269] | 3.87e-05 | yes |
| p3_minilm_expanded+authority | +0.203 [+0.117, +0.286] | 1.31e-05 | yes |
| p3_hybrid_rrf+authority | +0.201 [+0.119, +0.282] | 1.55e-05 | yes |


For the context-only and expanded systems the score rises significantly with the number of indexed contexts (context-only BM25 and dense: Spearman 0.31 and 0.35; expanded and hybrid 0.15-0.18, 0.19-0.20 with the prior; Holm-corrected), whereas abstract-only systems show no trend (-0.06, not significant). A target with a single context is hardest for the context-only systems (BM25 0.174, dense 0.132)
and is where the abstract helps: expanded BM25 (0.343) beats context-only BM25 (0.174) there.

**Implications for cold start.** 76-79% of the corpus has no citation inside it and can only be found through its abstract, and the benchmark never asks for such a paper (every target is cited at least once in the full graph). So the evaluation measures how well *already cited* papers are found; it says nothing direct about
retrieving new papers. What it does show is that the context-based and prior-augmented systems concentrate their results on cited papers (previous table), so they would rank a new, uncited paper behind well-cited ones; only the abstract-only systems keep uncited papers in their top-10 (74% of their lists).
A fair cold-start experiment needs queries about papers with no citing context in the index and a strict index; this corpus is too small and its split design (leak B) prevents it.

## 3.5 Publication year (temporal analysis)

The year analysis is limited by the corpus: 463 of the 558 test queries (83%) target papers from 2020-2022 (110, 246 and 107), 2019 has 44, 2018 has 28, 2017 has 10, and the earlier years have fewer than 10 queries each (shown, not interpreted).
MRR@10 by the publication year of the target:

| system \ publication year of the target | 2011 (n=2, small) | 2014 (n=2, small) | 2016 (n=9, small) | 2017 (n=10) | 2018 (n=28) | 2019 (n=44) | 2020 (n=110) | 2021 (n=246) | 2022 (n=107) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p2_abstract_bm25 | 0.000 | 0.000 | 0.321 | 0.069 | 0.273 | 0.187 | 0.252 | 0.299 | 0.277 |
| p2_context_bm25 | 0.556 | 0.000 | 0.290 | 0.212 | 0.345 | 0.285 | 0.209 | 0.299 | 0.203 |
| p2_expanded_bm25 | 0.500 | 0.000 | 0.578 | 0.233 | 0.424 | 0.350 | 0.392 | 0.402 | 0.378 |
| p3_minilm_abstract | 0.050 | 0.000 | 0.222 | 0.100 | 0.329 | 0.181 | 0.194 | 0.260 | 0.222 |
| p3_minilm_contexts | 0.500 | 0.000 | 0.250 | 0.228 | 0.262 | 0.282 | 0.202 | 0.252 | 0.143 |
| p3_minilm_expanded | 0.500 | 0.000 | 0.370 | 0.210 | 0.333 | 0.333 | 0.253 | 0.348 | 0.227 |
| p3_hybrid_rrf | 0.500 | 0.000 | 0.556 | 0.275 | 0.433 | 0.389 | 0.381 | 0.430 | 0.345 |
| p2_abstract_bm25+authority | 0.100 | 0.000 | 0.358 | 0.158 | 0.314 | 0.234 | 0.291 | 0.345 | 0.313 |
| p2_expanded_bm25+authority | 0.562 | 0.000 | 0.509 | 0.262 | 0.406 | 0.363 | 0.398 | 0.420 | 0.373 |
| p3_minilm_expanded+authority | 0.500 | 0.000 | 0.370 | 0.211 | 0.340 | 0.336 | 0.270 | 0.343 | 0.209 |
| p3_hybrid_rrf+authority | 0.600 | 0.000 | 0.528 | 0.300 | 0.377 | 0.387 | 0.373 | 0.435 | 0.332 |


No system shows a supported trend of the score with the target's publication year (Spearman between 0.07 and -0.04 for all systems, all corrected p-values above 0.95), so **we find no evidence that older or newer targets are retrieved better**. This is a statement about 2017-2022 only; the data cannot say anything about older papers.

| system | Spearman rho (95% CI) | p (Holm) | significant (Holm) |
|---|---|---:|---|
| p2_abstract_bm25 | +0.073 [-0.005, +0.156] | 0.954 | no |
| p2_context_bm25 | +0.011 [-0.071, +0.093] | 1 | no |
| p2_expanded_bm25 | +0.028 [-0.053, +0.113] | 1 | no |
| p3_minilm_abstract | +0.050 [-0.035, +0.135] | 1 | no |
| p3_minilm_contexts | -0.035 [-0.115, +0.048] | 1 | no |
| p3_minilm_expanded | +0.004 [-0.079, +0.090] | 1 | no |
| p3_hybrid_rrf | +0.021 [-0.061, +0.105] | 1 | no |
| p2_abstract_bm25+authority | +0.069 [-0.010, +0.150] | 1 | no |
| p2_expanded_bm25+authority | +0.017 [-0.063, +0.101] | 1 | no |
| p3_minilm_expanded+authority | -0.016 [-0.098, +0.069] | 1 | no |
| p3_hybrid_rrf+authority | +0.012 [-0.071, +0.096] | 1 | no |


## 3.6 Citation function (predicted labels)

Each query sentence is joined to P1's `context_functions.parquet` through its `context_id`. **The labels are predictions, not annotations.** They come from P1's text classifier trained on SciCite, a different corpus; its test accuracy on SciCite is about 0.81-0.82 and
**its accuracy on this corpus is unknown** (no sample was hand-labelled). The predicted distribution on the test queries is far from SciCite's training distribution (59% background, 28% method, 13% result): here 239 queries (43%) are labelled *background*, 317 (57%) *method* and only 2 *result*,
which suggests a domain shift, so the labels should be read as "looks like a method citation" rather than as facts. The *result* group (2 queries) is too small to use.

MRR@10 by predicted function:

| system \ predicted citation function | background (n=239) | method (n=317) | result (n=2, small) |
|---|---:|---:|---:|
| p2_abstract_bm25 | 0.220 | 0.305 | 0.500 |
| p2_context_bm25 | 0.218 | 0.294 | 0.500 |
| p2_expanded_bm25 | 0.332 | 0.435 | 0.500 |
| p3_minilm_abstract | 0.223 | 0.238 | 0.250 |
| p3_minilm_contexts | 0.213 | 0.229 | 0.500 |
| p3_minilm_expanded | 0.290 | 0.309 | 0.500 |
| p3_hybrid_rrf | 0.351 | 0.434 | 0.500 |
| p2_abstract_bm25+authority | 0.275 | 0.340 | 0.500 |
| p2_expanded_bm25+authority | 0.346 | 0.439 | 0.500 |
| p3_minilm_expanded+authority | 0.296 | 0.301 | 0.500 |
| p3_hybrid_rrf+authority | 0.355 | 0.423 | 0.500 |


Systems score higher on sentences predicted as *method* than as *background* (for example expanded BM25 0.435 vs 0.332, the hybrid 0.434 vs 0.351, abstract-only BM25 0.305 vs 0.220), but with two usable groups and Holm correction over the systems the differences are **not statistically supported** (the smallest corrected p is 0.050, for expanded BM25):

| system | Kruskal-Wallis H | p | p (Holm) | significant (Holm) |
|---|---:|---:|---:|---|
| p2_abstract_bm25 | 5.06 | 0.0245 | 0.196 | no |
| p2_context_bm25 | 5.73 | 0.0167 | 0.15 | no |
| p2_expanded_bm25 | 8.05 | 0.00455 | 0.05 | no |
| p3_minilm_abstract | 0.369 | 0.543 | 1 | no |
| p3_minilm_contexts | 0.201 | 0.654 | 1 | no |
| p3_minilm_expanded | 0.129 | 0.719 | 1 | no |
| p3_hybrid_rrf | 4.85 | 0.0276 | 0.196 | no |
| p2_abstract_bm25+authority | 2.74 | 0.0982 | 0.589 | no |
| p2_expanded_bm25+authority | 6.59 | 0.0102 | 0.102 | no |
| p3_minilm_expanded+authority | 0.00021 | 0.988 | 1 | no |
| p3_hybrid_rrf+authority | 2.42 | 0.12 | 0.6 | no |


We therefore report a tendency, not a finding. A per-function comparison of "with vs without a method filter" would need P2's method-filter test runs, which were not delivered.

## 3.7 Error analysis

Failure is a miss at rank 10 (the relevant paper is not in the top 10). Counts out of 558 test queries; "not in the returned list" is the number of queries whose relevant paper is not among the 100 (P2) or 1,000 (P3) results at all:

| system | hits in top 10 | miss@10 rate | median rank when found | rank 1 | rank 2-10 | rank 11+ | not in the returned list |
|---|---:|---:|---:|---:|---:|---:|---:|
| p2_abstract_bm25 | 228 | 0.591 | 3.000 | 116 | 112 | 102 | 228 |
| p2_context_bm25 | 213 | 0.618 | 3.000 | 118 | 95 | 107 | 238 |
| p2_expanded_bm25 | 293 | 0.475 | 2.000 | 180 | 113 | 110 | 155 |
| p3_minilm_abstract | 211 | 0.622 | 14.000 | 96 | 115 | 260 | 87 |
| p3_minilm_contexts | 205 | 0.633 | 26.000 | 92 | 113 | 297 | 56 |
| p3_minilm_expanded | 264 | 0.527 | 10.000 | 129 | 135 | 259 | 35 |
| p3_hybrid_rrf | 334 | 0.401 | 4.000 | 174 | 160 | 192 | 32 |
| p2_abstract_bm25+authority | 250 | 0.552 | 2.000 | 138 | 112 | 80 | 228 |
| p2_expanded_bm25+authority | 305 | 0.453 | 2.000 | 184 | 121 | 98 | 155 |
| p3_minilm_expanded+authority | 266 | 0.523 | 10.000 | 127 | 139 | 257 | 35 |
| p3_hybrid_rrf+authority | 332 | 0.405 | 4.000 | 170 | 162 | 194 | 32 |


* The hybrid fails least (40.1% miss@10, 224 queries); expanded BM25 47.5%, dense expanded 52.7%, abstract-only BM25 59.1%.
* **BM25 is precise when it finds the paper, dense is not**: the median rank of a found relevant paper is 2-3 for BM25 and 10-26 for the dense systems (the hybrid: 4). Dense systems rarely lose the paper entirely (35-87 queries not in the 1,000 returned) but rank it deep, whereas P2 lists of 100 papers miss it in 155-238 queries,
  which is also why fusing a P2 list can only reorder 100 papers.
* McNemar tests on hits@10 against abstract-only BM25 (Holm-corrected) agree with the paired tests of section 3.2: expanded BM25 (71 queries gained, 6 lost), the hybrid (115 gained, 9 lost) and dense expanded are significantly better; context-only BM25 (92 gained, 107 lost) and dense abstract are not.
* The authority prior moves hits into the top 10 for abstract-only BM25 (228 to 250) and leaves expanded BM25 almost unchanged (293 to 305), dense (264 to 266) and the hybrid (334 to 332), as in section 2.3.

An error analysis by query sentence length or query field and a set of hand-picked failure cases were **not carried out** (not enough time); the per-query results needed for them are in `p4/outputs/real/run_main/analysis/errors_per_query.csv`.

## 3.8 Efficiency (P2 only)

P2 logged index build time, index size and the median and mean query time of every configuration (measured by P2 on its machine on the validation queries; one measurement, no repetitions). P4 reports them as logged:

| index | build_seconds | size_mib |
|---|---:|---:|
| abstract-w0-plain-all | 3.39 | 36.68 |
| context-w0-plain-all | 0.58 | 4.37 |
| context-w0-plain-method | 0.43 | 2.63 |
| expanded-w0-plain-all | 3.83 | 40.43 |
| expanded-w0-plain-method | 4.18 | 38.75 |
| fielded-w0-plain-all | 4.10 | 43.46 |
| fielded-w0-plain-method | 4.29 | 41.80 |


Query time and index size of the best-on-validation configuration of each representation and model:

| variant | model | val MRR@10 | median ms/query | mean ms/query | index MiB |
|---|---|---:|---:|---:|---:|
| abstract | bm25 | 0.25 | 20.63 | 21.85 | 36.68 |
| abstract | ql | 0.24 | 19.84 | 21.29 | 36.68 |
| abstract | rm3 | 0.25 | 60.22 | 64.45 | 36.68 |
| abstract | tfidf | 0.21 | 20.34 | 22.56 | 36.68 |
| context | bm25 | 0.26 | 2.87 | 3.17 | 4.37 |
| context | ql | 0.29 | 5.49 | 5.76 | 4.37 |
| context | rm3 | 0.26 | 11.13 | 11.82 | 4.37 |
| context | tfidf | 0.05 | 1.54 | 1.67 | 2.63 |
| expanded | bm25 | 0.37 | 24.28 | 26.36 | 40.43 |
| expanded | ql | 0.33 | 20.51 | 21.88 | 40.43 |
| expanded | rm3 | 0.36 | 69.01 | 74.15 | 40.43 |
| expanded | tfidf | 0.18 | 21.35 | 23.40 | 38.75 |
| fielded | bm25f | 0.34 | 30.29 | 33.19 | 43.46 |


Expanded BM25 costs about the same as abstract-only BM25 (about 24 ms against 21 ms median per query; 40 against 37 MiB), while giving the largest lexical gain; context-only indexes are small (4.4 MiB, 2.6 MiB for the TF-IDF index) and fast (1.5-11 ms). RM3's pseudo-relevance feedback triples the query time
(60-69 ms) without a significant gain over BM25 (section 3.2). **No timing or index-size log exists for the P3 dense and hybrid systems, so their efficiency is not reported and was not estimated.**

## 3.9 Limitations and missing experiments

* **Residual same-paper leakage (leak B)** affects the context-based systems (sections 1.5 and 3.3); the strict-index rerun was not possible.
* **SPECTER, P3 timing and the co-citation / recommender evaluation were not available or not done**; nothing is reported for them. Bibliographic coupling and hide-references Recall@k were not implemented. The error analysis by sentence length and query field and the curated failure cases were not carried out; the method-filter test comparison needs P2 test runs that were not delivered.
* **Run depth:** P2 returns 100 papers, P3 1,000; MAP and fusion of P2 lists are limited by that.
* **Predicted citation-function labels, not ground truth** (section 3.6, accuracy on this corpus unknown); the year analysis is limited by the corpus's year skew (83% of targets are from 2020-2022).
* **One corpus (cs.CL, 8,843 papers) and one random split** (seed 42): results are for this dataset and the confidence intervals reflect query sampling only, not variation between splits.
* The evaluation graph protocol is P4's own and was not independently approved by the team.

## 3.10 Recall@10 and nDCG@10 versions of the significance table

Recall@10:

| Q | Metric | System | vs | mean sys | mean base | diff | 95% CI | p (Holm) | verdict |
|---|---|---|---|---:|---:|---:|---|---:|---|
| RQ1 | Recall@10 | p2_expanded_bm25 | p2_abstract_bm25 | 0.5251 | 0.4086 | +0.1165 | [+0.0878, +0.1470] | 2.06e-12 | significant, better |
| RQ1 | Recall@10 | p2_context_bm25 | p2_abstract_bm25 | 0.3817 | 0.4086 | -0.0269 | [-0.0771, +0.0233] | 1 | not significant |
| RQ1 | Recall@10 | p2_expanded_bm25 | p2_context_bm25 | 0.5251 | 0.3817 | +0.1434 | [+0.1039, +0.1828] | 3.66e-10 | significant, better |
| RQ1 | Recall@10 | p3_minilm_expanded | p3_minilm_abstract | 0.4731 | 0.3781 | +0.0950 | [+0.0573, +0.1326] | 1.42e-05 | significant, better |
| RQ1 | Recall@10 | p3_minilm_contexts | p3_minilm_abstract | 0.3674 | 0.3781 | -0.0108 | [-0.0591, +0.0394] | 1 | not significant |
| RQ2 | Recall@10 | p3_minilm_abstract | p2_abstract_bm25 | 0.3781 | 0.4086 | -0.0305 | [-0.0699, +0.0090] | 0.941 | not significant |
| RQ2 | Recall@10 | p3_minilm_expanded | p2_expanded_bm25 | 0.4731 | 0.5251 | -0.0520 | [-0.0950, -0.0090] | 0.151 | not significant |
| RQ2 | Recall@10 | p3_hybrid_rrf | p2_expanded_bm25 | 0.5986 | 0.5251 | +0.0735 | [+0.0466, +0.1004] | 4.77e-06 | significant, better |
| RQ2 | Recall@10 | p3_hybrid_rrf | p3_minilm_expanded | 0.5986 | 0.4731 | +0.1254 | [+0.0932, +0.1577] | 4.38e-12 | significant, better |
| P2 | Recall@10 | p2_expanded_bm25 | p2_expanded_rm3 | 0.5251 | 0.5305 | -0.0054 | [-0.0341, +0.0233] | 1 | not significant |
| P2 | Recall@10 | p2_expanded_bm25 | p2_expanded_ql | 0.5251 | 0.4875 | +0.0376 | [+0.0179, +0.0573] | 0.00178 | significant, better |
| P2 | Recall@10 | p2_expanded_bm25 | p2_fielded_bm25f | 0.5251 | 0.5090 | +0.0161 | [-0.0072, +0.0394] | 1 | not significant |
| RQ3 | Recall@10 | p2_abstract_bm25+authority | p2_abstract_bm25 | 0.4480 | 0.4086 | +0.0394 | [+0.0125, +0.0681] | 0.0677 | not significant |
| RQ3 | Recall@10 | p2_expanded_bm25+authority | p2_expanded_bm25 | 0.5466 | 0.5251 | +0.0215 | [+0.0036, +0.0394] | 0.151 | not significant |
| RQ3 | Recall@10 | p3_minilm_expanded+authority | p3_minilm_expanded | 0.4767 | 0.4731 | +0.0036 | [-0.0125, +0.0197] | 1 | not significant |
| RQ3 | Recall@10 | p3_hybrid_rrf+authority | p3_hybrid_rrf | 0.5950 | 0.5986 | -0.0036 | [-0.0215, +0.0143] | 1 | not significant |


nDCG@10:

| Q | Metric | System | vs | mean sys | mean base | diff | 95% CI | p (Holm) | verdict |
|---|---|---|---|---:|---:|---:|---|---:|---|
| RQ1 | nDCG@10 | p2_expanded_bm25 | p2_abstract_bm25 | 0.4237 | 0.3028 | +0.1209 | [+0.0966, +0.1461] | 2.03e-18 | significant, better |
| RQ1 | nDCG@10 | p2_context_bm25 | p2_abstract_bm25 | 0.2908 | 0.3028 | -0.0120 | [-0.0533, +0.0300] | 1 | not significant |
| RQ1 | nDCG@10 | p2_expanded_bm25 | p2_context_bm25 | 0.4237 | 0.2908 | +0.1329 | [+0.1004, +0.1652] | 4.5e-13 | significant, better |
| RQ1 | nDCG@10 | p3_minilm_expanded | p3_minilm_abstract | 0.3421 | 0.2663 | +0.0759 | [+0.0476, +0.1039] | 2.09e-06 | significant, better |
| RQ1 | nDCG@10 | p3_minilm_contexts | p3_minilm_abstract | 0.2577 | 0.2663 | -0.0086 | [-0.0479, +0.0302] | 1 | not significant |
| RQ2 | nDCG@10 | p3_minilm_abstract | p2_abstract_bm25 | 0.2663 | 0.3028 | -0.0365 | [-0.0667, -0.0059] | 0.115 | not significant |
| RQ2 | nDCG@10 | p3_minilm_expanded | p2_expanded_bm25 | 0.3421 | 0.4237 | -0.0815 | [-0.1139, -0.0501] | 8.06e-06 | significant, worse |
| RQ2 | nDCG@10 | p3_hybrid_rrf | p2_expanded_bm25 | 0.4468 | 0.4237 | +0.0231 | [+0.0036, +0.0428] | 0.197 | not significant |
| RQ2 | nDCG@10 | p3_hybrid_rrf | p3_minilm_expanded | 0.4468 | 0.3421 | +0.1046 | [+0.0840, +0.1261] | 1.25e-18 | significant, better |
| P2 | nDCG@10 | p2_expanded_bm25 | p2_expanded_rm3 | 0.4237 | 0.4158 | +0.0079 | [-0.0089, +0.0242] | 1 | not significant |
| P2 | nDCG@10 | p2_expanded_bm25 | p2_expanded_ql | 0.4237 | 0.3888 | +0.0348 | [+0.0222, +0.0480] | 3.41e-06 | significant, better |
| P2 | nDCG@10 | p2_expanded_bm25 | p2_fielded_bm25f | 0.4237 | 0.3936 | +0.0301 | [+0.0120, +0.0491] | 0.0195 | significant, better |
| RQ3 | nDCG@10 | p2_abstract_bm25+authority | p2_abstract_bm25 | 0.3453 | 0.3028 | +0.0426 | [+0.0250, +0.0603] | 5.64e-05 | significant, better |
| RQ3 | nDCG@10 | p2_expanded_bm25+authority | p2_expanded_bm25 | 0.4343 | 0.4237 | +0.0107 | [+0.0023, +0.0190] | 0.106 | not significant |
| RQ3 | nDCG@10 | p3_minilm_expanded+authority | p3_minilm_expanded | 0.3414 | 0.3421 | -0.0007 | [-0.0084, +0.0069] | 1 | not significant |
| RQ3 | nDCG@10 | p3_hybrid_rrf+authority | p3_hybrid_rrf | 0.4425 | 0.4468 | -0.0043 | [-0.0135, +0.0047] | 1 | not significant |


## 3.11 Authority-prior fusion on the test set (all metrics)

| system | n_queries | n_no_results | Recall@5 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | MAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p2_abstract_bm25 | 558 | 0 | 0.3566 | 0.4086 | 0.5914 | 0.2695 | 0.3028 | 0.2766 |
| p2_expanded_bm25 | 558 | 0 | 0.4767 | 0.5251 | 0.7222 | 0.3912 | 0.4237 | 0.3990 |
| p3_minilm_expanded | 558 | 0 | 0.3943 | 0.4731 | 0.7240 | 0.3015 | 0.3421 | 0.3117 |
| p3_hybrid_rrf | 558 | 0 | 0.5197 | 0.5986 | 0.7796 | 0.3990 | 0.4468 | 0.4066 |
| p2_abstract_bm25+authority | 558 | 0 | 0.4032 | 0.4480 | 0.5914 | 0.3127 | 0.3453 | 0.3192 |
| p2_expanded_bm25+authority | 558 | 0 | 0.4892 | 0.5466 | 0.7222 | 0.3991 | 0.4343 | 0.4062 |
| p3_minilm_expanded+authority | 558 | 0 | 0.3978 | 0.4767 | 0.7312 | 0.2994 | 0.3414 | 0.3099 |
| p3_hybrid_rrf+authority | 558 | 0 | 0.5197 | 0.5950 | 0.7903 | 0.3944 | 0.4425 | 0.4030 |
