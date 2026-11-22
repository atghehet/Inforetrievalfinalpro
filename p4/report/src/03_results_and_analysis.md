# 3. Results and Analysis

*P4 section. **All results are REAL**: 17 systems from P2 and P3 on the real cs.CL corpus, evaluated with P4's harness. Tables are generated from the files in `p4/outputs/real/` by `p4/experiments/make_report.py`. Test numbers come from the frozen
test runs (558 queries), evaluated once; validation numbers (549 queries) were used for analysis and tuning only.*

## 3.1 Main results on the test queries

{{table:test_main}}

How to read it: every system answers all 558 queries (`n_no_results` is 0), so no query is scored as a missing result. Recall@100 is comparable across systems; MAP is not exactly comparable between P2 (100 results per query) and P3 (1,000), see section 1.3.

Main observations (MRR@10 unless stated):

* **Highest point estimates: the hybrid (RRF of dense `minilm_expanded` and BM25 expanded), MRR@10 0.3990** (not significantly above expanded BM25 on MRR@10, section 3.2), with the highest Recall@10 (0.5986), nDCG@10 (0.4468) and Recall@100 (0.7796). The best purely lexical system is **expanded BM25 (0.3912)**; the best dense system is `minilm_expanded` (0.3015).
* **Representation (lexical):** expanded (abstract plus citing sentences) is the strongest for BM25 (0.3912 vs 0.2695 abstract-only and 0.2624 context-only) and for RM3 and QL; context-only is weak except with query likelihood (0.2914) and
  nearly useless with TF-IDF (0.0485). Expanded TF-IDF (0.2058) is *worse* than abstract-only TF-IDF (0.2469), i.e. expansion does not help every scoring model.
* **Representation (dense):** expanded 0.3015 > abstract 0.2317 > contexts 0.2235; context-only embeddings are no better than abstracts.
* **Lexical vs dense:** with the same representation BM25 beats the MiniLM dense system for the expanded representation (0.3912 vs 0.3015) and is slightly ahead for abstracts (0.2695 vs 0.2317).
* **Validation and test agree on the main ordering:** on the 549 validation queries the hybrid (0.3865) is ahead of expanded BM25 (0.3669), dense expanded (0.2866) and abstract-only BM25 (0.2502), as on test; the selection on validation did not visibly overfit (smaller differences between neighbouring systems can change places).

Validation results (analysis only):

{{table:val_main}}

## 3.2 Significance of the pre-specified comparisons

Paired Wilcoxon signed-rank tests on the same 558 test queries, 95% bootstrap confidence intervals for the mean difference, Holm correction over the 16 comparisons of each metric (list fixed from the research questions, see section 1.4). MRR@10:

{{table:key_comparisons_mrr}}

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

{{table:leak_b_table}}

Systems that use citation contexts (context-only, expanded, hybrid) score **+0.16 to +0.23 MRR@10 higher** on exposed queries, with confidence intervals far from zero, whereas the abstract-only systems, which cannot use another sentence of the citing paper, show
only +0.03 with intervals that include zero. This is strong evidence that part of the advantage of the context-based systems on the full test set is due to the residual same-paper exposure, not only to genuinely better retrieval.

**Exploratory robustness check (defined after the main test results were known, therefore not a pre-specified analysis).** We repeated the 16 comparisons on the 256 test queries that have *no* same-paper, same-target sentence in the index (Holm over those 16):

{{table:key_comparisons_unexposed_mrr}}

On these queries: expanded BM25 is still significantly better than abstract-only BM25, but by **+0.052 instead of +0.122**, so more than half of the full-set gain disappears; **dense expansion no longer helps** (-0.003, not significant) and **context-only systems are significantly worse than abstract-only**
(BM25: -0.114; dense: -0.089). The hybrid still beats dense expanded but is not better than expanded BM25 (+0.005, not significant). The authority prior is significant after correction for all four fused systems on this subset (largest for abstract-only, +0.081; for dense expanded the interval just touches zero, [-0.0001, +0.0151]);
unexposed queries are those the same-paper context cannot rescue, so a popularity signal has more room to matter. The ranking of the systems is therefore more fragile than the headline test table suggests, and conclusions about *citation contexts* should be read with the unexposed numbers in mind.
Rerunning P2 and P3 with a strict index (no sentence of any held-out citing paper) would settle this; it was out of reach for P4.

## 3.4 Popularity bias and cold start

Popularity is the number of citing papers inside the corpus, taken from the leakage-safe evaluation graph (the full-graph count would include each query's own citation). Buckets are those of the handoff and `dataset_stats.md`
(1-2, 3-10, 11-50, >50), plus a bucket **0** for papers with no citation in the evaluation graph. Groups with fewer than 10 queries (>50: 6 queries) are shown but not interpreted.

### Performance by target popularity

MRR@10 on the test queries, by the number of citations of the relevant paper:

{{table:pop_mrr_by_bucket}}

* **Abstract-only systems do not benefit from popularity** (BM25: 0.252 for 1-2 citations, 0.208 for 3-10, 0.226 for 11-50; the trend over popularity is not significant after correction). **Context-using systems do**: context-only BM25 rises from 0.191 (1-2 citations) to 0.374 (3-10) and 0.377 (11-50); expanded BM25 from 0.335 to 0.410 and 0.583; the hybrid from 0.328 to 0.444 and 0.583. A paper that is cited more has more sentences that can match a query.
* **The bucket with 0 evaluation citations looks easy but is not a cold-start test.** These 143 test queries (25.6%) target papers cited only by validation or test papers, so every index sentence about them comes from a held-out citing paper.
  For 134 of the 143 (93.7%) the index contains another sentence of the *same* citing paper that cites the same target (leak B, section 3.3); only 9 queries are free of it:

{{table:exposure_by_bucket}}

  Expanded BM25 scores 0.462 on the exposed queries of this bucket (0.236 on the 9 unexposed ones, too few to interpret), the same level as exposed queries in the other buckets, so the index holds sentences about these papers and they are not "cold" for a context-based system. Exposure is probably not the only difference: abstract-only BM25, which cannot use the citing paper's other sentences, also scores higher
  here (0.368) than for 1-2 citations (0.252), and we did not investigate why. Either way, the benchmark cannot support a clean cold-start estimate for papers without citing context:

{{table:mrr_bucket_exposure}}

### Do the systems retrieve popular papers?

For the top-10 list of each system, the share of retrieved papers that are cited at least once (21.4% of the corpus; with 79% of papers uncited the "most-cited 10%" cut-off collapses to this), the median citation count of a retrieved paper, and how often a wrong top-1 result is more cited than the true target:

{{table:pop_retrieval}}

* **Context-using systems retrieve popular papers**: 91-94% of the top-10 of context-only systems are cited at least once (corpus 21%), 47-70% for expanded BM25 and dense, versus 26% for the abstract-only systems, whose lists reflect the corpus (and can therefore reach uncited papers). Context-only systems can only return papers that have at least one indexed context.
* **The authority prior strengthens this bias.** Fusing with the leakage-safe prior raises the share of top-10 papers that are cited from 26% to 88% (abstract-only BM25), 47% to 70% (expanded BM25), 70% to 83% (dense) and 61% to 81% (hybrid), and the rate at which a wrong top-1 is more cited than the target
  rises by 28 points for abstract-only BM25 (9% to 37%) and by 14 points for the hybrid (52% to 66%). Section 2.3 showed that this buys a significant accuracy gain only for abstract-only BM25; for expanded BM25, dense and hybrid it moves retrieval toward popular papers **without a measurable accuracy gain**.

### Cold start: how many contexts does the target have?

The number of indexed citing sentences of the target paper is the direct cold-start variable for a context-based system. The benchmark keeps only queries whose target has at least one indexed context (16 queries were dropped by P1 for having none), so **targets with zero indexed contexts cannot occur**; the
experiment below compares targets with 1, 2-5, 6-20 and more contexts. Note that these counts include same-paper sentences (section 3.3).

{{table:cold_mrr_by_contexts}}

{{table:cold_trend}}

For the context-only and expanded systems the score rises significantly with the number of indexed contexts (context-only BM25 and dense: Spearman 0.31 and 0.35; expanded and hybrid 0.15-0.18, 0.19-0.20 with the prior; Holm-corrected), whereas abstract-only systems show no trend (-0.06, not significant). A target with a single context is hardest for the context-only systems (BM25 0.174, dense 0.132)
and is where the abstract helps: expanded BM25 (0.343) beats context-only BM25 (0.174) there.

**Implications for cold start.** 76-79% of the corpus has no citation inside it and can only be found through its abstract, and the benchmark never asks for such a paper (every target is cited at least once in the full graph). So the evaluation measures how well *already cited* papers are found; it says nothing direct about
retrieving new papers. What it does show is that the context-based and prior-augmented systems concentrate their results on cited papers (previous table), so they would rank a new, uncited paper behind well-cited ones; only the abstract-only systems keep uncited papers in their top-10 (74% of their lists).
A fair cold-start experiment needs queries about papers with no citing context in the index and a strict index; this corpus is too small and its split design (leak B) prevents it.

## 3.5 Publication year (temporal analysis)

The year analysis is limited by the corpus: 463 of the 558 test queries (83%) target papers from 2020-2022 (110, 246 and 107), 2019 has 44, 2018 has 28, 2017 has 10, and the earlier years have fewer than 10 queries each (shown, not interpreted).
MRR@10 by the publication year of the target:

{{table:year_mrr}}

No system shows a supported trend of the score with the target's publication year (Spearman between 0.07 and -0.04 for all systems, all corrected p-values above 0.95), so **we find no evidence that older or newer targets are retrieved better**. This is a statement about 2017-2022 only; the data cannot say anything about older papers.

{{table:year_trend}}

## 3.6 Citation function (predicted labels)

Each query sentence is joined to P1's `context_functions.parquet` through its `context_id`. **The labels are predictions, not annotations.** They come from P1's text classifier trained on SciCite, a different corpus; its test accuracy on SciCite is about 0.81-0.82 and
**its accuracy on this corpus is unknown** (no sample was hand-labelled). The predicted distribution on the test queries is far from SciCite's training distribution (59% background, 28% method, 13% result): here 239 queries (43%) are labelled *background*, 317 (57%) *method* and only 2 *result*,
which suggests a domain shift, so the labels should be read as "looks like a method citation" rather than as facts. The *result* group (2 queries) is too small to use.

MRR@10 by predicted function:

{{table:function_mrr}}

Systems score higher on sentences predicted as *method* than as *background* (for example expanded BM25 0.435 vs 0.332, the hybrid 0.434 vs 0.351, abstract-only BM25 0.305 vs 0.220), but with two usable groups and Holm correction over the systems the differences are **not statistically supported** (the smallest corrected p is 0.050, for expanded BM25):

{{table:function_tests}}

We therefore report a tendency, not a finding. A per-function comparison of "with vs without a method filter" would need P2's method-filter test runs, which were not delivered.

## 3.7 Error analysis

Failure is a miss at rank 10 (the relevant paper is not in the top 10). Counts out of 558 test queries; "not in the returned list" is the number of queries whose relevant paper is not among the 100 (P2) or 1,000 (P3) results at all:

{{table:error_summary}}

* The hybrid fails least (40.1% miss@10, 224 queries); expanded BM25 47.5%, dense expanded 52.7%, abstract-only BM25 59.1%.
* **BM25 is precise when it finds the paper, dense is not**: the median rank of a found relevant paper is 2-3 for BM25 and 10-26 for the dense systems (the hybrid: 4). Dense systems rarely lose the paper entirely (35-87 queries not in the 1,000 returned) but rank it deep, whereas P2 lists of 100 papers miss it in 155-238 queries,
  which is also why fusing a P2 list can only reorder 100 papers.
* McNemar tests on hits@10 against abstract-only BM25 (Holm-corrected) agree with the paired tests of section 3.2: expanded BM25 (71 queries gained, 6 lost), the hybrid (115 gained, 9 lost) and dense expanded are significantly better; context-only BM25 (92 gained, 107 lost) and dense abstract are not.
* The authority prior moves hits into the top 10 for abstract-only BM25 (228 to 250) and leaves expanded BM25 almost unchanged (293 to 305), dense (264 to 266) and the hybrid (334 to 332), as in section 2.3.

An error analysis by query sentence length or query field and a set of hand-picked failure cases were **not carried out** (not enough time); the per-query results needed for them are in `p4/outputs/real/run_main/analysis/errors_per_query.csv`.

## 3.8 Efficiency (P2 only)

P2 logged index build time, index size and the median and mean query time of every configuration (measured by P2 on its machine on the validation queries; one measurement, no repetitions). P4 reports them as logged:

{{table:efficiency_index}}

Query time and index size of the best-on-validation configuration of each representation and model:

{{table:efficiency_query}}

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

{{table:key_comparisons_recall}}

nDCG@10:

{{table:key_comparisons_ndcg}}

## 3.11 Authority-prior fusion on the test set (all metrics)

{{table:test_fused}}
