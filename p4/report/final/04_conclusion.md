# 4. Conclusion

*P4 section. All statements refer to the real cs.CL corpus (8,843 papers, 558 test queries) and to the delivered runs; "significant" means significant after Holm correction at 0.05.*

## 4.1 Findings

1. **Citation contexts help, but less than the headline numbers suggest (RQ1).** On the full test set, adding the citing sentences to the abstract (the *expanded* representation) improves BM25 by +0.122 MRR@10 (0.2695 to 0.3912) and dense retrieval by +0.070; contexts alone do not beat the abstract.
   However, the index still contains other sentences of the query's own citing paper for 54% of test queries; context-based systems score 0.16-0.23 higher on those queries, abstract-only systems do not. On the 256 queries without that exposure (an exploratory check) the BM25 gain falls to +0.052, the dense gain disappears (-0.003) and context-only
   systems are significantly worse than the abstract. The robust statement is: **expansion helps BM25 retrieval; context-only retrieval does not help; the size of the benefit is overstated on the full test set.**
2. **Lexical, dense and hybrid (RQ2).** The best system is the hybrid (MRR@10 0.3990, Recall@10 0.5986, Recall@100 0.7796). It is significantly better than the dense system it contains, but its MRR@10 advantage over expanded BM25 (0.3912) is **not significant** (+0.008, CI [-0.014, +0.030]); it is significant for Recall@10
   (+0.074) and not for nDCG@10 after correction. Dense MiniLM is significantly worse than BM25 on the expanded representation (-0.090) and not significantly different for abstracts. BM25 is precise when it finds the paper (median rank 2-3); dense systems find more papers but rank them deeper (median 10-26), which is where fusion helps.
3. **A leakage-safe authority prior (RQ3).** Built from a graph without the citing papers of validation and test queries, and tuned on validation, the prior significantly improves abstract-only BM25 (+0.043 MRR@10) and has **no significant effect** on expanded BM25 (+0.008), dense (-0.002) or hybrid (-0.005). The leakage-safe citation count beat PageRank (all three damping values) on validation for every system,
   so we have no evidence that graph propagation adds anything beyond counting citations in this small, shallow graph. The prior also strongly increases the share of retrieved papers that are already cited (for example 61% to 81% for the hybrid), i.e. **popularity bias**, even where accuracy does not improve.
4. **Popularity and cold start.** Systems that use citation contexts do better for more-cited targets and return mostly cited papers (91-94% of the top-10 for context-only systems against 21% of the corpus); abstract-only systems are insensitive to popularity and are the only ones that keep uncited papers in their lists.
   With respect to the number of indexed contexts of the target, context-only and expanded systems improve significantly (Spearman 0.15-0.35) and abstract-only systems do not. The benchmark's zero-citation bucket is not a cold-start test (94% of its queries are exposed to same-paper sentences), and every target has at least one citing context by construction, so
   **the cold-start behaviour for genuinely new papers is not measurable here**; the evidence only indicates that context-based and prior-augmented systems rank uncited papers behind cited ones.
5. **Year and citation function.** No supported effect of the target's publication year (2017-2022) and no statistically supported effect of the predicted citation function (a tendency towards higher scores for sentences predicted as *method*; the labels are predictions with unknown accuracy on this corpus).
6. **Efficiency.** Only P2 is measured: expanded BM25 costs about 24 ms per query and 40 MiB against 21 ms and 37 MiB for abstract-only BM25, RM3 about three times more query time. P3's efficiency is unknown.

## 4.2 Practical reading

For this corpus the evidence supports **expanded BM25 as a cheap, strong baseline** and a **hybrid of dense and BM25 retrieval when recall in the top 10 matters**; an authority prior is only worthwhile for systems that do not use citation contexts and should be accompanied by a check on popularity bias.
Any claim about the added value of *citation contexts* should be conditioned on the residual same-paper leakage described above.

## 4.3 What was not done

* A strict-index rerun of P2 and P3 (requires the teammates' pipelines), SPECTER runs, P3 timing and index sizes, the co-citation / bibliographic-coupling recommender evaluation, error analysis by sentence length and query field, curated failure cases and graphical figures (the tables carry the results; no plotting package was installed).
* A weighted-PageRank, HITS or rank-fusion variant of the authority prior, and any test of how conclusions change with other random splits.
* Independent approval of the leakage-safe graph protocol by the other team members.

## 4.4 Files

The report sections are `p4/report/final/01_evaluation_methodology.md`, `02_graph_based_ranking.md`, `03_results_and_analysis.md` and `04_conclusion.md`; `REPRODUCE.md` lists the commands. All numbers come from `p4/outputs/real/`.
