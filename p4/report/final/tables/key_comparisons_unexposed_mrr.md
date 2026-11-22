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
