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
