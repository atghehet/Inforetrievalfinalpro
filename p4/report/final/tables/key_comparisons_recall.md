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
