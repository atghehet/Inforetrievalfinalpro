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
