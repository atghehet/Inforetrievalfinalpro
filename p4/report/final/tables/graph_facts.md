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
