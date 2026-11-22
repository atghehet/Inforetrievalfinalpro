\# P3: Dense and Hybrid Retrieval



\## Overview



P3 implements dense retrieval for scientific-paper search and combines it with P2's BM25 lexical retrieval using Reciprocal Rank Fusion (RRF).



\## Components



\- `src/dense\_baseline.py` — generates dense rankings using Sentence Transformers.

\- `src/dense\_search.py` — dense search functionality.

\- `src/evaluate\_dense.py` — evaluates dense retrieval.

\- `src/prepare\_p2\_data.py` — converts P1 processed data into P2's expected input format.

\- `src/verify\_p2\_data.py` — checks the converted data.

\- `src/hybrid\_rrf.py` — evaluates RRF settings on validation data.

\- `src/evaluate\_hybrid\_test.py` — evaluates the frozen hybrid configuration on held-out test data.

\- `src/check\_setup.py` and `src/inspect\_p1.py` — setup and data inspection utilities.



\## Requirements



Use Python and install the required packages in a virtual environment:



```bash

pip install numpy pandas pyarrow sentence-transformers faiss-cpu scikit-learn

```



P3 also uses P2's lexical retrieval package. Run P2 experiments from the group repository's `lexical` package and configure `PYTHONPATH` as required by P2's instructions.



\## Experimental status



The current experimental outputs were generated using P1's synthetic sample dataset. They are useful for validating the pipeline but are not representative of retrieval performance on a real scientific-paper corpus.



The RRF constant was selected using validation results and frozen before test evaluation. The held-out test metrics are saved in `outputs/hybrid\_test\_metrics.csv`.



\## Reproducibility



The scripts expect the local project layout and processed dataset files described in the P3 workflow. Dataset files, virtual environments, caches, and generated search indexes are intentionally excluded from this commit.



