# P3: Dense and Hybrid Retrieval

Dense retrieval with Sentence Transformers and FAISS for the same three
representations P2 uses, plus hybrid Reciprocal Rank Fusion (RRF) with P2's
BM25 run. All paths are repo-relative (via `citeds.config`), so the scripts
run on any machine. Run them from the repository root.

## Scope of the reported results

The reported dense and hybrid results use **MiniLM only**
(`sentence-transformers/all-MiniLM-L6-v2`). SPECTER is supported but was not
run for the report, because it takes 45–60 minutes on CPU. That's future work:

```powershell
python p3/src/dense_baseline.py --model sentence-transformers/allenai-specter
```

Or use `python scripts/run_all.py --models sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/allenai-specter`.
`hybrid_rrf.py` then picks the best dense run on validation automatically.

The BM25 side of the hybrid comes from P2's **middle grid** (BM25 tuned over 6
settings; see `lexical/README.md`).

## Install

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r p3/requirements.txt
python p3/src/check_setup.py
```

## Run (in order)

```powershell
python p3/src/dense_baseline.py
python p3/src/evaluate_dense.py
python p3/src/hybrid_rrf.py
python p3/src/evaluate_hybrid_test.py
```

Or run the whole project with `python scripts/run_all.py`.

`dense_baseline.py` runs MiniLM by default (about 10 minutes on a 4-core CPU).
`evaluate_dense.py` scores the validation split.
`hybrid_rrf.py` picks the dense system and RRF k on validation and freezes
them. `evaluate_hybrid_test.py` scores test once with those frozen choices.

## Representations

| Variant | Encoded units | Paper score |
|---|---|---|
| `abstract` | one vector per paper: title + abstract (SPECTER: `title [SEP] abstract`) | cosine similarity |
| `contexts` | one vector per leakage-controlled citing sentence | max over its sentences (MaxP); papers never cited cannot be retrieved |
| `expanded` | the abstract vector + its citing-sentence vectors | max over all of them (MaxP) |

MaxP is used instead of concatenating the abstract and contexts into one text,
because concatenation would be truncated at the model's maximum length (256
tokens for MiniLM, 512 for SPECTER) and drop most contexts of highly cited papers.

Search is an exact FAISS inner-product index over normalised vectors. At
8,843 papers, exact search is fast, so no approximate index is needed.

## Outputs (`runs/p3/`, git-ignored)

| File | Content |
|---|---|
| `dense/<model>_<variant>.<split>.run` | Dense runs, top 1,000 per query, for val and test |
| `dense/dense_validation_metrics.csv` | Validation metrics of every dense run |
| `hybrid/selection.json` | Frozen choices: dense system, RRF k, BM25 source |
| `hybrid/hybrid_validation_metrics.csv` | RRF k grid on validation |
| `hybrid/hybrid_rrf.{val,test}.run` | Hybrid runs with the frozen k |
| `hybrid/hybrid_test_metrics.csv` | One-time test check (P4 produces the official numbers) |

Embeddings are cached in `cache/p3/<model>/`. Delete that folder if the data changes.

## Other scripts

* `dense_search.py "query text"`: interactive search for the demo
* `prepare_p2_data.py`: wrapper around `python -m citeds.export_p2`
* `verify_p2_data.py`: checks `data/p2_input`
* `inspect_p1.py`: prints P1's tables

## Changes from the first version (integration)

* Hard-coded `~/Desktop/...` paths replaced with repo-relative paths.
* Synthetic-sample outputs removed. Real runs are regenerated into `runs/p3`.
* RRF k is selected on real validation data and frozen in `selection.json`, instead of `BEST_K = 20`.
* Runs keep the top 1,000 papers instead of all 8,843, so a run file is about 30× smaller.
* Added the contexts and expanded dense variants (research question 1), SPECTER and FAISS.
* The P1 → P2 converter moved to `citeds/export_p2.py`. It writes citation-function labels (not section names) and two neighbour sentences on each side.
