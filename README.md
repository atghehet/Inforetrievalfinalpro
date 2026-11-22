# Finding the Right Paper: Citation-Context Search for Scientific Literature

Researchers often remember what a paper *did*, not its title. This project
tests whether the sentences that cite a paper (citation contexts) retrieve it
better than its abstract alone, using lexical (P2), dense and hybrid (P3)
retrieval, evaluated with leakage-controlled queries (P1) and analysed by P4.

## Repository layout

| Path | Owner | What it is |
|---|---|---|
| `citeds/`, `scripts/`, `tests/` | P1 | Data pipeline: unarXive → papers, citation contexts, edges, queries, qrels; leakage control; citation-function classifier; export for P2 |
| `lexical/` | P2 | Lexical retrieval (SQLite FTS5): TF-IDF, BM25, BM25F, query likelihood, RM3; validation tuning |
| `p3/` | P3 | Dense retrieval (Sentence Transformers + FAISS) and hybrid RRF fusion |
| `runs/` | all | Generated TREC run files + `MANIFEST.csv` (git-ignored; shared via the team drive) |
| `docs/` | all | Schema, data card, statistics, P1 guide, P4 handoff |

Data (`data/`), caches, models and runs are never committed. Share them through the team drive.

## Setup (once)

Windows PowerShell (use `source .venv/bin/activate` on Linux/macOS):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r p3/requirements.txt
python -m pytest -q
```

Expect `23 passed`. The first P3 run downloads the embedding models from
Hugging Face (MiniLM about 90 MB, SPECTER about 440 MB).

## Get the data

Unzip the shared `member1-data-v3.zip` into `data/processed/`. To rebuild it
from scratch instead, see `docs/P1_DATA_PIPELINE.md` (needs the unarXive
download and SciCite).

## Run everything

```powershell
python scripts/run_all.py
```

This runs: leakage check → export for P2 → P2 lexical experiment (middle grid)
→ P3 dense retrieval (MiniLM, three representations) → P3 validation metrics
→ hybrid RRF selected on validation → format check of every frozen run →
`runs/MANIFEST.csv`. Options:

| Option | Effect |
|---|---|
| `--p2 quick` | P2 quick grid: one setting per model (minutes) |
| `--p2 full` | P2's full tuning grid: windows 0/1/2, stemming, BM25F weights, all parameters (hours) |
| `--p2 skip` | Reuse existing `runs/p2` |
| `--models sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/allenai-specter` | Also run SPECTER |
| `--build` | Rebuild P1 data from `data/raw` first |

## Rules every member follows

* Index only `index_contexts.parquet` (leakage-controlled), never `contexts.parquet`.
* Run files: `qid Q0 docid rank score tag`, with `query_id` as the qid and `paper_id` as the docid; top 1,000 per query at most.
* Tune on validation (`val`), freeze choices, then run test once.
* Check any run file with `python scripts/check_run.py <run> --split val`.

## Docs

* `docs/SCHEMA.md`: every table and column
* `docs/DATA_CARD.md`, `docs/dataset_stats.md`: the dataset
* `docs/P1_DATA_PIPELINE.md`: building the data
* `lexical/README.md`: P2 details
* `p3/README.md`: P3 details
* `docs/P4_HANDOFF.md`: everything P4 needs
