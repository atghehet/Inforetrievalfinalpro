# Finding the Right Paper
Citation-context search for scientific literature. P2 builds lexical search systems to test whether descriptions written by citing authors help retrieve the paper they cite.

## Ubuntu quick start
Run from the directory containing this README. Use Ubuntu Python 3.10+ with SQLite FTS5. The Porter analyzer is included in FTS5. No pip install, model download or API key is needed.

```bash
sudo apt update
sudo apt install -y python3 git gh unzip
python3 -c "import sqlite3; sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)'); print('FTS5 ready')"
python3 -m unittest discover -v
python3 -m p2.experiment --data data/demo --out outputs/demo --quick
python3 -m p2.search outputs/demo "small adapter layers"
```

The search should return D1. The four fictional papers exercise the code. Use the real corpus for all report findings. Every experiment needs a fresh output directory. Keep the output with its matching input release.

## P1 handoff
Ask P1 for the following UTF-8 files from one fixed, leakage-checked release. IDs should be strings with no spaces. Use the same IDs for P3 and P4. If P1 stores Parquet, request a JSONL export or map the column names to these examples. Each JSONL line is a separate object.

| File under data/real/ | Required data and example |
|---|---|
| papers.jsonl | One row per candidate, including candidates with no contexts. Example: `{"paper_id":"D1","title":"Small modules","abstract":"An encoder uses trainable modules."}` |
| contexts.jsonl | One row per **indexable** context: `{"context_id":"C1","cited_paper_id":"D1","sentence":"They train adapter layers.","before":["Earlier sentence.","Immediately before."],"after":["Immediately after.","Later sentence."],"label":"method"}`. Neighbours are in reading order, up to two on each side, without the central sentence. Give empty arrays at boundaries. Group on the **cited** paper ID. |
| validation.jsonl and test.jsonl | `{"query_id":"v1","text":"How can adapters transfer knowledge?"}`. Use distinct IDs across splits. |
| validation.qrels | Whitespace-separated `v1 0 D1 1`, meaning query, iteration, cited paper ID, relevance. Positive labels are relevant. |
| test.qrels | P4 holds this for final scoring. P2's runner never opens it. |

Ask P1 to remove query sentences and exact/near duplicates from all indexable contexts **and neighbour windows**. P1 owns this check. Verify together on a few query IDs that the target exists and has another indexable context. Keep candidate IDs identical across all variants. Preserve papers with few contexts for P4's cold-start analysis. For the window comparison, P1 must provide neighbours wherever available. The reader accepts omitted arrays as empty, so inspect coverage before interpreting window results. Training queries and citation edges are unnecessary for these unsupervised rankers.

## Run the experiments
1. Run the quick demo above. A quick run includes all models and representations, using one setting each.
2. Put P1's files in data/real/. Start with a coherent pilot subset containing its query targets, their indexable contexts and distractor papers. Check time, free disk and representative query latency before the full grid. SQLite stores postings on disk. Common terms, disk sorts during build and exact language-model scoring can be expensive.
3. Run the main experiment:
   ```bash
   python3 -m p2.experiment --data data/real --out outputs/real-v1
   ```
4. When P1 supplies citation-function labels, merge them into contexts.jsonl by context_id if delivered separately. Check every indexed context has a label. Keep the same splits and rerun with:
   ```bash
   python3 -m p2.experiment --data data/real --out outputs/real-labelled --method-filter
   ```
   This compares all contexts against filtering to method contexts. Report whether filtering helped on validation. An entirely empty context-only representation raises an error, so confirm that some rows are labelled method.
5. For the joint demo choose a system key from best.json:
   ```bash
   python3 -m p2.search outputs/real-v1 "a model that learns from hard negatives" --system expanded_bm25
   ```
   The CLI returns paper IDs and scores. Look up the IDs in papers.jsonl for titles. Coordinate the shared demo with P3.

## What the code implements
| Experiment | Implementation |
|---|---|
| Preprocessing | Lowercase Unicode word tokens, a fixed English stop list in p2/core.py, FTS5 Unicode normalization, optional Porter stemming. Documents and queries share the analyzer. |
| Abstract only | Index the abstract of each candidate paper. |
| Context only | Index cleaned sentences citing the candidate paper. |
| Expanded | Treat abstract and contexts as one bag of words for scoring, summing term frequencies and lengths. |
| Fielded | Keep title, abstract and citation contexts separate for BM25F. |
| Window | 0, 1 or 2 surrounding sentences on each side. The same candidates and queries are used throughout. |
| TF-IDF | Sum query frequency times log-scaled document TF times smoothed IDF. Unnormalized baseline. |
| BM25 | Positive Robertson IDF, length normalization and saturation. Tune k1 in {0.8,1.2,1.6} and b in {0.4,0.75}. |
| BM25F | Normalize TF by each field's relative length, weight title/abstract/context, sum, then saturate once. Compare (1,1,1), (2,1,1) and (2,1,2) with k1=1.2 and b=0.75. |
| Query likelihood | Dirichlet smoothing against collection term frequencies. Compare mu in {300,1000,2000}. Rank the complete collection for in-vocabulary queries, dropping only the document-independent constant. |
| RM3 | Select top 3 or 5 documents by query likelihood (mu=1000), weight their unigram distributions by normalized query likelihood, retain 10 terms, interpolate with the original query (weight 0.5 or 0.8), then run weighted BM25. Compare with plain BM25 on the same representation, window and analyzer; for a controlled feedback comparison use BM25 k1=1.2, b=0.75. |

Lexical matching methods return up to 100 matching documents. Queries with no in-vocabulary terms yield no rows and contribute zero to validation MRR. Score ties use ascending paper ID. The custom index stores per-field postings and document lengths. Phrase search is outside these experiments.

## Selection, outputs and handoff
Tune on **validation MRR@10**. This is a P2 selection criterion, distinct from P4's final metric suite. Pick the highest validation score per representation/model, breaking ties by first configuration in the documented script order. Freeze that selection before running test queries. The single BM25 run for fusion is selected across abstract, context and expanded winners on validation. Full search can generate hundreds of run files.

| Output inside the chosen output directory | Use |
|---|---|
| indexes/*.db | Reusable index per representation/analyzer/window/filter. |
| validation/*.run | A six-column TREC run for every validation configuration: query_id Q0 paper_id rank score run_tag. |
| tuning.csv and best.json | Full parameter grid, validation MRR@10, selected indexes and parameters. |
| test/*.run | Frozen test run for each representation/model family. Give these, validation runs and configuration details to P4. |
| fusion/bm25.validation.run, fusion/bm25.test.run, fusion/selection.txt | Give to P3 for validation tuning and test-time hybrid fusion. |
| efficiency.csv and tuning.csv | Index build seconds and database bytes per index; median and mean search milliseconds per query per configuration. Timings exclude opening the index and writing results. Record machine specs, corpus size and cache conditions when reporting. |

For the report, use P4's final metrics to compare abstract-only, context-only and expanded BM25 on the same queries. Also show a matched-setting comparison using the validation logs to isolate representation changes. Report the effect of stemming, context window, BM25F field weights, RM3 and optional method filtering. Include index size, build time and latency. Select a query where rankings changed, inspect the cited paper's abstract and its indexable contexts, and quote the terms that explain the change. Do not select parameters using test results.

## GitHub from Ubuntu
If a team repository already exists, clone it and copy the P2 files into a feature branch of that repository. For a new team repository, run these commands from the extracted project directory. Start with a README on main and submit your implementation as a reviewable branch.

```bash
gh auth login
gh auth setup-git
git init -b main
git add README.md .gitignore
git commit -m "Set up citation-context search project"
gh repo create citation-context-search --private --source=. --remote=origin --push
git switch -c p2-lexical
git add p2 tests data/demo requirements.txt
git commit -m "Implement lexical indexing, retrieval and validation experiments"
git push -u origin p2-lexical
gh pr create --base main --head p2-lexical --title "Add P2 lexical retrieval pipeline" --body "Includes four index variants, validation tuning, TREC runs and a runnable sample."
```

If git asks for author identity, set your own name and email with git config user.name and git config user.email, then retry the commit. To invite the three teammates, open the repo with `gh repo view --web`, then Settings → Collaborators → Add people. Give P1 access to maintain the README and coordinate repo ownership, as assigned in the team plan. Ask one teammate to review the PR. The ignore file excludes data/real and outputs. Share large runs separately through the team's agreed storage, with the corpus version, git commit and command. Keep raw licensed material out of Git.

## Schedule
Week 1: agree on IDs, fields, neighbours, qrels and the repository. Week 2: build abstract-only retrieval on P1's first delivery. Week 3: compare BM25 on all three text representations. Week 4: compare BM25F and query likelihood; send BM25 runs to P3. Week 5: run RM3 and the label experiment. Week 6: finalize efficiency figures and send final lexical runs to P4. Weeks 7–8: write Indexing and Document Expansion, Lexical Retrieval Models and Efficiency Analysis; prepare your joint demo with P3 and present P2's results.
