# P4: Evaluation, statistics and analysis

P4 scores every retrieval system the team builds (P2 lexical, P3 dense and hybrid) on the **same queries**, tests
whether the differences are real, and reports cost. Nothing here builds an index or runs a search: P4 only reads
the **TREC run files** the other members produce and the **qrels** (the answer key) from P1.

> **Status of the numbers.** Every result file currently in the repository (`p3/outputs/`) was produced on P1's
> **synthetic** sample (random text from `scripts/make_sample.py`, 330 papers, 32 test queries). They check that the
> pipeline works. They are **not** findings about real scientific retrieval. Real results need P1's real data.

Implemented so far: evaluation harness, significance testing, CLI runner, P2 efficiency summary, PageRank, co-citation recommender,
the five analyses (popularity bias, cold start, publication year, citation function, error analysis), data preparation and audit tools,
the leakage-safe evaluation graph, authority-prior fusion, an integration verifier and an experiment orchestrator.
Prepared but **not run on real data**: the evaluation graph, fusion and orchestrator (real experiments wait for the integrated project).
Not implemented yet: recommender evaluation (bibliographic coupling, hide-references Recall@k), error analysis by sentence length and query field, figures.

## Architecture

```
run files (P2, P3)  --+
                      +--> trec_io --> harness (metrics.py) --> per-query scores --> significance.py --+
qrels (P1)  ----------+                                                                                 |
P2 efficiency.csv / tuning.csv --> efficiency.py ------------------------------------------------------+--> report.py --> report.md
                                                                                                        +--> CSV files
```

| Module (`p4/eval/`) | Job |
|---|---|
| `trec_io.py` | Read run files and qrels; reject malformed rows, duplicate documents, bad ranks, bad encodings |
| `metrics.py` | Recall@K, MRR, nDCG@K, average precision for ONE query |
| `harness.py` | Score many systems on the same queries; per-query table, summary table, `wide()` layout for paired tests; wrong-split check |
| `significance.py` | Paired bootstrap confidence interval, Wilcoxon / t-test, Holm correction |
| `efficiency.py` | Read and join P2's `efficiency.csv` and `tuning.csv` (no invented numbers) |
| `report.py` | Markdown report |
| `cli.py` | The command-line program (`python -m p4.eval.cli`, or `python scripts/run_eval.py`) |

`scripts/run_eval.py` is a short launcher that makes the repository root importable; it is the same program.

| Module (`p4/graph/`) | Job |
|---|---|
| `edges.py` | Read P1's `edges.parquet` (or csv/tsv) and clean it: drop missing endpoints and self-citations, merge duplicates |
| `pagerank.py` | PageRank over the citation graph |
| `cocitation.py` | Co-citation counts and the recommender |
| `cli.py` | `python -m p4.graph.cli pagerank ...` and `... cocitation ...` |

PageRank and co-citation do not import each other; both only use `edges.py`.

| Module (`p4/analysis/`) | Job |
|---|---|
| `data.py` | Read P1's `queries`, `papers`, `context_functions`, `index_contexts`; join them into one row per judged query |
| `groups.py` | Group definitions (citation buckets, years, function), per-group summaries, paired comparisons inside groups, trend tests |
| `popularity.py` | Retrieval-side popularity bias (how popular are the papers a system returns) |
| `errors.py` | Failures, missed papers, hub papers, differences between systems |
| `analyses.py` | The five analyses and their Markdown reports |
| `cli.py` | `python -m p4.analysis.cli {popularity,coldstart,year,function,errors}` |
| `synthetic_demo.py` | Regenerates P1's **synthetic** sample tables so the analyses can be demonstrated (clearly marked) |

| Module (`p4/dataprep/`, `p4/graph/`, `p4/experiments/`) | Job |
|---|---|
| `dataprep/subset_unarxive.py` | Stream the unarXive tar once and copy one category into new JSONL files (read-only on the source) |
| `dataprep/resolution_audit.py` | Compare reference-resolution policies; the raw-text extractor as a pure function (proposal for P1) |
| `graph/leakage_audit.py` | Split audit, leakage-safe edges, popularity-prior diagnostic, gate for the context files P2/P3 index |
| `graph/eval_graph.py` | Builds the leakage-safe evaluation graph, safe citation counts and PageRank priors (damping 0.5 / 0.85 / 0.95) |
| `graph/fusion.py` | Authority-prior fusion: tune on validation, apply to test, with leakage and label guards |
| `experiments/verify_integration.py` | Read-only verification of an integrated project |
| `experiments/run_all.py` | The whole protocol in order; refuses real data without the integration confirmation |

## Metrics

All metrics use **binary relevance** and take one ranking per query. Our queries have one relevant paper (the paper
the citing sentence cites), but the code also handles several. A query that retrieves nothing relevant scores 0.

| Name (summary) | Per-query column | Definition |
|---|---|---|
| `Recall@K` | `recall@K` | relevant documents in the top K / all relevant documents. With one relevant paper: 1 if it is in the top K, else 0 |
| `MRR@K` | `rr@K` | 1 / rank of the first relevant document if that rank is at most K, else 0 |
| `nDCG@K` | `ndcg@K` | DCG / ideal DCG, where DCG = sum of 1/log2(rank+1) over relevant documents in the top K. With one relevant paper: 1/log2(rank+1) |
| `MAP` | `ap` | sum of precision@r at each relevant document, divided by the number of relevant documents, over the whole run |

Default cutoffs: 1, 5, 10, 20, 50, 100. The headline cutoff in the report is 10 (or the largest cutoff if 10 is absent).

- **Missing queries score 0.** A system is scored on *every* judged query. If it returned nothing for one, that query counts as 0
  (not dropped, which would flatter the system). `n_no_results` in the summary counts these.
- **Wrong split protection.** If a run contains fewer than `--min-coverage` (default 0.5) of the judged queries it is rejected
  with an error, because scoring a validation run against test qrels would produce misleading all-zero results.
  An empty run is allowed (it scores 0). Use `--min-coverage 0` to turn the check off.
- **No duplicate documents.** A ranking must list each document once. `read_run` rejects duplicates and every metric
  raises `ValueError` if given a ranking with duplicates (otherwise AP and nDCG could exceed 1).
- **MAP depends on run depth.** Recall/MRR/nDCG at K look only at the top K, but MAP uses the whole run. The committed runs
  have different depths (P3 dense/hybrid list all 330 papers, most P2 runs 100, `abstract_bm25` and `abstract_tfidf` 43 to 100),
  so compare MAP only between runs of similar depth, or truncate them first.
- The metric code was checked against P2 and P3: it reproduces P3's published metrics and all 13 of P2's validation MRR@10
  values exactly (`p4/tests/test_compat_p2_p3.py`).

## Statistical testing

Question answered: *is the difference between two systems bigger than what the choice of queries could produce by chance?*

For each query take `score(system) - score(baseline)` (all metrics: higher is better). Because both systems answer the same
queries, these paired differences cancel out "this query is just hard".

1. **Paired bootstrap confidence interval.** Resample the queries with replacement (default 10,000 times), take the mean
   difference each time, and report the 2.5th to 97.5th percentiles. If the interval excludes 0, the difference is unlikely
   to be luck. Seeded (`--seed`, default 42), so results are reproducible.
2. **Paired test for a p-value.** Default: **Wilcoxon signed-rank** (no assumption that differences are normally distributed).
   `--test ttest` gives a paired t-test. For Wilcoxon, zero differences are removed first, then the p-value is **exact for up to
   13 non-zero differences** (ties included; checked against a brute-force test) and a normal approximation with tie correction
   above that (within about 0.05 of exact at 14 to 17 tied differences, closer for larger samples).
3. **Holm correction.** Comparing one baseline with many systems makes a small p-value likely by accident.
   `p_holm` corrects for that within each metric. **Prefer `significant_holm` when you compare several systems.**

Rules the code follows:

- Significance is decided **only** from the per-query test, never from averages. Example: a mean gain of 0.9 on 3 queries
  is reported as *not significant*.
- **Direction must agree.** `direction` is the sign of the mean difference, but the Wilcoxon test reacts to ranks. If one huge
  gain makes the mean positive while many small losses make the rank test favour the baseline, the comparison is **never called
  significant**: `direction_agrees` is False and the note says so. (A t-test tests the mean itself, so it cannot disagree.)
- Both systems must cover exactly the same query IDs, else an error. Pairing is by query ID, not by row position.
- Missing scores (NaN) are dropped pairwise and counted (`n_dropped`, reported per comparison); nothing is filled in. The harness
  never produces NaN: a query a run lacks is scored 0 in every comparison. Differences below 1e-12 count as zero.
- Holm correction is applied per metric, over the systems compared in one call (in the CLI: one metric, one baseline). Metrics are
  never mixed. Explicit `pairs=` with different baselines in one `compare_systems` call form a single family.
- Identical systems: p = 1, direction `tie`. One query: no confidence interval (NaN). Fewer than 6 non-zero differences: a
  two-sided Wilcoxon test cannot reach p < 0.05, and the note says so. A paired t-test with zero variance is undefined (NaN).
- With one relevant paper per query, `Recall@K` is a yes/no per query, so its Wilcoxon p-values are only approximate; MRR and
  nDCG are better suited. Multiple metrics are tested separately and are not corrected against each other.

## Run the evaluation

Needs Python 3.10+ with `numpy`, `pandas`, `scipy` (`pip install -r p4/requirements.txt`). Run from the repository root
with the virtual environment active (`.venv\Scripts\Activate.ps1` in PowerShell).

**1. Demo that works today.** Uses P2's committed 2-query demo qrels and three P2 demo runs (in `p4/tests/fixtures/`):

```powershell
python scripts/run_eval.py `
  --qrels lexical/data/demo/validation.qrels `
  --run bm25=p4/tests/fixtures/p2_demo_context_bm25.run `
  --run rm3=p4/tests/fixtures/p2_demo_context_rm3.run `
  --run nothing=p4/tests/fixtures/p2_demo_abstract_bm25_empty.run `
  --out p4/outputs/demo `
  --note "DEMO fixtures (2 queries) - not real results"
```

**2. All committed P2 and P3 test runs.** These runs exist in the repository, but the **qrels for them are not committed**.
They must come from P1 (`data/processed/qrels/qrels.test.txt`, produced by `python -m citeds.build`). Once you have that file:

```powershell
python scripts/run_eval.py `
  --qrels data/processed/qrels/qrels.test.txt `
  --run dense=p3/outputs/dense_test.run `
  --run hybrid=p3/outputs/hybrid_test.run `
  --run bm25=p3/outputs/p2_quick/fusion/bm25.test.run `
  --runs-glob "p3/outputs/p2_quick/test/*.run" `
  --baseline bm25 `
  --efficiency p3/outputs/p2_quick/efficiency.csv `
  --tuning p3/outputs/p2_quick/tuning.csv `
  --out p4/outputs/synthetic_test `
  --note "SYNTHETIC sample data - not real results"
```

Quote glob patterns (PowerShell does not expand them; the program does). Run files for the validation split use
`p3/outputs/dense_val.run`, `p3/outputs/hybrid_validation.run`, `p3/outputs/p2_quick/fusion/bm25.validation.run`
and `p3/outputs/p2_quick/validation/*.run` with `qrels.val.txt`. Never mix splits: score test runs with test qrels.

Options (`--help` lists all):

| Option | Meaning |
|---|---|
| `--qrels FILE` | qrels (required) |
| `--run NAME=PATH` | a run file with a table name; repeat per system. A bare `PATH` uses the file name. The first `--run` is the default baseline |
| `--runs-glob PATTERN` | add every matching run file, named by file name; repeatable |
| `--out DIR` | output folder (required). Existing result files are not overwritten unless `--overwrite` |
| `--cutoffs 1,5,10,...` | metric cutoffs |
| `--baseline NAME` | system the others are compared with |
| `--test-metrics MRR@10,nDCG@10,MAP` | metrics to test (default: MRR, nDCG, Recall at the headline cutoff) |
| `--test wilcoxon\|ttest`, `--n-boot N`, `--seed N`, `--alpha A` | statistics settings (defaults: wilcoxon, 10000, 42, 0.05) |
| `--min-coverage F` | wrong-split protection (default 0.5; 0 disables) |
| `--efficiency FILE`, `--tuning FILE` | P2's logs (see below) |
| `--note TEXT` | text printed at the top of the report; use it to say which dataset the numbers are from |
| `--quiet`, `--debug` | no table on screen / full traceback on errors |

Exit code 0 means success. Bad input (missing file, malformed row, wrong split, ...) prints one line `error: ...` and exits with 2.

## Input formats

- **Run file** (TREC): `qid Q0 docid rank score run_name`, six whitespace-separated fields per line. The `rank` column
  decides the order (not the score). `docid` is the arXiv `paper_id`; `qid` is P1's `query_id`.
- **Qrels**: `qid 0 docid relevance`, four fields. Only `relevance > 0` counts. Every query must have one relevant document.
- **Encoding**: UTF-8 (with or without BOM) and UTF-16 with BOM (what PowerShell's `>` writes) are read; Windows line endings
  work. Other encodings (UTF-32, UTF-16 without BOM, Latin-1) are rejected with a message asking you to re-save as UTF-8.
- **P2 `efficiency.csv`**: `index, build_seconds, bytes`. **P2 `tuning.csv`**: `run, variant, model, parameters,
  validation_mrr_at_10, median_ms, mean_ms`.

## Output files (in `--out`)

| File | Content |
|---|---|
| `summary.csv` | one row per system: `n_queries`, `n_no_results`, then the mean of every metric at every cutoff, and `MAP` |
| `per_query.csv` | one row per (system, query): `n_retrieved` and every per-query metric. This is the raw material for any further test |
| `comparisons.csv` | one row per (metric, system vs baseline): `n_queries`, `n_dropped`, `mean_system`, `mean_baseline`, `mean_diff`, `ci_low`, `ci_high`, `ci_level`, `p_value`, `p_holm`, `test`, `significant`, `significant_holm`, `direction_agrees`, `direction`, `conclusion`, `note` |
| `report.md` | readable summary: inputs, headline table, coverage warnings, comparison tables, efficiency |
| `efficiency_index.csv`, `efficiency_runs.csv` | only with `--efficiency` (and `--tuning` for the second) |

`p4/outputs/` is not git-ignored by the repository `.gitignore`; do not commit result files from synthetic runs as if they were findings.

## Efficiency (P2)

P2's logs give exactly this, and P4 reports exactly this:

- **Index build time and size** per index variant (`efficiency.csv`).
- **Query time** (median and mean ms per query) for every configuration P2 tried, on the *validation* queries (`tuning.csv`),
  joined to the index it ran on, plus the best-validation configuration per (representation, model).

**Not recorded anywhere, so not reported:** memory usage, test-set query times, and any timing for P3's dense and hybrid
systems. Timings come from the machine that ran P2's experiment and are only comparable with each other. Latency of dense and
hybrid retrieval would have to be measured by P3 first. The committed logs are from P2's `--quick` mode (one setting per model,
no stemming, no context windows), so they do not cover the full grid.

## Graph algorithms (PageRank and co-citation)

Both algorithms read the citation graph P1 builds from the cleaned citing sentences (`data/processed/edges.parquet`).
They use only citations **between papers inside the corpus**, so the graph is a cs.CL-only subgraph of arXiv.

**Input schema** (P1's `citeds/schema.py` `EDGES`; checked against a real `edges.parquet` written by P1's build):

| Column | Type | Meaning |
|---|---|---|
| `citing_paper_id` | string | source node: the paper containing the citation |
| `cited_paper_id` | string | target node: the paper being cited |
| `num_contexts` | int32 | number of citing sentences; optional edge weight (used only with `--weighted`) |

An edge points **from the citing paper to the cited paper**. `papers.parquet` (`paper_id`, optional `title`) is optional input: it
adds papers that have no edges and shows titles. Cleaning, applied identically for both algorithms and reported on screen:
rows with a missing endpoint are dropped, **self-citations are dropped**, **duplicate (citing, cited) rows are merged**
(unweighted: one edge; weighted: weights added). P1 already does the last two, but nothing here relies on that.

**Weights are explicit.** They are read only when asked for (`--weighted`, or `weight_col=` in Python); unweighted runs never look at
the column. If weights are requested, every weight must be a positive finite number: zero, negative, missing, NaN, text or
infinite values raise an error that says how many values are invalid and where the first one is. `pagerank(table, weighted=True)` on
a table that was read without weights also raises an error instead of quietly ignoring them. Co-citation counts citing papers, so
weights never affect it.

**Determinism.** Results do not depend on the row order of the input, on Python's hash seed, or on duplicate or self-citation rows
in the unweighted case (the weighted case differs only through the summed weights of duplicate rows). Checked across separate
processes with different hash seeds.

### PageRank

*A paper is important if important papers cite it.* A reader moves between papers: with probability d = 0.85 follow a random
citation of the current paper, with probability 1 - d jump to a random paper. The PageRank is the long-run share of time spent
on each paper.

- **Direction.** Score flows along each edge, from the **citing** paper to the **cited** paper, so a paper gains by being cited,
  most of all by well-ranked papers. A paper that cites many papers splits its score between them.
- **Formula** (N papers, out(j) = number of papers j cites, or their total weight when weighted, D = total score held by papers
  that cite nothing):

      r(i) = (1 - d) / N  +  d * ( sum over papers j that cite i of  r(j) / out(j)   +   D / N )

- **Dangling papers** (cite nothing in the corpus; very common, because old papers cite things outside the corpus) spread their
  score evenly over all papers, so no score is lost. **Isolated papers** (no edges) are dangling papers nobody cites: they keep the
  (1 - d)/N teleport share plus an equal share of the dangling score, never zero. No papers at all gives an empty result;
  papers but no edges gives 1/N each.
- **Computation.** Power iteration from the uniform vector until the total absolute change is below `--tol` (default 1e-10) or
  `--max-iter` (default 100) iterations; the result says whether it converged. Scores are normalised to **sum to 1**;
  `score_scaled` divides by the maximum (best paper = 1.0), which is convenient for combining with other signals later.
- **Ranking.** Highest score first; scores within 1e-12 are ties and are ordered by `paper_id`, so the order is identical on every
  machine and for any input row order. `--top-k` limits the printed or saved ranking.
- **Cost.** Time O(iterations * (N + E)), memory O(N + E), one sparse matrix-vector product per iteration.

**Output** (`ranking` table): `paper_id`, `score` (sums to 1 over all papers), `score_scaled` (max = 1), `rank` (1, 2, 3, ...,
consecutive; ties broken as above), plus `title` on the command line when `--papers` has titles.

### Co-citation recommender

*Two papers are co-cited when one citing paper cites both.* The co-citation count of a pair is the number of distinct citing papers
that cite both. For a given paper the recommendations are the other papers with the highest count.

- **Formula.** With A the 0/1 matrix (citing paper x cited paper), `Co = A^T A`, so Co[x, y] counts the citing papers that cite both
  x and y. The diagonal (a paper's own citation count) is set to 0: a paper is never co-cited with itself and never recommended
  to itself.
- **Ordering.** Count descending, then `paper_id` ascending (deterministic). `--top-k` limits the list.
- **Missing and empty cases.** A paper the edges never mention (and the papers file does not list) raises an error (exit code 2 on
  the command line). A paper that is known but has no co-cited neighbours returns an empty table. An empty graph knows no papers.
- **Cost.** Building: O(sum over citing papers of r^2) time, r = number of references of that paper (a paper citing r papers
  creates r(r-1)/2 pairs), memory O(number of co-cited pairs). `--max-references N` skips citing papers with more than N
  references to bound the cost. Each recommendation: O(k log k) for a paper with k co-cited neighbours.

**Output**: `paper_id`, `cocitation_count`, `rank` (plus `title` on the command line).

### Run them

From the repository root. These commands use a tiny **synthetic demo graph** (`p4/tests/fixtures/demo_edges.csv`: 10 hand-made
edges between 7 made-up papers; `demo_papers.csv` adds an 8th, isolated one). It only demonstrates the tools; it is not data.

```powershell
python -m p4.graph.cli pagerank --edges p4/tests/fixtures/demo_edges.csv --papers p4/tests/fixtures/demo_papers.csv --top-k 5
python -m p4.graph.cli cocitation --edges p4/tests/fixtures/demo_edges.csv --papers p4/tests/fixtures/demo_papers.csv --paper A
```

On real data (after P1's build) use P1's files and add `--out` to save a CSV:

```powershell
python -m p4.graph.cli pagerank --edges data/processed/edges.parquet --papers data/processed/papers.parquet --top-k 50 --out p4/outputs/pagerank_top50.csv
python -m p4.graph.cli cocitation --edges data/processed/edges.parquet --papers data/processed/papers.parquet --paper 2106.09685 --top-k 10
```

Other options: `--damping`, `--tol`, `--max-iter`, `--weighted` (PageRank); `--max-references` (co-citation); `--overwrite`, `--debug`.
Bad input exits with code 2 and one `error:` line. From Python:

```python
from p4.graph.edges import read_edges, read_paper_ids
from p4.graph.pagerank import pagerank
from p4.graph.cocitation import CoCitationIndex

edges = read_edges("data/processed/edges.parquet")                    # cleaned EdgeTable
papers = read_paper_ids("data/processed/papers.parquet")
result = pagerank(edges, nodes=papers["paper_id"], top_k=20)           # result.ranking, result.scores, result.converged
index = CoCitationIndex.from_edges(edges, nodes=papers["paper_id"])
index.recommend("2106.09685", top_k=10)                                 # DataFrame: paper_id, cocitation_count, rank
```

### Graph limitations

- **No real graph results exist yet.** The only real `edges.parquet` available so far (P1's 49-paper sample build) has 0 rows,
  because the sample contains no citations between its own papers. Everything verified so far used hand-made graphs, an independent
  exact linear-system solution, and P1's synthetic sample. Real numbers need P1's real build.
- The graph holds only citations that resolve to arXiv IDs of papers in the corpus (P1's data card lists the losses), so it is
  sparse and under-represents older work. Many papers will be dangling.
- **PageRank favours old, much-cited papers**: citations accumulate over time and there is no correction for publication year.
  New papers have almost no score (the cold-start problem this project studies).
- **Co-citation is popularity-biased**: a heavily cited paper is co-cited with many papers. Counts are raw; there is no
  normalisation (such as Jaccard or cosine). Recommendations use citation structure only, not text.
- The weighted variant treats `num_contexts` (number of citing sentences) as the strength of a citation: an assumption, not a fact.
- PageRank is a stand-alone importance score here. Combining it with retrieval scores is a later step.

## Analyses: popularity, cold start, year, citation function, errors

These analyses ask *where* retrieval works or fails. They describe **associations in the data**; none of them shows that popularity,
age or citation function *causes* a difference.

### Rules that apply to every analysis

- **`--data-label real|synthetic|demo` is required.** It is written into the top of every report and into a `data_label` column of
  every CSV. It is your declaration; the tool cannot verify it. Synthetic and demo outputs are demonstrations, never findings.
- **Reports record their provenance:** input file paths, dataset statistics (queries, papers, how many queries lack a year, a
  citation count, a function label, or a row in the queries file; mismatches between files), the analysis date (UTC; override with
  `--analysis-date` for byte-identical reruns), and all settings.
- **Nothing is substituted for missing data.** A missing input is an error (for example `function` without
  `context_functions.parquet`), and a missing value becomes an explicit `unknown` group with its count.
- **Matched runs.** Systems are compared only if they are matched: the same judged queries, evaluated together by one harness call
  with the same cutoffs, and every system returned results for every judged query. Otherwise comparisons are skipped and the report
  says so. The tool cannot check that two systems differ *only* in the document representation. Example: P2 tunes BM25 separately
  for abstract-only and expanded indexes (`best.json`), so a comparison of the frozen test runs mixes representation and tuning; a
  strictly matched comparison needs P2's validation logs.
- **Small groups.** A group with fewer than `--min-group-size` (default 10) queries is shown with its size and mean but has no
  confidence interval, no system comparison, and is left out of trend and group-difference tests. Empty groups appear with `n = 0`.
- **Statistics.** Intervals are bootstrap intervals over queries. System differences use the paired tests of the evaluation package
  (Wilcoxon, direction check, Holm correction per metric). Trends are Spearman correlations (with a bootstrap interval) over the
  queries in non-small groups, and a trend is "supported" only if at least two non-small groups remain. Unordered groups
  (citation function) use a Kruskal-Wallis test.

### Popularity measures

`papers.citation_count` is the **primary** measure everywhere: the number of citing papers inside the corpus, defined for every
paper. `queries.cited_citation_count` is the same number copied onto the query row by P1; it is kept as supplementary metadata and
cross-checked against `papers.citation_count` (the report counts disagreements). Neither is subtracted from or adjusted. Both
**include the query's own citing paper**, so a paper that is the relevant paper of a query has `citation_count >= 1`.
**Zero-citation papers cannot be evaluated as query targets** (a paper nobody cites cannot be the target of a citing sentence); they
can only be studied as distractors in rankings (the popularity analysis reports how often they are returned).

### The five analyses

| Command | Question | Groups / measures | Inputs |
|---|---|---|---|
| `popularity` | Do systems favour highly cited papers? | Query side: score by citation bucket (0, 1-2, 3-10, 11-50, >50) and a Spearman trend. Retrieval side: popularity of the top-k results (mean log10 citations, share from the most-cited 10% against that group's share of the corpus, share of zero-citation papers) and how often a wrong top-1 paper is more or less cited than the relevant one | runs, qrels, `papers` (needs `citation_count`), `queries` |
| `coldstart` | Is performance worse for papers with few citations or little expansion text? | Citation buckets as above; optionally the number of **indexable contexts** of the relevant paper (the expansion text actually available, from `index_contexts`); system differences inside each bucket | runs, qrels, `papers`, `queries`, optional `index_contexts` |
| `year` | Does performance depend on the relevant paper's publication year? | One group per year (default) or `--year-bins 2019,2021` (gives <=2019, 2020-2021, >=2022); year-by-popularity table to show how entangled the two are; supplementary citation-age groups (citing year minus cited year) | runs, qrels, `papers` (needs `year`), `queries` |
| `function` | Does performance depend on how the query sentence cites the paper? | background / method / result / unlabelled (`--min-confidence` sets weak labels aside); Kruskal-Wallis across groups | runs, qrels, `papers`, `queries`, **`context_functions`** |
| `errors` | Where do systems fail, which papers are missed, how do systems differ? | Per-query rank of the relevant paper, failure = not in the top k (`--k`, default 10), categories (rank 1, 2-k, deeper, not retrieved), differences between systems (exact binomial test on queries where only one hits), hardest queries, missed relevant papers, hub papers (wrong papers often at rank 1), failure rate by group | runs, qrels; `queries` / `papers` / `context_functions` add texts, titles and groups |

Citation-function labels describe the **query sentence** (joined through `queries.context_id`). They are predictions of P1's classifier,
trained on SciCite (a different corpus), not human annotations; their accuracy on this corpus is unknown.

### Outputs

For each command, `--out DIR` receives `<analysis>_report.md` and CSV tables (every CSV starts with `data_label`). Examples:
`popularity_query_side_groups.csv` (long format: system, group, metric, `n_queries`, `small_group`, mean, interval),
`..._comparisons.csv` (paired differences with `p_holm`, `significant_holm`), `..._trend.csv` (Spearman rho, interval, `n_queries_used`,
`n_queries_excluded`, `supported`), `popularity_retrieval.csv`, `errors_summary.csv`, `errors_per_query.csv`, `errors_pairwise.csv`,
`errors_hard_queries.csv`, `errors_hubs.csv`, `errors_missed_papers.csv`, `errors_failure_by_group.csv`.

### Reproduce the demonstration (synthetic data only)

No real queries exist yet (P1's real sample build has 0 queries and no `context_functions.parquet`), so the commands below use P1's
**synthetic** sample and P3's committed runs. The outputs are demonstrations, kept in `p4/outputs/` (not committed), and say nothing
about real retrieval. The `function` analysis cannot be demonstrated on real-pipeline data: its labels would have to be invented,
so it is covered only by unit tests with hand-made labels.

```powershell
python -m p4.analysis.synthetic_demo --out p4/outputs/synthetic_inputs
python -m p4.analysis.cli popularity --data-label synthetic `
  --qrels p4/outputs/synthetic_inputs/qrels.test.txt `
  --run abstract_bm25=p3/outputs/p2_quick/test/abstract_bm25.run --run expanded_bm25=p3/outputs/p2_quick/test/expanded_bm25.run `
  --run dense=p3/outputs/dense_test.run --run hybrid=p3/outputs/hybrid_test.run `
  --queries p4/outputs/synthetic_inputs/queries.parquet --papers p4/outputs/synthetic_inputs/papers.parquet `
  --index-contexts p4/outputs/synthetic_inputs/index_contexts.parquet --baseline abstract_bm25 `
  --note "SYNTHETIC sample - demonstration only" --out p4/outputs/synthetic_analysis
```

Replace `popularity` with `coldstart`, `year` or `errors` for the others. On real data, point the same options at P1's
`data/processed/` files, P2's and P3's real run files, and use `--data-label real`:

```powershell
python -m p4.analysis.cli coldstart --data-label real --qrels data/processed/qrels/qrels.test.txt `
  --run abstract_bm25=<P2 test/abstract_bm25.run> --run expanded_bm25=<P2 test/expanded_bm25.run> `
  --queries data/processed/queries.parquet --papers data/processed/papers.parquet `
  --index-contexts data/processed/index_contexts.parquet --baseline abstract_bm25 --out p4/outputs/real_coldstart
python -m p4.analysis.cli function --data-label real ... --functions data/processed/context_functions.parquet
```

### Analysis limitations

- **No real results yet.** Everything above was checked on hand-made data and on P1's synthetic sample.
- **Confounding.** A popular paper also tends to be older and to have more citing sentences to index. Group differences cannot be
  assigned to one of these. The year-by-popularity table only shows how much they overlap.
- **Small samples.** With few queries most groups are small (the synthetic test split has 32 queries, so only one popularity group
  reaches 10). Intervals are wide and "not significant" is not evidence of no difference.
- **Popularity is corpus-internal** (cs.CL papers citing cs.CL papers that resolve to arXiv IDs), not global citation counts.
- **Relevant papers are not a random sample**: each has at least one citation, so corpus shares in the retrieval-side analysis are
  reference points, not a neutral expectation.
- **One relevant paper per query.** Other papers that would also answer the query count as failures, so failure rates overstate how
  often a system returns nothing useful.
- **Run depth differs** (P2 up to 100 results, P3 dense/hybrid every paper), so "not retrieved" and deep ranks are not equally
  informative across systems.
- **Function labels are machine-predicted** and unvalidated on this corpus.
- **Multiple comparisons:** Holm correction is applied per metric inside one analysis, not across analyses.

## Leakage-safe evaluation, authority-prior fusion and the experiment orchestrator

> **Status: prepared, tested on SYNTHETIC and DEMO data only.** The protocol below is **proposed and pending team approval**
> (`report/proposals/leakage_safe_graph_protocol.md`). Nothing here has been run on real data, and no real fusion or graph result exists.
> Real experiments wait until P1, P2 and P3 have synced their final files and the integrated project is verified.

**Why.** `edges.parquet` and `papers.citation_count` contain each query's own citation, so a PageRank or citation-count prior computed
from them rewards the paper the query asks for. P1 splits per *citing paper*, so the clean rule is: **drop every edge whose citing paper has
a validation or test query**, and compute every graph signal from what is left (the *evaluation graph*).

| Tool | Command | What it does |
|---|---|---|
| Leakage audit and gate | `python -m p4.graph.leakage_audit --processed DIR --out report.json [--context-file P2_contexts.jsonl]` | Splits disjoint by citing paper? How much does the leak change a popularity prior? Same-citing-paper index contexts? **Gate for the files P2/P3 index:** no row may be a query context, a query sentence, or absent from the leakage-controlled index |
| Evaluation graph | `python -m p4.graph.eval_graph --processed DIR --out NEW_DIR --data-label L [--strict-index] [--dampings 0.5,0.85,0.95]` | Writes `priors.parquet` (`citation_count_eval`, `pagerank_d050/_d085/_d095`), `edges_eval.parquet`, optionally the strict index, and `MANIFEST.json` (protocol, hold-out splits, edge counts, PageRank convergence, input hashes, data label) |
| Fusion: tune | `python -m p4.graph.fusion tune --data-label L --run VAL.run --qrels qrels.val.txt --prior-dir NEW_DIR --out TUNE_DIR` | Grid over prior column x lambda on the **validation** queries only; `fusion_tuning.csv`, `fusion_best.json` (says whether the prior improved on the no-prior baseline) |
| Fusion: apply | `python -m p4.graph.fusion apply --data-label L --run TEST.run --prior-dir NEW_DIR --best TUNE_DIR/fusion_best.json --out FUSED.run` | Re-ranks a run with the frozen choice; the result is evaluated by the normal tool (`scripts/run_eval.py`) next to the base run |
| Analyses with safe counts | `python -m p4.analysis.cli popularity ... --citation-counts NEW_DIR/priors.parquet` | Uses `citation_count_eval` instead of `papers.citation_count`; the report states which one was used |
| Verify an integrated project | `python -m p4.experiments.verify_integration --processed DIR --config runs.json --out verify.json` | **Run this first** when the teammates' final files arrive: P1 schemas, ID consistency, qrels versus queries, run files (documents known, queries answered), the leakage gate, and whether P4's harness reproduces P2's logged validation MRR@10 (`p2_check`). Exit 2 on any failure |
| Everything in order | `python -m p4.experiments.run_all --data-label L --processed DIR --config runs.json --out NEW_DIR` | Leakage gate, evaluation graph, evaluation, fusion tune/apply, analyses; **refuses `--data-label real` unless `--confirm-integrated-project` is given** |

**Naming systems from run files.** P2 names frozen test runs like `context_bm25.run`, which hides which configuration won. Both evaluation CLIs accept
`--name-from-tag` (for `--runs-glob` matches and bare `--run PATH`): the system is named by the run tag in the file's sixth column
(for example `context-w0-plain-method-bm25-0`). Duplicate tags are an error.

**Fusion method** (research question 3). Per query, normalise the retrieval scores of the retrieved list (`minmax`, `zscore` or `rank`);
transform the prior corpus-wide (`identity` or `log`) and rescale to [0, 1]; combine with weight lambda:
`convex: s = (1 - lambda) * score + lambda * prior` (lambda in [0, 1]) or `additive: s = score + lambda * prior`. lambda = 0 reproduces the
original ranking; ties break by original rank, then document id. Lambda and the prior column are chosen on validation (ties go to the smaller
lambda, so "no prior" wins unless the prior *strictly* improves the validation metric), then frozen for the test split.
Only documents a system already retrieved can be re-ranked.

**Safety guards.** The prior's manifest must list the evaluated split among the held-out splits, otherwise the tool refuses (a flag exists only
to *demonstrate* the leak on synthetic data and stamps the output). The data label on the command line must equal the label in the manifest.
Existing output folders and files are never overwritten. Tuning functions take validation data only.

**Demonstration (SYNTHETIC; shows that the machinery works, not any finding).** The unit tests contain a controlled DEMO scenario in which every
target is cited only by its own query's citing paper: with the leakage-safe prior the fused ranking is exactly the base ranking (MRR@10 0.25), and a
prior built from a graph that still contains the queries' edges lifts every target to rank 1 (MRR@10 1.0). On P1's synthetic sample the effect of the
leak on fusion is small and not robust (the synthetic graph is dense), so no claim is made from it.

```powershell
python -m p4.analysis.synthetic_demo --out p4/outputs/synthetic_processed --layout processed
python -m p4.graph.eval_graph --processed p4/outputs/synthetic_processed --out p4/outputs/synthetic_eval_graph --data-label synthetic
python -m p4.graph.fusion tune --data-label synthetic --run p3/outputs/p2_quick/fusion/bm25.validation.run `
  --qrels p4/outputs/synthetic_processed/qrels/qrels.val.txt --prior-dir p4/outputs/synthetic_eval_graph --out p4/outputs/synthetic_fusion
python -m p4.graph.fusion apply --data-label synthetic --run p3/outputs/p2_quick/fusion/bm25.test.run `
  --prior-dir p4/outputs/synthetic_eval_graph --best p4/outputs/synthetic_fusion/fusion_best.json --out p4/outputs/synthetic_fusion/fused_test.run
```

**Orchestrator and verifier configuration** (`runs.json`; relative paths are resolved against the file's folder; the verifier also reads an optional `p2_check` block `{"tuning_csv", "validation_dir", "qrels"}`):

```json
{"systems": {"bm25": {"val": "bm25.validation.run", "test": "bm25.test.run"},
             "dense": {"val": "dense_val.run", "test": "dense_test.run"}},
 "baseline": "bm25",
 "context_files": ["p2_input/contexts.jsonl"],
 "efficiency": {"efficiency_csv": "efficiency.csv", "tuning_csv": "tuning.csv"},
 "functions": "context_functions.parquet", "index_contexts": "index_contexts.parquet"}
```

**Limitations.** Protocol pending approval. Temporal leakage (citations from papers published after the query's citing paper) is not addressed.
The strict-index sensitivity analysis needs P2/P3 to rebuild their indexes from `index_contexts_strict.parquet`. The evaluation graph is smaller
(on the real baseline it keeps 2,218 of 2,720 edges), so prior effects will be estimated on a very sparse graph.

## Tests

```powershell
python -m pytest p4 -q                 # P4 (evaluation, statistics, graph)
python -m pytest tests -q              # P1 (25 tests)
cd lexical; python -m unittest discover; cd ..    # P2 (4 tests)
```

`p4/tests/test_compat_p2_p3.py` checks that P4 reproduces numbers P2 and P3 published. It regenerates the synthetic qrels
with P1's deterministic code (read-only) and skips itself if the committed P2/P3 outputs are missing.
A tiny sample with tied differences makes scipy's Wilcoxon slower (about half a second per test); real query sets do not.

## Assumptions and limitations

- Relevance is binary and each query has one relevant paper (P1's design), so absolute scores are underestimates when other
  papers would also satisfy the query; comparisons between systems remain valid.
- Statistics assume queries are independent samples. Queries from the same citing paper are split together by P1, but may
  still be correlated.
- `--min-coverage` only catches gross split mix-ups. A run that overlaps the qrels by 50% or more passes (and is reported in
  `warnings` / the report's coverage section).
- Synthetic-data numbers say nothing about real retrieval quality.

## Real-data experiments and final report (2026-11-22)

The integrated project (P1 data, P2 and P3 runs) was delivered and the real experiments were run once; everything is in `p4/outputs/real/` (REAL data, not committed) and the report sections are in
`p4/report/final/` (`01_evaluation_methodology.md`, `02_graph_based_ranking.md`, `03_results_and_analysis.md`, `04_conclusion.md`, `REPRODUCE.md`). The exact commands are in `p4/report/final/REPRODUCE.md`.

| Module | Purpose |
|---|---|
| `p4/experiments/discover_runs.py` | Builds `runs.json` from `runs/MANIFEST.csv`, pairing validation and test runs from metadata and file contents; reports missing, duplicate, unexpected and mismatched runs |
| `p4/experiments/eval_config.py` | Evaluates every system of a `runs.json` on one split |
| `p4/experiments/crosscheck_check_run.py` | Compares P4's metrics with the team's `scripts/check_run.py` |
| `p4/experiments/key_comparisons.py` | The pre-specified paired comparisons with bootstrap CIs and Holm correction (also on a subset of queries) |
| `p4/experiments/leak_b_diagnostic.py` | Measures how many queries have other sentences of their own citing paper in the index, and how scores move with it |
| `p4/experiments/graph_facts.py` | Graph direction check, size of the leakage-safe graph, prior versus popularity |
| `p4/experiments/make_report.py`, `analysis_tables.py` | Fill the report sources with tables generated from the output files |

`run_all` gained `fusion_systems` / `analysis_systems` in the configuration and `--fusion-mode` / `--fusion-lambdas`; `p4.graph.fusion` gained the `raw` mode (`score + lambda*log(prior)`, the handoff formula).
