# Running record (P4)

Single source for every number, decision and file that the final report may use. Newest information is added to the relevant section;
nothing is deleted without a note.

## Labelling rules

- **REAL** = measured on the real corpus (unarXive open subset). Only REAL results are findings.
- **SYNTHETIC** = P1's generated sample (`scripts/make_sample.py`). **DEMO** = hand-made fixtures. Neither is ever quoted as a result.
- **EST** = an estimate; always says how it was made and is replaced when measured.
- Entry IDs: D = decision, M = module/file, C = corpus fact, X = configuration, R = result, T = test run, L = limitation, O = open item.
- Last updated: 2026-11-22. Real data locations are under `C:\Users\kriti\IRdata\` (outside the repository). Nothing generated from real data is stored in the repository.

## A. Decisions

| ID | Decision | Why | Status |
|---|---|---|---|
| D-01 | Dataset scope A: **cs.CL, all years** (approved) | Smallest scope that supports RQ1, RQ2, RQ4 with real data; wider scopes add volume, not density (see `proposals/dataset_scope_decision.md`) | Approved; built |
| D-02 | Build a filtered subset by streaming the tar once with a category pre-filter, instead of unpacking 33.6 GB | P1's reader parses every record before filtering (51 MB/s, about 11 min for the archive) and cannot read a tar | Done (C-2) |
| D-03 | Run P1's build **unchanged**, isolated with `CITEDS_ROOT` and `CITEDS_DATA` under `IRdata\builds\cscl_baseline\` | Nothing generated inside the repository; reproducible | Done (C-3) |
| D-04 | Use `PYTHONUTF8=1` for the build | P1 writes topics/qrels files without an encoding; the first real build crashed on Windows (cp1252) (L-02). Workaround is a runtime setting, no P1 code change | Done |
| D-05 | The failed first build is kept as `builds\cscl_baseline_failed_attempt1_cp1252\` (with its error log), not overwritten | Evidence; rule: do not overwrite | Kept |
| D-06 | Every analysis requires `--data-label real|synthetic|demo`, written into every CSV and report | Prevents synthetic results being mistaken for findings | Implemented |
| D-07 | `papers.citation_count` is the primary popularity measure in analyses; `queries.cited_citation_count` is supplementary | Approved by the team lead | Implemented; **must switch to leakage-safe counts before real results** (D-09) |
| D-08 | Systems are compared only with matched runs (same judged queries, one harness call, all queries answered) | Fair comparison | Implemented |
| D-09 | Leakage-safe evaluation graph and counts (proposed): drop edges of citing papers that have val/test queries | Verified leak on the real build (R-04) | **Proposed, needs approval** |
| D-10 | Index sensitivity analysis: report P1's index (primary) and a strict index without same-citer contexts | 65% of test queries have same-citer contexts in the index (R-04, earlier baseline build; **54.1%** on the integrated data, section I) | **Proposed, needs decision** |
| D-11 | Raw-text reference fallback for arXiv IDs | Doubles resolved references, edges and contexts (R-03) | **Proposal written for P1; P1 code untouched** |
| D-13 | **Team workflow (2026-11-22):** P4 prepares code, tests, protocol and report structure; **no further real experiment or real finding until P1, P2, P3 have synced and the integrated project is verified**. The cs.CL baseline outputs are preserved and are dataset statistics only | Team lead | **Superseded the same evening**: data and runs delivered, real experiments authorised (section I) |
| D-14 | Authority prior is applied as a re-ranking of the retrieved list: per-query score normalisation, corpus-wide prior transform, convex or additive combination; lambda and prior column tuned on validation only, ties to the smaller lambda | Plan task 4 (weighted sum or log-scaled prior, tune on validation) | Implemented (`p4/graph/fusion.py`); DEMO/SYNTHETIC tests only |
| D-15 | Fusion and analyses refuse leaky inputs: the prior's manifest must hold out the evaluated split; data labels must match; real data needs `--confirm-integrated-project` | Leakage protocol | Implemented |
| D-16 | `--name-from-tag` names runs by their sixth column | P2 hides the winning configuration in the file name | Implemented |
| D-12 | Crawler: enrichment of cited papers outside the corpus via the arXiv API, plus the existing freshness crawler kept optional | Coverage lever (C-5) | **Design only; no API call made; pilot needs approval and a contact email** |

## B. Modules and files (P4, all inside `p4/`, none of P1/P2/P3 modified)

| ID | Path | Purpose | Tests |
|---|---|---|---|
| M-01 | `p4/eval/trec_io.py`, `metrics.py`, `harness.py` | TREC run/qrels reading with validation; Recall@K, MRR, nDCG, MAP; multi-system harness; wrong-split protection | `test_trec_io.py`, `test_metrics.py`, `test_harness.py` |
| M-02 | `p4/eval/significance.py` | Paired bootstrap CI, Wilcoxon and t-test, Holm correction, direction-agreement check | `test_significance.py` |
| M-03 | `p4/eval/efficiency.py`, `report.py`, `cli.py`; `scripts/run_eval.py` | P2 efficiency tables; Markdown reports; command-line evaluation | `test_efficiency.py`, `test_report.py`, `test_cli.py`, `test_compat_p2_p3.py` |
| M-04 | `p4/graph/edges.py`, `pagerank.py`, `cocitation.py`, `cli.py` | Edge loading/cleaning, PageRank, co-citation recommender | `test_graph_*.py`, `test_pagerank.py`, `test_cocitation.py` |
| M-05 | `p4/analysis/*` | Popularity, cold-start, year, citation-function and error analyses; synthetic input generator | `test_analysis_*.py`, `test_analyses.py` |
| M-06 | `p4/dataprep/subset_unarxive.py` | Stream the unarXive tar and copy one category into new JSONL files (read-only on the source) | `test_dataprep_subset.py` |
| M-07 | `p4/dataprep/resolution_audit.py` | Compare reference-resolution policies; the raw-text extractor as a pure function | `test_dataprep_resolution.py` |
| M-08 | `p4/graph/leakage_audit.py` | Split audit, leakage-safe edges, popularity-prior diagnostic, same-citer contexts | `test_leakage_audit.py` |
| M-09 | `p4/report/` | Outline, this record, integration checklist, proposals | n/a |
| M-10 | `p4/graph/eval_graph.py` | Leakage-safe evaluation graph: safe edges, `citation_count_eval`, PageRank at damping 0.5/0.85/0.95, optional strict index, manifest | `test_eval_graph.py` |
| M-11 | `p4/graph/fusion.py` | Authority-prior fusion (tune on validation, apply to test), guards | `test_fusion.py` |
| M-12 | `p4/experiments/verify_integration.py` | Read-only integration verifier (schemas, IDs, qrels, runs, leakage gate, P2 reproduction) | `test_verify_integration.py` |
| M-13 | `p4/experiments/run_all.py` | Orchestrator for the whole protocol; refuses real data without confirmation | `test_experiments.py` |
| M-14 | `p4/graph/leakage_audit.py` (extended) | Added `audit_context_file`: gate for the JSONL files P2/P3 index | `test_leakage_audit.py` |

## C. Corpus facts (REAL)

**C-1. Source archive.** `unarXive_230324_open_subset.tar.xz` (4,844,918,096 bytes), tar `unarXive_extracted\unarXive_230324_open_subset.tar` (33,574,574,080 bytes,
5,599 JSONL files `YY/arXiv_src_YYMM_NNN.jsonl`). Unchanged (size and modification time compared before and after all work). Licence per the Zenodo record: CC BY-SA 4.0.
Year skew (by bytes, EST from file names): 2022 = 39.6%, 2021 = 35.5%, 2020 = 9.5%; everything before 2019 is about 10%.

**C-2. cs.CL subset.** Command: `python -m p4.dataprep.subset_unarxive --tar <tar> --out C:\Users\kriti\IRdata\subsets\cscl_all_years --category cs.CL`.
- Records scanned **165,000** (matches the published "165k"); **8,844 cs.CL records kept** (rule: `cs.CL` is a token of `metadata.categories`, the same as P1).
- 0 records without a categories field; 618 records needed the whole-line fallback (categories beyond the first 8 KB); 0 duplicate paper IDs.
- Output: 3,328 JSONL files, 826,937,949 bytes of records; runtime **142.8 s**; manifest `MANIFEST.json` in the folder.
- The earlier EST of 9,600 cs.CL papers (10% stratified sample) was 8% above the measured 8,844 (within the stated sampling uncertainty).

**C-3. Baseline build (P1 code unchanged).** Command (PowerShell/bash equivalent), run from the repository root:
`PYTHONUTF8=1 CITEDS_ROOT=C:/Users/kriti/IRdata/builds/cscl_baseline/root CITEDS_DATA=C:/Users/kriti/IRdata/builds/cscl_baseline/data python -m citeds.build --raw C:/Users/kriti/IRdata/subsets/cscl_all_years --field cs.CL`
(P1 defaults: seed 42, max papers 200,000, max queries 20,000.) Note: the working tree contains my earlier 3-line fix in `citeds/tables.py` for empty tables; it cannot change a normal run.
- Runtime **325.7 s**; outputs in `...\builds\cscl_baseline\data\processed\` total **7.9 MB** (papers 5.5 MB, contexts 1.3 MB, index_contexts 0.77 MB, queries 0.27 MB, edges 30 KB).
- Leakage check inside the build: **passed**.

| Quantity | Value |
|---|---:|
| Papers seen / kept | 8,844 / **8,843** (1 dropped: abstract under 20 tokens) |
| Papers cited at least once | 1,226 (13.9%); 7,617 papers (86.1%) have 0 in-corpus citations |
| Citing sentences seen | 37,736 |
| Contexts dropped: cited paper outside corpus | 42,033 |
| Contexts dropped: self-citation / too short / too long / duplicate | 3 / 27 / 3 / 15 |
| Citation contexts kept | **5,014** (10.65% of all candidate contexts) |
| Contexts in the leakage-controlled index | **3,084** (1,930 removed) |
| Citation edges | **2,720** |
| Queries | **1,918**: train 1,328, val 272, **test 318** (5 queries dropped for lack of index contexts) |
| Distinct relevant papers across queries | 674 |
| Contexts per cited paper (index) | mean 2.52, median 1, p90 3, max 132 |
| Citation count per paper | mean 0.31, median 0, p90 1, max 88 |
| Abstract / context / query length (tokens) | mean 152.7 / 24.6 / 24.3 |

Citation-count buckets (papers; query targets): 0: 7,617; 0 | 1-2: 1,005; 999 | 3-10: 194; 784 | 11-50: 25; 125 | >50: 2; 10.
Publication years of the 8,843 papers: 2006:1, 2007:1, 2011:3, 2012:14, 2013:14, 2014:16, 2015:27, 2016:99, 2017:137, 2018:313, 2019:439, **2020:1,049, 2021:3,082, 2022:3,648** (88.0% from 2020-2022; 625 before 2019).
Query targets by year: 2013:1, 2014:5, 2015:5, 2016:24, 2017:32, 2018:67, 2019:162, 2020:395, 2021:818, 2022:409.

**C-4. Reference resolution audit** (`p4.dataprep.resolution_audit`, 251.8 s; report `IRdata\analysis\resolution_audit_cscl.json`): 8,844 papers, **320,803 references** (36.3 per paper).

| Policy | Resolved | Share | In-corpus edges | Usable contexts | Distinct cited papers | of which outside the corpus |
|---|---:|---:|---:|---:|---:|---:|
| `ids.arxiv_id` only | 17,881 | 5.6% | 1,134 | 2,111 | 5,901 | 5,374 |
| current P1 (`ids` + `contained_arXiv_ids`) | 30,638 | 9.6% | 2,883 | 5,014 | 10,715 | 9,431 |
| proposed (+ raw text) | 61,757 | 19.3% | 6,275 | 10,422 | 17,547 | 15,349 |

Cross-check: contexts under the current policy = **5,014**, identical to P1's real build. Edges differ (2,883 vs 2,720) only because P1 applies further filters.
Raw-text quality: 31,119 references resolved only by raw text; 8,897 agree and 10 disagree with the other sources; title consistency of raw-only hits
that land in the corpus 96.1% (3,403 hits) vs 96.2% for the existing sources (2,897 hits); 81 raw-only IDs newer than the citing paper; 1 reference with several IDs.
Cited papers outside the corpus by ID year (proposed): 2015: 446, 2016: 919, 2017: 1,370, 2018: 2,018, 2019: 2,832, 2020: 3,637, 2021: 2,561, 2022: 981, pre-2007 style: 104.

**C-5. Coverage consequence.** 42,033 contexts are dropped because the cited paper is outside the corpus; 15,349 distinct cited arXiv papers outside the corpus are candidates for metadata enrichment (design only).

## D. Configurations

| ID | Configuration |
|---|---|
| X-01 | Build: seed 42, `MAX_QUERIES` 20,000, `MAX_QUERIES_PER_CITED_PAPER` 5, `MIN_REMAINING_CONTEXTS` 1, near-duplicate Jaccard 0.5, sentence length 6-120 tokens, abstract at least 20 tokens, split 70/15/15 by citing paper |
| X-02 | Evaluation defaults: cutoffs 1, 5, 10, 20, 50, 100; headline 10; bootstrap 10,000 resamples, seed 42, alpha 0.05, Wilcoxon, Holm per metric; minimum group size 10 |
| X-03 | Python 3.11.9, numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, pytest 9.2 (virtualenv `.venv`); Windows 11 |

## E. Results

**Retrieval and graph-fusion results now exist: see section I.** The rows below (R-01 to R-04) are dataset statistics of the earlier cs.CL baseline build, not experiment results; the committed runs in `p3/outputs/` (if still present) are SYNTHETIC.

REAL data audits (describe the data, not retrieval quality):

| ID | Finding | Source |
|---|---|---|
| R-01 | cs.CL subset: 8,844 papers; baseline: 2,720 edges, 5,014 contexts, 1,918 queries (318 test) (C-2, C-3) | build, subset manifest |
| R-02 | Graph sparsity: 86.1% of papers have no in-corpus citation; mean in-degree 0.31 | build stats |
| R-03 | Raw-text fallback doubles resolution, edges and contexts without added false-match signal (C-4) | resolution audit |
| R-04 | Leakage audit: splits are **disjoint by citing paper** (965 citing papers, 0 shared); 100% of val/test queries' own edge is in the full graph; evaluation graph removes 502 of 2,720 edges (18.5%); mean citation count of held-out targets 4.44 (full) vs 2.91 (safe); 34.9% of test targets have 0 citations in the safe graph (0.0% in the full graph); the query-independent "most cited first" prior: median target rank 332 (full) vs 722 (safe), MRR 0.0151 vs 0.0129 (test); **65.1% of test queries (207 of 318) have other index contexts by their own citing paper**; a strict index would leave 143 test targets (45%) with no context | `IRdata\analysis\leakage_audit_cscl_baseline.json`, `proposals/leakage_safe_graph_protocol.md` |

SYNTHETIC/DEMO outputs (code demonstrations only): `p4/outputs/synthetic_inputs/`, `p4/outputs/synthetic_analysis/` (not committed; labelled SYNTHETIC).

## F. Tests

Latest runs (updated after each batch). Test inventory by module is in section B.

| ID | Date | Suite | Result |
|---|---|---|---|
| T-01 | 2026-11-22 | P4 `python -m pytest p4 -q` | **567 passed** in 136.9 s (477 before the preparation batch; +90: eval graph, fusion, orchestrator, verifier, leakage-audit and CLI additions). SYNTHETIC/DEMO data only |
| T-02 | 2026-11-22 | P1 `python -m pytest tests -q` | **25 passed** in 13.4 s |
| T-03 | 2026-11-22 | P2 `python -m unittest discover` (in `lexical/`) | **4 tests OK** |
| T-04 | 2026-11-22 (evening) | P4 `python -m pytest p4 -q` after the real-data stage | **617 passed** in 201 s (597 after A.2, +3 orchestrator options, +12 stage scripts, +5 report builder). P1 `python -m pytest tests -q`: **25 passed**; P2 `python -m unittest discover`: **4 OK**. All P4 tests use toy, DEMO or SYNTHETIC fixtures; the real-data run is checked by `verify_integration` and the `check_run.py` cross-check (section I) |

SYNTHETIC technical checks of the fusion machinery (not findings): on P3's committed synthetic BM25 runs the tuned prior changed validation MRR@10 from 0.2674 to 0.2819
(lambda 0.1), and the leak was not visible on synthetic retrieval scores (the synthetic graph is dense); the exact leak effect is demonstrated only by the controlled DEMO scenario
in `test_fusion.py` (safe prior: MRR@10 0.25 unchanged; leaky prior: 1.0).

## G. Limitations and challenges (collected for report section 12)

| ID | Limitation |
|---|---|
| L-01 | The open subset holds 8,844 cs.CL papers, far below the plan's 50,000-paper target; documented deviation |
| L-02 | P1's build crashes on Windows without UTF-8 mode: `splits.py` lines 86 and 89, `build.py` lines 64 and 71 write files without `encoding`. Not seen on the ASCII-only synthetic sample |
| L-03 | Only 9.6% of references resolve to arXiv IDs with P1's current policy, giving a very sparse graph (86% of papers uncited). *Earlier baseline build; the integrated P1 data, which includes the raw-text fallback, has 76.0% of papers uncited (section I).* |
| L-04 | `edges.parquet` and `papers.citation_count` include the query's own citation (leak A); P1's index keeps same-citer contexts (leak B). Evaluation graph and strict index proposed |
| L-05 | 88% of papers (earlier baseline build) are from 2020-2022; older groups are small. On the real test split 83% of the target papers are from 2020-2022 (463 of 558) |
| L-06 | The existing OpenAlex crawler was never run against the live API, is integrated with nothing, and appears incompatible with current OpenAlex paging rules (per-page 200, cursor without key) |
| L-07 | *Resolved for the integrated data:* P1 delivered `context_functions.parquet` and the function analysis was run (report section 3.6). The labels are **predictions** from a classifier trained on SciCite; their accuracy on this corpus is unknown and the predicted distribution (317 method, 239 background, 2 result among test queries) suggests a domain shift |
| L-08 | Temporal leakage in the graph (citations from papers published after the query's citing paper) is not addressed |

## H. Open items

| ID | Item | Owner / dependency |
|---|---|---|
| O-01 | Approve the leakage-safe protocol and the index setting (D-09, D-10) | Team |
| O-02 | P1: approve and apply the raw-text fallback and the UTF-8 fixes (proposals written) | P1 |
| O-03 | Crawler pilot: approval and contact email | You |
| O-04 | **Delivered** (17 systems on the integrated corpus; SPECTER and P3 timing logs still missing) | P2, P3 |
| O-05 | **Delivered** (`context_functions.parquet`; accuracy on this corpus unknown) | P1 |
| O-06 | **Done on real data (section I).** Still not built: sentence-length and query-field error analysis, curated failure cases, recommender evaluation (bibliographic coupling, hide-references Recall@k), figures | P4 |
| O-08 | **Done** (section I): verifier, leakage gate, check_run cross-check, `run_all` on real data | P4 |
| O-07 | P1's current local code is not yet on GitHub; P4's integration checks must be repeated on the final branches | Team |

## I. Real experiments on the integrated project (2026-11-22, evening)

The team delivered `data/processed` (P1, 8,843 papers, 3,742 queries) and `runs/` (P2 and P3, 17 systems; SPECTER and P3 timing logs were not delivered). The user then instructed P4 to run the real experiments. Decision **D-13 is superseded**;
the baseline build of section C remains valid as dataset statistics only and was not touched. Everything below is REAL and lives in `p4/outputs/real/` (never committed; in the submission archive `P4_Real_Outputs.zip` the same files sit at the archive root, i.e. `config/`, `run_main/`, `key_comparisons/` ... without the `p4/outputs/real/` prefix, and the archive was checked to be byte-identical to that folder); the report sections are in `p4/report/final/` (sources in `p4/report/src/`).

| ID | Decision / step | Evidence |
|---|---|---|
| D-17 | Systems discovered and paired from `runs/MANIFEST.csv` with `p4.experiments.discover_runs` (metadata and tags, not file names): 17 systems, 0 excluded, 0 errors, 0 warnings | `config/discovery_report.json` |
| D-18 | Stage 1 (read-only): `verify_integration` 0 failed (includes the leakage gate on a regenerated P2 input and P4's reproduction of P2's logged validation MRR@10, 47 runs, max difference 5.6e-17); P1's `assert_no_leakage` passes; cross-check against `scripts/check_run.py` on all 17 validation runs, max difference 4.99e-5 (its 4-decimal rounding) | `stage1/` |
| D-19 | Test discipline: P2/P3 configurations frozen by the teammates on validation; P4's fusion tuned on validation only; one test run (`run_all`, 17 steps, all exit 0); the 16 comparisons were fixed before test numbers were inspected | `run_main/EXPERIMENT_MANIFEST.json`, `key_comparisons/` |
| D-20 | Leakage-safe evaluation graph (`leakage-safe-v1`, P4's protocol, applied on the user's instruction to use a defensible graph; **not independently approved by the team**): 1,198 of 5,869 edges removed (20.4%), unweighted PageRank at damping 0.5 / 0.85 / 0.95 | `run_main/eval_graph/MANIFEST.json`, `graph_facts.json` |
| D-21 | Fusion mode `raw` (the handoff formula `score + lambda*log(prior)`) for four systems (abstract BM25, expanded BM25, dense expanded, hybrid); fusing all 17 was not done (cost, 1,000-document lists) | `run_main/fusion/` |
| D-22 | Residual same-paper index leakage (leak B) measured, not removed: 302 of 558 test queries have another sentence of the same paper citing the same target in the index; a clean-subset comparison (256 queries) was added **after** the main test results and is labelled exploratory | `leak_b_diagnostic/`, `key_comparisons_unexposed/` |
| D-23 | Validation-only sensitivity of lambda (grid up to 100): PageRank peaks at lambda 3 and does not overtake the citation count; the frozen test configuration was not changed | `sensitivity_extended_lambda_val/` |

### Key results (REAL, test split, 558 queries; full tables in the report)

* Highest MRR@10: hybrid 0.3990; expanded BM25 0.3912; dense expanded 0.3015; abstract BM25 0.2695. Hybrid vs expanded BM25 not significant on MRR@10 (+0.008); significant for Recall@10 (+0.074).
* RQ1: expanded BM25 vs abstract BM25 +0.122 (significant); dense expanded vs abstract +0.070 (significant); context-only not better than abstract. On the 256 queries without same-paper exposure: +0.052 (BM25), -0.003 (dense).
* RQ3: leakage-safe citation-count prior (chosen over PageRank on validation) helps abstract-only BM25 (+0.043 MRR@10, significant), not expanded BM25 (+0.008, n.s.), dense (-0.002), hybrid (-0.005); it raises the share of already-cited papers in the top 10 (hybrid 61% to 81%).
* Popularity, cold start, year, citation function, errors and P2 efficiency: see report section 3. Function labels are predictions (317 method, 239 background, 2 result among test queries).

### Tests (2026-11-22, evening)

See section F, T-04.

### Not done / limitations added

Strict-index rerun, SPECTER, P3 efficiency, recommender evaluation, error analysis by sentence length and query field, curated failure cases, figures (no plotting package installed), weighted PageRank/HITS, other random splits.
