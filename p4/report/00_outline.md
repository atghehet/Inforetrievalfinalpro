# Final report: outline and writing plan

**Project:** Finding the Right Paper: Citation-Context Search for Scientific Literature (IR course project, group of 4).
**Format:** Markdown first; converted to the final format later. **Merging owner:** Member 4 (P4), as assigned in the project plan.
**Source of truth:** `Citation_Context_Search_Project_Plan.pdf` (research questions RQ1-RQ4, member responsibilities, deliverables).

## Writing rules

1. **Every number in the report comes from `RECORD.md`**, which says where it came from (file, command, date).
2. **Every result carries a label:** `REAL` (real corpus), `SYNTHETIC` (P1's generated sample) or `DEMO` (hand-made fixtures).
   Only `REAL` results are findings. `SYNTHETIC` and `DEMO` appear only to show that code works, in the methodology or in an appendix, and are never quoted as results.
3. **Nothing is invented.** A section that depends on an experiment that has not run says "not yet run" and names what is missing.
4. Estimates are labelled as estimates and replaced by measured values as soon as they exist.
5. Each member writes the section the plan assigns to them (about 2-3 pages) and reviews one other member's section. P4 drafts the shared sections and merges.

## Structure (14 sections)

| # | Section | Plan owner | Content | Main evidence | Status |
|---|---|---|---|---|---|
| 1 | Problem statement and motivation | P4 (merge) | Vocabulary mismatch; people remember what a paper did, not its title; citing sentences describe papers in functional language (anchor-text analogy) | Plan section 1 | Not started |
| 2 | Research objectives | P4 | RQ1 document expansion vs abstract-only; RQ2 BM25/BM25F vs dense vs hybrid; RQ3 PageRank prior vs popularity bias; RQ4 cold start and citation function | Plan section 1 | Not started |
| 3 | Related work and background | shared | Anchor text and document expansion; BM25/BM25F; dense retrieval (SPECTER, SciBERT); fusion; PageRank; citation-context retrieval; popularity bias | Literature to be collected | Not started |
| 4 | Dataset and data collection | P1 | unarXive 2022 open subset; field restriction; extraction, cleaning, query sampling, splits; statistics; licences; **why the final size is what it is** | `RECORD.md` C, `proposals/dataset_scope_decision.md` (to write), P1's data card | Partial (REAL counts in RECORD) |
| 5 | Crawler design and implementation | P1 (with P4 design) | Existing OpenAlex crawler; coverage gap; enrichment design; sources, limits, terms, deduplication; what was actually run | `proposals/crawler_enrichment_design.md` | Design drafted; nothing crawled |
| 6 | System architecture and methodology | shared | End-to-end pipeline; file formats (TREC runs, qrels, parquet tables); ID conventions; leakage control | Repository, `docs/SCHEMA.md`, P4 README | Not started |
| 7 | Retrieval methods | P2, P3 | Four index variants, TF-IDF/BM25/BM25F/QL/RM3, context windows, citation-function weighting; embeddings, FAISS, RRF/weighted fusion, re-ranking | P2, P3 write-ups; run files | Waiting for P2/P3 |
| 8 | Citation graph and PageRank | P4 | Edge direction; PageRank formula and dangling nodes; damping sweep; **leakage-safe evaluation graph**; authority prior (score combination); co-citation recommender | `p4/graph/`, `proposals/leakage_safe_graph_protocol.md` | Partial (algorithms done; prior experiment not run) |
| 9 | Evaluation metrics and statistical testing | P4 | MRR, Recall@k, nDCG@k, MAP definitions; handling of missing queries; paired bootstrap, Wilcoxon, t-test, Holm; matched-run rule | `p4/eval/`, P4 README | Mostly written in code docs |
| 10 | Experimental setup | shared | Splits, seeds, tuning on validation only, test once, hardware, run inventory | `RECORD.md` D | Not started |
| 11 | Results and analysis | P4 | Main system comparison; expansion effect; graph prior; popularity bias; cold start; year; citation function; errors; efficiency | `RECORD.md` E (REAL only) | **Blocked on real runs** |
| 12 | Limitations and challenges | P4 | Sparse graph; reference resolution; small corpus vs plan target; labels from a classifier; single relevant paper per query; synthetic-vs-real gap; Windows encoding bug | `RECORD.md` G | Collecting |
| 13 | Contributions of each team member | P4 (merge) | Per-member tasks, deliverables, evidence (commits, files, tests) | Repository history, plan section 2 | Not started |
| 14 | Conclusion and future work | P4 | Answers to RQ1-RQ4 from REAL results only; future work (S2ORC / full unarXive, crawler enrichment, cross-encoder) | Section 11 | Not started |

## Appendices (planned)

- A. Data card and licences. B. Reproduction commands (exact, with paths). C. Test inventory and results.
- D. Synthetic and demo outputs (clearly labelled; methodology only).
- E. Failure cases selected for qualitative discussion (10-15, from the error analysis).

## Open items that block sections

- Section 7 and 11: real run files from P2 and P3 on the real corpus.
- Section 8 and 11: the leakage-safe graph protocol must be approved before any graph-based result is reported.
- Section 5: the crawler pilot (approval and contact email pending) decides whether enrichment is part of the final system.
- Section 11 (citation function): P1's classifier output (`context_functions.parquet`) does not exist yet.

> **Update 2026-11-22 (evening):** the P4 sections of the final report are written from REAL results and are in `p4/report/final/` (evaluation methodology, graph-based ranking, results and analysis, conclusion). The table above is the plan for the merged team report; its P4-owned sections (8, 9, 11, 12, 14) are covered by those files. See `RECORD.md` section I.
