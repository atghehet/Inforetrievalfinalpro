# Data card: Citation-Context Search corpus (cs.CL)

> Numbers are from build v2 (2026-11-22, URLs stripped). Full statistics: `docs/dataset_stats.md`.

## Summary

A retrieval test collection for finding scientific papers from the way other
papers describe them. Documents are arXiv **cs.CL** papers (title + abstract).
Each paper is optionally expanded with the sentences that cite it (citation
contexts). Queries are held-out citing sentences; the relevant document is
the paper the sentence cites.

| Item | Value |
|---|---|
| Source | unarXive 2022 (Saier, Krause & Färber, 2023), full-text arXiv with resolved references |
| Field | arXiv category `cs.CL` (paper lists it among its categories) |
| Snapshot | unarXive 2022 open subset, `unarXive_230324_open_subset.tar.xz` ([Zenodo 7752615](https://doi.org/10.5281/zenodo.7752615)); papers 2006-2022 |
| Papers | 8,843 cs.CL papers (2,119 cited at least once in-corpus) |
| Citation contexts | 10,376 cleaned; 6,598 in the leakage-controlled index |
| Queries train / val / test | 2,635 / 549 / 558 |
| Seed | 42 (`citeds/config.py`) |
| Build command | `python -m citeds.build --raw data/raw --field cs.CL` |

## Construction

1. **Selection.** Stream unarXive JSONL; keep papers whose arXiv categories
   include `cs.CL`, capped at `MAX_PAPERS` (200,000).
2. **Documents.** Drop papers with a missing title or an abstract under 20
   tokens after cleaning.
3. **Citation contexts.** Split paragraphs into sentences (rule-based,
   abbreviation-aware). A sentence with a `{{cite:..}}` marker whose
   bibliography entry resolves to an arXiv ID becomes one context per cited
   paper. Kept only if the cited paper is itself in the corpus.
4. **Cleaning.** Remove citation, figure, table and formula markers, inline
   math and LaTeX commands, and normalise whitespace. Drop self-citations,
   contexts under 6 or over 120 tokens, and duplicate sentences within a
   citing paper. Counts for each step are in the cleaning log section of
   `dataset_stats.md`.
5. **Queries.** Only sentences citing exactly one corpus paper are eligible,
   so each query has one relevant document. Sampled with seed 42, at most 5 per
   cited paper. The cited paper must keep at least 1 other context. Split
   70/15/15 by **citing paper**, so a paper's sentences never cross splits.
6. **Leakage control.** The index contexts exclude, for every query in every
   split: the query sentence, exact copies (after lower-casing and stripping
   punctuation), and near-duplicates citing the same paper (word 3-gram
   Jaccard ≥ 0.5). Neighbour sentences equal to a query are blanked. Tested
   in `tests/test_leakage.py`; `assert_no_leakage` runs at the end of every build.
7. **Citation functions.** TF-IDF + Naive Bayes trained on SciCite labels
   every context as background, method or result. Logistic regression is the
   comparison baseline. Results are in `docs/citation_function_results.md`.

## Known limitations and biases

* **Small corpus (v1).** The open subset holds 8,843 cs.CL papers, below the plan's
  50,000 target, and 76% of them are never cited inside the corpus. 83,371 citing
  sentences were dropped because the cited paper is outside the corpus. The full
  unarXive set (Zenodo 7752754) is needed to reach the target size.
* **Sparse expansion text.** Median 1 index context per cited paper (mean 3.1).
* **Few "result" labels.** The classifier labels only 134 of 10,376 contexts (1.3%) as
  result, against 13% in SciCite training data: likely domain shift; treat with care.
* **Ground-truth incompleteness.** A query has one relevant paper. Other
  papers that would also satisfy it count as non-relevant, so absolute scores
  are underestimates. Comparisons between systems remain valid.
* **arXiv-only references.** Citations to non-arXiv venues (most ACL
  Anthology papers that were never posted to arXiv) are dropped. The graph
  under-represents older work.
* **Popularity skew.** Citation counts are heavy-tailed. Queries are capped
  per cited paper, but highly cited papers still have far more expansion text.
  This is exactly what RQ3 and RQ4 study.
* **Corpus-internal citation counts.** `citation_count` counts only citing
  papers inside this corpus, not global citations.
* **Sentence splitting** is rule-based and occasionally merges or splits sentences wrongly.
* **Leftover grammar gaps.** Removed markers leave phrases like "as
  described in." This is harmless for retrieval.
* **Citation-function labels** come from a model trained on a different
  corpus (SciCite: biomedical and CS papers), so expect domain shift.

## Licenses and terms

| Data | License / terms | Our use |
|---|---|---|
| unarXive 2022 open subset | CC BY-SA 4.0 (Zenodo record 7752615) | Research use only. Full text is **not** redistributed: the repo holds code, and `data/` is git-ignored |
| arXiv metadata (titles, abstracts) | CC0 for metadata | Stored in `papers.parquet` |
| SciCite (Cohan et al., 2019) | See the allenai/scicite repository license | Classifier training only |
| ACL-ARC (Jurgens et al., 2018) | Research use; check the source before redistribution | Optional alternative training set |
| OpenAlex (crawler) | CC0 | Fresh-paper metadata and abstracts |

## Ethics and responsible use

* Only open datasets and public APIs are used. No personal data beyond
  author names already in public metadata, and we don't use or store those.
* The crawler identifies itself with a contact email, respects robots.txt,
  stays at or below 5 requests/second, and caches every response so nothing
  is fetched twice.
* Copyrighted full text is not republished. Shared outputs are derived
  tables for the course team only.

## Citation

Saier, T., Krause, J., & Färber, M. (2023). unarXive 2022: All arXiv Publications
Pre-Processed for NLP, Including Structured Full-Text and Citation Network. JCDL 2023.

Cohan, A., Ammar, W., van Zuylen, M., & Cady, F. (2019). Structural Scaffolds for
Citation Intent Classification in Scientific Publications. NAACL 2019.
