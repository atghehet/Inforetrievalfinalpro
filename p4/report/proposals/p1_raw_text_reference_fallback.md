# Proposal for P1: raw bibliography-text fallback for arXiv-ID resolution

**Status:** proposal for team approval. **No P1 code has been modified.** Evidence below is REAL, measured on the cs.CL subset of
the unarXive open subset (8,844 papers, 320,803 references); commands and file locations are in `RECORD.md`.

## 1. The limitation today

`citeds/unarxive.py`, function `_bib_to_arxiv` (lines 83-94), resolves a bibliography entry to an arXiv ID from two places only:

1. `bib_entries[].ids.arxiv_id`
2. `bib_entries[].contained_arXiv_ids` (unarXive builds this list only from **hyperlinks** in the LaTeX source; plain-text
   mentions such as "arXiv preprint arXiv:2104.06921" are not extracted)

The reference text itself, `bib_entries[].bib_entry_raw`, is ignored even when it contains the ID in plain text. Result on the real subset:

| Policy | References resolved | Share | In-corpus edges | Usable contexts | Distinct cited papers outside the corpus |
|---|---:|---:|---:|---:|---:|
| `ids` only | 17,881 | 5.6% | 1,134 | 2,111 | 5,374 |
| **current P1 policy** (`ids` + `contained`) | 30,638 | **9.6%** | 2,883 | **5,014** | 9,431 |
| **proposed** (+ raw text) | 61,757 | **19.3%** | 6,275 | **10,422** | 15,349 |

My audit reproduces P1's real baseline build **exactly** on contexts under the current policy (5,014 = 5,014). Its edge count (2,883)
is slightly above P1's (2,720) because P1's later cleaning filters remove some edges; the audit does not apply those filters.
So the raw-text fallback roughly **doubles** the resolved references, in-corpus edges and usable contexts. It is the cheapest large
gain available: no download, no crawling, about 15 lines of code.

## 2. How the fallback would work

For each bibliography entry that neither source resolved:

1. Search `bib_entry_raw` with one regular expression (reference implementation and tests: `p4/dataprep/resolution_audit.py`,
   function `extract_raw_ids`, and `p4/tests/test_dataprep_resolution.py`). It accepts:
   - `arXiv:2104.06921`, `arXiv preprint arXiv:2104.06921v3`, `ARXIV: 1810.04805`
   - URLs: `arxiv.org/abs/2010.11929`, `arxiv.org/pdf/hep-th/9901001`
   - the arXiv DOI form `10.48550/arXiv.2208.10923`
   - old-style IDs `hep-th/9901001`, `cs/0112017`
   - It does **not** accept a bare number such as `2104.06921` or `pp. 1234.5678` without the "arXiv" marker.
2. Normalise with the existing `normalise_arxiv_id` (drops the version suffix).
3. **Validate the shape** (`valid_arxiv_id`): month 01-12; new-style numbering starts at 0704; 4-digit sequence numbers before 2015 and
   5-digit from 2015 (so `1501.0001` and `2113.00001` are rejected).
4. **Date plausibility (new rule):** reject a cited ID whose YYMM is later than the citing paper's own YYMM (a paper cannot cite the future).
5. **Ambiguity:** if the entry contains several different valid IDs, resolve none (only 1 of 320,803 references is ambiguous).
6. Count each outcome in the build's cleaning log (`refs_resolved_raw_text`, `refs_rejected_implausible`, `refs_ambiguous`), so the
   effect is visible in `dataset_stats.md`.

Order of precedence stays: `ids.arxiv_id`, then `contained_arXiv_ids`, then raw text, so the fallback can only **add** resolutions; it
never overrides an existing one. No schema change; `edges`, `contexts` and `queries` keep their columns.

## 3. Expected benefits (measured, not estimated)

- References resolved: **9.6% -> 19.3%** (+31,119 references found only in the text).
- In-corpus edges: 2,883 -> 6,275 (+118%). Usable contexts: 5,014 -> 10,422 (+108%).
- Queries: the baseline build produced 1,918 queries from 5,014 contexts; with twice the contexts I **estimate** roughly 4,000 queries
  (about 650 test queries). This is an estimate; the real number needs a build.
- More edges make the citation graph less sparse (baseline: 86% of papers are cited by no one in the corpus), which matters for PageRank and the
  popularity analyses.
- It also enlarges the list of cited papers outside the corpus (15,349), which is the input list of the crawler enrichment design.

## 4. Possible false matches, and how much of a problem they are

Measured on the real data:

| Check | Result | Meaning |
|---|---|---|
| Raw-text ID agrees with `ids.arxiv_id` / `contained` where both exist | **8,897 agree, 10 disagree** (99.89%) | The extractor finds the same paper the other sources found |
| Title check: when a raw-only ID lands on a corpus paper, does that paper's title appear in the reference text (>= 80% of its words)? | **96.1%** of 3,403 hits; the existing sources score **96.2%** of 2,897 on the same test | Raw-text matches are as consistent as the sources already trusted |
| Cited ID newer than the citing paper | 81 of 31,119 raw-only references (0.26%) | Rejected by the date rule above |
| Invalid year (`2030`) | 1 distinct ID | Rejected by an upper bound on the year |

The 3.9% that fail the title test are mostly **title variants**, not wrong papers: the sampled failures are correct matches whose
bibliography title differs from the arXiv title (for example "Using Syntactical and Logical Forms to Evaluate Textual Inference
Competence" for arXiv 1905.05704, whose arXiv title is "A logical-based corpus for cross-lingual evaluation"). Remaining risks:

- A reference that *mentions* a second arXiv ID in passing ("see also arXiv:...") would be resolved to the first ID. Only one
  entry in the data contains several IDs, so this is rare, and ambiguous entries are not resolved.
- A typo in the author's bibliography would yield a wrong but valid-looking ID (the same risk applies to the existing sources).
- The preprint and the published version are the same paper, so resolving to the arXiv version is correct.

## 5. Tests needed to validate it

1. **Unit tests for the extractor** (positive and negative examples above): already written in `p4/tests/test_dataprep_resolution.py`
   (40 tests) and reusable as P1's tests.
2. **Golden-number regression:** on the cs.CL subset the current policy gives exactly 5,014 contexts. After the change, the same
   build with the fallback disabled must still give 5,014, and with it enabled must give at least 5,014 (it can only add).
3. **Precision sample:** manually review 100 random raw-only matches that land on corpus papers; require at least 95% correct.
4. **Existing P1 tests** (25) and the leakage assertion (`assert_no_leakage`) must pass unchanged, since new contexts become new
   query candidates and go through the same leakage control.
5. **Cleaning-log test:** the three new counters appear and add up (resolved + rejected + ambiguous = references with a raw-text ID).
6. **Date-rule test:** a reference to a future ID is rejected; an old-style ID is accepted for any citing year.

## 6. A separate bug found by the real build (needs the same approval)

`citeds/splits.py` lines 86 and 89 (`open(..., "w")` for the qrels and topics files) and `citeds/build.py` lines 64 and 71
(`write_text`) use no explicit encoding. On Windows this defaults to cp1252, and the **real build crashed** with
`UnicodeEncodeError` on non-ASCII characters in query text (the synthetic sample is pure ASCII, so tests never saw it).
Proposed fix: add `encoding="utf-8"` to those four calls (the same applies to `citation_function.py` lines 141 and 143 and
`crawler.py` lines 86 and 97). Workaround used for the baseline without changing code: run with `PYTHONUTF8=1`.
