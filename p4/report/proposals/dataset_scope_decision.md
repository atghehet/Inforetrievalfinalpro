# Dataset scope decision: cs.CL, cs.CL + cs.IR, or cs.CL + cs.LG + cs.AI + cs.IR

**Status:** recommendation, updated with the real cs.CL baseline build. Labels: REAL = measured on the real subset; EST = estimate from a stratified
10% sample of the archive (the EST method was validated: for the cs.CL raw-text policy it predicted 0.71 edges per paper and the full audit measured 0.71).

## 1. What the plan asks for

The plan targets "roughly 50,000 to 200,000 papers", restricted to one field (cs.CL or cs.IR). The **open subset cannot deliver that for cs.CL**:
the whole open subset has 165,000 papers (REAL: exactly 165,000 records), of which **8,844 are cs.CL** (REAL). Even the broadest option below
(about 35,000 papers, EST) is under 50,000. The final report must state this as a documented deviation and justify it (the smaller, field-focused
corpus keeps the citation-context experiment meaningful; the 50,000 target would need the full unarXive release, which is access-restricted
and far larger, or another source such as S2ORC).

## 2. The three scopes

| | A: cs.CL | B: cs.CL + cs.IR | C: cs.CL + cs.LG + cs.AI + cs.IR |
|---|---:|---:|---:|
| Papers | **8,844 (REAL)** | ~11,000 (EST) | ~35,000 (EST) |
| Share from 2020-2022 | 88.0% (REAL: 7,779 of 8,843 kept papers) | ~86% (EST) | ~90% (EST) |
| Filtered JSONL | 0.83 GB (REAL) | ~1.1 GB (EST) | ~4.9 GB (EST) |
| Subset extraction time | 143 s (REAL) | similar (one pass over the tar) | similar |
| P1 build time | 326 s (REAL) | ~7 min (EST: scales with papers) | ~22 min (EST: scales with papers) |
| In-corpus edges, **baseline policy** | **2,720 (REAL)** | ~3,300 (EST: assumes about 46% of the with-fallback figure, as in A) | ~9,800 (EST: same assumption) |
| In-corpus edges, **with raw-text fallback** | **6,275 (REAL, audit)** | ~7,100 (EST) | ~21,000 (EST) |
| Usable contexts, baseline / with fallback | **5,014 / 10,422 (REAL)** | ~4,800 / 9,900 (EST, same assumption) | ~19,700 / 41,000 (EST, same assumption) |
| Queries (train/val/test), baseline | **1,328 / 272 / 318 (REAL)** | n/a | n/a |
| Queries with fallback | ~4,000 (EST, about 650 test) | n/a | n/a |

(Per-paper rates hardly change between scopes: about 0.7 edges and 1.0-1.2 contexts per paper with the fallback. A wider scope adds volume, not density.)

## 3. Is cs.CL enough for the research questions?

| Research question | Needs | Verdict for scope A (REAL numbers) |
|---|---|---|
| RQ1 expansion vs abstract-only | Many queries, targets with contexts | **Sufficient:** 318 test queries already; about 650 with the fallback (EST). But index contexts per target are few (median 1) and 65% of test queries have same-citer contexts in the index (see the leakage protocol) |
| RQ2 BM25 vs dense vs hybrid | Many queries and a candidate pool | **Sufficient:** 8,843 candidates, 318 test queries |
| RQ3 PageRank prior vs popularity bias | A citation graph with structure | **Weak:** 2,720 edges on 8,843 papers, 86% of papers cited by nobody (REAL); the leakage-safe graph keeps 2,218 edges. The fallback roughly doubles this but the graph stays sparse |
| RQ4 cold start and citation function | Papers with few contexts; labels | **Sufficient by construction** (most targets have 1-2 contexts), but the function labels do not exist yet (P1's classifier) |

## 4. Recommendation

1. **Stay with scope A (cs.CL, all years).** It is the smallest scope that supports RQ1, RQ2 and RQ4 with real data today, and it is already built.
2. **Do not widen the scope now.** Scope B adds about 5% edges for 25% more papers; scope C quadruples the volume while diluting the NLP focus with
   general machine-learning papers, and it does not make the graph denser (about 0.6 edges per paper in every scope).
3. **Increase density, not size,** in this order: (a) the raw-text fallback (free, about 2x edges and contexts, proposal written);
   (b) the crawler enrichment of cited papers outside the corpus (measured by a small pilot first, design written).
4. **Decision rule for widening later:** only if, after (a) and (b), the validation or test query count is below about 300 per split or the
   graph still cannot support RQ3, consider scope B first, scope C last.

## 5. Risks of this recommendation

- The corpus is far below the plan's 50,000-paper target (documented deviation).
- 88% of papers are from 2020-2022 (REAL): temporal analyses will have thin older groups (only 625 papers before 2019 and 1,064 before 2020 in the baseline).
- The graph-based question (RQ3) may end with a negative or inconclusive result because of sparsity; the report should present that honestly.
