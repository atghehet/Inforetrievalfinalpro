# Leakage-safe evaluation protocol for graph and popularity signals

**Status:** proposal for team approval. **No graph-based retrieval result will be reported until this protocol is approved and
applied.** The audit tool is `p4/graph/leakage_audit.py` (tested: `p4/tests/test_leakage_audit.py`). The numbers below are REAL
data audits of the cs.CL baseline build; they are not retrieval results.

## 1. How P1 splits the queries (verified)

`citeds/splits.py` assigns each query to a split by hashing `(seed, citing_paper_id)`. All queries from one citing paper therefore
fall in the same split. **Verified on the real baseline:** 965 citing papers, 1,918 queries (train 1,328 / val 272 / test 318),
**0 citing papers appear in more than one split**.

## 2. The leaks (verified on the real baseline)

**Leak A: edges and citation counts.** `edges.parquet` and `papers.citation_count` are built from *all* cleaned contexts, including
the sentences that became queries.
- **100% of val/test queries' own citation edge is in the full graph.**
- `papers.citation_count` equals the full-graph in-degree exactly, so it includes each query's own citing paper.
- Any signal computed from this graph (in-degree, PageRank, citation-count buckets, a popularity prior) is raised for exactly the paper the
  query asks for. In the full graph **no** val/test target has zero citations (0.0%); once the held-out citing papers are removed,
  **34.9% of test targets have zero citations**. A filter such as "citation count > 0" or a prior that rewards cited papers would leak.

**Leak B: index contexts by the query's own citing paper (residual, in P1's index).** P1 removes the query sentence and near-duplicates, but
other sentences by the same citing paper about the same target stay in `index_contexts.parquet`.
- **207 of 318 test queries (65.1%)** have at least one such sentence (mean 2.5 among them, max 7).
- If a citing paper repeats its description of the target elsewhere, expansion retrieval benefits from that. This is not covered by the plan's
  "remove the query sentence and exact or near-duplicate copies", so it is a design question for the team, not a P1 bug.

## 3. Proposed protocol

**Rule 1: the evaluation graph.** Remove every edge whose **citing paper has a query in the validation or test split**. Use this graph
for PageRank, in-degree, co-citation, and any fusion prior, for tuning on validation **and** for reporting on test.
- Real effect: 292 held-out citing papers; **502 of 2,720 edges removed (18.5%)**, 2,218 kept.
- One graph, computed once, deterministic. Splits are per citing paper, so this cleanly removes all edges that could have produced a
  held-out query, including the same paper's other citations.

**Rule 2: leakage-safe counts.** Popularity buckets, cold-start buckets and any prior must use the in-degree of the evaluation graph
(`safe_citation_count`), not `papers.citation_count`. Real effect: mean citation count of held-out targets 4.44 -> 2.91.
(P4's existing analyses use `papers.citation_count`, as approved earlier; they must switch to the safe counts before real results are
reported, and say so.)

**Rule 3: index sensitivity analysis (Leak B).** Report two settings for every expansion result:
- **Primary:** P1's index as defined in the plan (query sentence and near-duplicates removed).
- **Strict:** additionally drop all index contexts written by the citing papers of validation/test queries.
  Real effect: 534 of 3,084 index contexts (17.3%) are removed; **143 of 318 test queries (45%) are left with a target that has no index
  context at all**, so those become abstract-only cold-start cases.
The gap between the two settings measures how much of the expansion gain depends on same-paper repetition. I recommend deciding
before experiments which one is primary; the plan implies the first.

**Rule 4: tuning and reporting.** Tune fusion weights and damping on validation only (using the evaluation graph); report test once per
system. Train-split queries may be used for learning only. Never score train queries against the evaluation graph and quote the result.

**Rule 5: verification before any result is reported** (the plan assigns this check to P4):
1. `assert_no_leakage(index_contexts, queries)` from P1 passes.
2. `python -m p4.graph.leakage_audit --processed <P1 processed folder> --out <report.json>` shows the splits disjoint by citing paper, and
   records the edge removal and the numbers above.
3. The run files of P2 and P3 were produced from `index_contexts.parquet` (not `contexts.parquet`); P4 checks this by comparing the
   `context_id`s in the files given to P2 and P3 with the index.

## 4. Diagnostic: how much does Leak A help a prior? (REAL data audit, not a retrieval result)

Rank all 8,843 papers by citation count (the same list for every query) and see where the targets land.

| Split | Graph | MRR | Recall@10 | Median rank of target | Targets with 0 citations |
|---|---|---:|---:|---:|---:|
| test (318) | full graph | 0.0151 | 0.022 | 332 | 0.0% |
| test (318) | evaluation graph | 0.0129 | 0.022 | 722 | 34.9% |
| val (272) | full graph | 0.0203 | 0.026 | 332 | 0.0% |
| val (272) | evaluation graph | 0.0181 | 0.026 | 722 | 33.5% |

Both priors are weak on their own (the graph is sparse), but the leak halves the median rank of the target (722 -> 332), because 86% of
papers have no citations and the query's own edge lifts the target out of that group. In a fusion with a retrieval score this advantage
would add to whatever the text signal does, which is why Rule 1 is required.

## 5. Not covered (limitations)

- **Temporal leakage:** the graph contains citations from papers published after the query's citing paper. A deployed system could not
  know them. A variant that only uses edges from earlier papers is possible but removes more edges.
- **Near-duplicates of query sentences:** left to P1's own `assert_no_leakage`.
- **Co-citation recommender evaluation** (hide some references of a paper): needs its own protocol (the hidden references must be removed
  from the co-citation graph), to be specified when that batch starts.
- The evaluation graph is smaller (2,218 edges on the baseline), so graph-prior effects will be estimated on a very sparse graph.

## 6. Decisions requested from the team

1. Approve Rules 1, 2 and 4 (graph and counts). 2. Decide which index setting is primary (Rule 3). 3. Confirm P2 and P3 build their
indexes from `index_contexts.parquet` only.
