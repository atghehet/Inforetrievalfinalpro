# Crawler enrichment design

**Status:** design proposal. **Nothing has been crawled.** No live API call has been made. The pilot below needs your approval and a
contact email. API facts are quoted from the official documentation read on 2026-11-22; items I could not verify are marked
**unverified**.

## 1. Audit of the existing crawler (`citeds/crawler.py`, P1)

| Aspect | What the code does |
|---|---|
| Source | OpenAlex `works` API only |
| Purpose | The plan's optional "freshness / cold-start" crawler: recent papers with abstracts, published after `--since` (default 2024-01-01), max 2,000 |
| Query | `concepts.id:C204321447` (NLP concept), `has_abstract:true`, `locations.source.id:S4306400194`; sorted by date; cursor paging, `per-page 200` |
| Politeness | Requires a contact email (refuses the placeholder), checks robots.txt, minimum 0.2 s between requests, retries with backoff on 429/5xx, caches every response on disk by URL hash |
| Output | `fresh_papers.parquet`: P1's papers columns + `openalex_id`, `openalex_cited_by_count`, `referenced_works` (OpenAlex IDs), `publication_date`; `citation_count` and `num_contexts` are 0 by construction |
| Duplicates | Dropped by arXiv ID within the crawl and against an existing `papers.parquet` |
| Missing data | Rows without an arXiv ID or with an abstract under 20 tokens are dropped **silently** (not counted or logged) |
| References | Kept as raw OpenAlex IDs; never resolved to arXiv IDs, so it adds **no graph edges** |
| Integration | **None.** Nothing reads `fresh_papers.parquet` (checked by search). It was never run against the live API (its own docstring says so) |

**Problems against the current OpenAlex documentation** (help.openalex.org, read today; check again before use):
- `per-page: 200` is deprecated; the supported maximum is **100**.
- Cursor paging "needs a free API key past the first page"; without a key, following `next_cursor` returns **401**.
- I could not find the "polite pool" / `mailto` mechanism the crawler relies on anywhere in the current documentation (**unverified**:
  it may have been removed or may be documented elsewhere).
- The source ID `S4306400194` **is** arXiv (confirmed on the locations page); the filter syntax for it on `/works` is not shown in the
  documentation (**unverified**).
- Concepts are "superseded" by topics; the NLP concept filter may stop working.

So the existing crawler would likely need a key and small changes even for its original purpose.

## 2. The coverage gap (REAL, from the cs.CL baseline)

The open subset holds only permissively licensed papers (about 9% of arXiv). Most cited papers are not in it.

- 8,844 cs.CL papers, 320,803 references (36.3 per paper).
- Resolved to an arXiv ID: 9.6% with P1's current policy, **19.3%** with the raw-text fallback (see `p1_raw_text_reference_fallback.md`).
- Of the resolved citations, 42,033 citing contexts (current policy) point at a cited paper that is **not** in the corpus and are dropped
  (only 5,014 contexts, 10.7%, survive).
- Distinct cited arXiv papers outside the corpus: **9,431** (current policy), **15,349** (with the fallback), mostly from 2018-2021
  (2018: 2,018; 2019: 2,832; 2020: 3,637; 2021: 2,561; 2022: 981; pre-2007 style IDs: 104).

If a cited paper outside the corpus had a title and abstract, it could join the corpus as a **document**, and the citing sentences from the open
subset that cite it would become usable contexts and query candidates. That is the coverage lever: it is applied to references that
already exist, so it needs no new full text.

## 3. Proposed enrichment crawler (improve the existing one in place, same schema family)

**Goal:** fetch title, abstract, categories and year for the cited arXiv IDs that are outside the corpus, keep the cs.CL ones, and
add them to P1's papers table before the "cited paper must be in the corpus" filter.

**Input.** A text file of distinct cited arXiv IDs outside the corpus. P4's `resolution_audit` already computes this set; P1's build
would export it (`--export-missing-ids`).

**Source (recommended): the arXiv API.** Documented facts (info.arxiv.org):
- `id_list` takes a comma-separated list of arXiv IDs and returns one Atom entry per ID. **The maximum number of IDs per request is not
  documented** (unverified); `max_results` can be 2,000 per slice. The batch size must be set by the pilot (start at 50).
- Fields per entry: `id`, `title`, `summary` (abstract), `category` terms, `arxiv:primary_category`, `published` (first version date),
  `updated`, optional `arxiv:doi`, `arxiv:journal_ref`, `arxiv:comment`. Malformed IDs return an Atom error entry.
- **Rate limit (Terms of Use): "no more than one request every three seconds", one connection at a time.** The bulk-data page suggests
  bursts of 4 requests per second, which conflicts with the ToU; I will follow the stricter ToU.
- **Caching:** the user manual says "Please cache your results."
- **Terms:** descriptive metadata (title, abstract, authors, identifiers, classification) is free to "retrieve, store, transform, and share"
  under **CC0**; storing or serving the e-prints themselves is not allowed without the copyright holder's permission. We store only metadata.
- For bulk needs the docs point to OAI-PMH ("the preferred way to bulk-download"); "do not attempt to download the complete corpus
  programmatically". For about 15,000 IDs the API is proportionate; for more than about 50,000, OAI-PMH should be considered instead.

**Why not OpenAlex or Semantic Scholar as the primary source:**
- OpenAlex: no confirmed works-level arXiv-ID filter (unverified); an OR filter takes up to 100 values (documented for DOIs); limits 10 requests per
  second without a key, 30 with a free key; abstracts arrive as an inverted index that must be rebuilt. Useful later for `referenced_works`
  edges, but not needed for coverage.
- Semantic Scholar: 1 request per second with a key; the shared unauthenticated pool can be throttled; I could not read the batch-endpoint
  documentation (unverified); terms require attribution and restrict redistribution.

**Politeness and reliability.** One connection, at most 1 request per 3 seconds, a User-Agent that identifies the project and a contact
address (supplied by you), response cache keyed by the request, resume from the cache, back off on HTTP 429/5xx, stop on repeated errors.
Reuse the existing `PoliteClient` structure (cache, backoff, robots check) rather than building a new pipeline.

**Expected cost (estimate).** With the fallback: 15,349 IDs. At 50 per request that is about 307 requests, about 15 minutes at one request
per 3 seconds; at 100 per request about 8 minutes. Storage: a few tens of MB of cached XML.

## 4. Output format

`external_papers.parquet` (new, in a folder outside the repo): P1's `PAPERS` schema (`paper_id`, `title`, `abstract`, `year`, `categories`,
`citation_count` = 0, `num_contexts` = 0) plus `source` (`arxiv_api`), `primary_category`, `published`, `fetched_at`.
`crawl_rejects.parquet`: `paper_id`, `reason` (`not_found`, `withdrawn`, `no_abstract`, `abstract_too_short`, `wrong_field`, `bad_id`, `http_error`)
so nothing disappears silently.

## 5. Deduplication, missing metadata, invalid references

- IDs are normalised with the existing `normalise_arxiv_id` (version suffix removed, old-style `archive/YYMMNNN`); duplicates inside the
  input and against the corpus and the cache are removed **before** requests.
- A paper already in the open subset is never re-fetched. Entries returned as errors or without an abstract of at least 20 tokens
  (the same threshold P1 uses) go to `crawl_rejects` with a reason.
- Invalid references: IDs failing `valid_arxiv_id`, or newer than the citing paper (see the fallback proposal), are not queried.
- The category filter (cs.CL, the approved scope) is applied after the fetch using `categories`.

## 6. How it fits P1's pipeline (no separate pipeline)

1. P1's build runs as today and exports the missing-ID list (small P1 change, needs approval).
2. The crawler fills `external_papers.parquet`.
3. P1's build gets an optional `--extra-papers` input: the extra papers are added to the corpus **before** `contexts_cited_outside_corpus`
   is applied. Everything downstream (queries, splits, leakage control, edges) is unchanged.
4. Contexts still come only from open-subset full text; the added papers contribute **abstracts and metadata (CC0)**, never full text, which
   keeps to the plan's ethics rule (no republishing of copyrighted full text).

Consequence to state in the report: documents added this way have **no citing contexts written by themselves** and are candidates whose
contexts come from open-subset citers; the corpus becomes a mix of open-subset papers and metadata-only papers, and the report must say so.

## 7. Expected benefit and how to measure it (no number is assumed)

Unknown until measured: what share of the 15,349 outside-corpus IDs are cs.CL, and have abstracts. **Pilot** (needs your approval):
- Sample **500** outside-corpus IDs, stratified by ID year, from the 15,349.
- Request them in batches of 50 (10 requests, about 30 seconds at one request per 3 seconds).
- Measure: found rate, abstract rate, cs.CL share (with a confidence interval), and an extrapolated gain in corpus size, edges and contexts.
- **Decision rule:** proceed to the full crawl only if the projected gain justifies the added complexity (for example, a projected increase of at
  least 25% in usable contexts); otherwise skip enrichment and keep the fallback alone.

## 8. What I need from you before any API call

1. Approval of the pilot (about 10 requests).
2. The contact email to put in the User-Agent. I will not use yours without explicit approval.
3. Decision on the P1 changes this design needs (missing-ID export, `--extra-papers`).

The plan's original **freshness crawler** stays optional. If wanted later, it needs the OpenAlex fixes in section 1 (per-page 100, API key).
