# Handout: OpenAlex crawler for fresh papers (optional P1 task)

**Goal:** collect recent cs.CL-style papers from OpenAlex, published after our
unarXive snapshot (March 2023), for the freshness and cold-start experiment.
These papers have few or no citations, so they show how retrieval behaves when
a paper has no citation contexts. The output is one table,
`data/processed/fresh_papers.parquet`.

The crawler code is already written and unit-tested **offline**. It has
**never been run against the live OpenAlex API**. Your job is to configure
it, check its filter against the live API, run it, and add a small merge step.

## 1. Where the code is

| File | What is there |
|---|---|
| `citeds/crawler.py` | The crawler. Key parts: `API` (line 41), `DEFAULT_FILTER` (line 44), `PoliteClient` (caching, rate limiting, robots.txt, retries, line 57), `abstract_from_inverted_index` (line 108), `arxiv_id_of` (line 118), `work_to_row` (line 127), `crawl` (paging, line 140), `main` (command line, line 164) |
| `citeds/config.py` | `CONTACT_EMAIL` (line 67, placeholder), `CRAWL_CACHE_DIR` = `data/external/crawl_cache/`, `CRAWL_MIN_INTERVAL_S = 0.2` (at most 5 requests per second), `FRESH_PAPERS_PATH` |
| `citeds/schema.py` | `FRESH_PAPERS` schema (line 70): the `papers` columns plus `openalex_id`, `openalex_cited_by_count`, `referenced_works`, `publication_date` |
| `tests/test_citation_function_and_crawler.py` | Offline tests: placeholder email refused, abstract rebuilt, row conversion, response caching |

Line numbers are as of the `integration` branch; search for the names if they've moved.

## 2. Setup

From the repo root, with the project venv active (see the root README):

```powershell
pip install -r requirements.txt
python -m pytest -q tests/test_citation_function_and_crawler.py
```

Also put the processed data in `data/processed/` (unzip `member1-data-v3.zip`
there). The crawler skips papers already in `papers.parquet`.

## 3. How to connect to OpenAlex

* **API:** `https://api.openalex.org/works`. It's free and needs no login.
* **Polite pool:** OpenAlex asks clients to identify themselves with an email. The crawler sends it twice: in the `User-Agent` header (`citeds-course-project/0.1 (mailto:...)`) and as the `mailto=` query parameter. Identified clients get faster, more reliable service.
* **Limits:** keep to a few requests per second and stay well below the daily cap. Check the current limits, and whether an API key is now required, at https://docs.openalex.org before running. Our setting of 0.2 s between requests is conservative.
* **Paging:** the crawler uses cursor paging (`cursor=*`, then `meta.next_cursor`) with 200 results per page.
* **Caching:** every response is saved in `data/external/crawl_cache/`, keyed by URL, so re-running never re-downloads a page. Delete that folder to force a fresh crawl.
* **Abstracts:** OpenAlex returns `abstract_inverted_index` (word → positions). `abstract_from_inverted_index` rebuilds the plain text.

Quick connectivity check in a browser or PowerShell. Use the real email in place of the placeholder:

```powershell
Invoke-RestMethod "https://api.openalex.org/works?per-page=1&mailto=you@university.edu" | Select-Object -ExpandProperty meta
```

## 4. What you need to add or edit

### 4.1 Contact email (required)

The crawler refuses to run while the email is the placeholder
`YOUR_EMAIL@example.com`. Set it as an environment variable rather than editing
the code, so it never ends up in Git:

```powershell
$env:CITEDS_CONTACT_EMAIL = "you@university.edu"
```

(Linux/macOS: `export CITEDS_CONTACT_EMAIL=you@university.edu`.) Use an address the team agrees to share with OpenAlex.

### 4.2 Check and fix the filter (required)

`DEFAULT_FILTER` in `citeds/crawler.py` is:

```
concepts.id:C204321447,has_abstract:true,locations.source.id:S4306400194
```

* `C204321447` was meant to be the concept "Natural language processing".
* `S4306400194` was meant to be the source "arXiv".

**Neither ID has been checked against the live API.** OpenAlex is also
replacing "concepts" with "topics". Look the IDs up:

```powershell
Invoke-RestMethod "https://api.openalex.org/sources?search=arxiv&mailto=you@university.edu" | Select-Object -ExpandProperty results | Select-Object id, display_name -First 5
```
```powershell
Invoke-RestMethod "https://api.openalex.org/concepts?search=natural%20language%20processing&mailto=you@university.edu" | Select-Object -ExpandProperty results | Select-Object id, display_name -First 5
```
```powershell
Invoke-RestMethod "https://api.openalex.org/topics?search=natural%20language%20processing&mailto=you@university.edu" | Select-Object -ExpandProperty results | Select-Object id, display_name -First 5
```

Then test the filter with a small count query before crawling:

```powershell
Invoke-RestMethod "https://api.openalex.org/works?filter=concepts.id:C204321447,has_abstract:true,locations.source.id:S4306400194,from_publication_date:2024-01-01&per-page=1&mailto=you@university.edu" | Select-Object -ExpandProperty meta
```

* If `meta.count` is in the thousands, the filter works.
* If it's 0 or there's an error, correct the IDs in `DEFAULT_FILTER`. You can switch to a topic filter, for example `primary_topic.id:<id>`, or pass `--filter "..."` on the command line instead of editing the code.
* Keep `has_abstract:true` and the arXiv source filter. The crawler only keeps works that have an arXiv ID (`arxiv_id_of`), because our `paper_id` is the arXiv ID.

Note in the data card which filter you used, since OpenAlex's subject labels
are not the same as arXiv's cs.CL category.

### 4.3 Run it, small first

```powershell
python -m citeds.crawler --since 2024-01-01 --max 50
```
```powershell
python -c "import pandas as pd; d=pd.read_parquet('data/processed/fresh_papers.parquet'); print(len(d)); print(d[['paper_id','year','title']].head())"
```

Check that the IDs look like `2401.01234`, the titles look like NLP papers,
and the abstracts are readable. Then run the real crawl, for example:

```powershell
python -m citeds.crawler --since 2023-04-01 --max 2000
```

### 4.4 Add: a merge step for the freshness experiment (new code)

Right now the crawler only writes `fresh_papers.parquet`. Nothing uses it yet.
Add a small script, `scripts/merge_fresh.py`, that:

1. reads `papers.parquet` and `fresh_papers.parquet`;
2. drops fresh papers whose `paper_id` is already in `papers`;
3. writes `data/processed/papers_with_fresh.parquet` with the `papers` columns (`citation_count` = 0 and `num_contexts` = 0 for the fresh ones);
4. prints how many papers were added.

It must not touch `queries`, `qrels` or `index_contexts`. Fresh papers are
extra candidates (distractors), not new queries. That lets P2 and P3 rerun
retrieval with a larger candidate set and measure the effect. Add one unit
test with a tiny in-memory example, and update `docs/SCHEMA.md` and `docs/DATA_CARD.md`.

### 4.5 Optional extensions

* **Citation links:** `referenced_works` holds OpenAlex IDs. Mapping them to our corpus needs each corpus paper's OpenAlex ID. A lookup by arXiv DOI (`10.48550/arXiv.<id>`) through `/works/doi:...` can add those. That could also recover some of the 83,371 citing sentences dropped during the build, but it's a separate, larger task.
* **Exclude after the snapshot date:** pass `--since 2023-04-01` so nothing overlaps the unarXive snapshot (March 2023).

## 5. Done means

- [ ] `CITEDS_CONTACT_EMAIL` set (not committed)
- [ ] Filter IDs checked against the live API; the filter used is written in the data card
- [ ] `fresh_papers.parquet` created; count, year range and 5 sample titles reported to the team
- [ ] `scripts/merge_fresh.py`, with a test, writes `papers_with_fresh.parquet`
- [ ] `python -m pytest -q` passes
- [ ] Work on a branch (for example `p1-crawler`), with a pull request into `main`, reviewed by one teammate

## 6. Rules

* Don't commit data, the crawl cache or the email (`data/` is git-ignored).
* Respect rate limits and robots.txt. The crawler already does both; don't lower `CRAWL_MIN_INTERVAL_S`.
* OpenAlex data is CC0. Say so in the data card.
* Don't use fresh papers as queries, and don't change the test split.
