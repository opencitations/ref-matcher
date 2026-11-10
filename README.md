# Reference Matching Tool

Repository for a bibliographic reference matching tool designed to match references from Crossref JSON files against [OpenCitations Meta](https://opencitations.net/meta). It implements a heuristic-based approach, enabling the retrieval and validation of bibliographic metadata even in cases of incomplete or inconsistent citation records and generates comprehensive reports with detailed statistics.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Workflow](#workflow)
- [Scoring System](#scoring-system)
- [Output Files](#output-files)
- [Logging System](#logging-system)
- [Error Handling](#error-handling)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Multi-Format Support**: Processes Crossref JSON and TEI XML files
- **Async Architecture**: Concurrent processing with asyncio and aiohttp for high performance
- **Intelligent SPARQL Matching**: 6 query strategies with early stopping when threshold is met
- **Sophisticated Scoring System**: Weighted scoring (max 48 points) based on DOI, title, authors, year, volume, and pages
- **GROBID Integration**: Enriches references using GROBID for unstructured text parsing
- **Bulk DOI Mode** (`--bulk`): Answers the ingestion question for DOI-bearing references thousands per request instead of one at a time, with optional author corroboration
- **Batch Matching of a Whole Dump** (`script/batch/`): The tool's rules as DuckDB joins over Meta's tables exported to Parquet, ~14× faster with the same matches, then a merge of all signals with reliability levels
- **Comprehensive Logging**: Multi-file logging system with 5 specialized logs
- **Rate Limiting**: Token bucket algorithm (2.5 req/s, burst of 10)
- **Concurrent Processing**: Semaphore-controlled parallelism (10 concurrent references)
- **Dynamic Threshold**: Automatic threshold adjustment (90% trigger)
- **Detailed Statistics**: Match rates, field contributions, query type distribution

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Reference Matching Tool                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
          ┌─────────────────────────────────────────┐
          │      1. Reference Extraction Phase      │
          └─────────────────────────────────────────┘
                      │                │
                      ▼                ▼
            ┌──────────────┐     ┌──────────────┐
            │JSON/TEI Files│  +  │    GROBID    │
            │              │     │   Fallback   │
            └──────────────┘     └──────────────┘
                      │                 │
                      └────────┬────────┘
                               ▼
            ┌─────────────────────────────────────────┐
            │  2. Reference Normalization Phase       │
            │  - Clean titles, authors, DOIs          │
            │  - Normalize text (Unicode, accents)    │
            │  - Extract numeric fields (year, pages) │
            └─────────────────────────────────────────┘
                                │
                                ▼
            ┌─────────────────────────────────────────┐
            │  3. SPARQL Query Construction Phase     │
            │                                         │
            │  Query Execution (sequential):          │
            │  1. year_and_doi                        │
            │  2. doi_title                           │
            │  3. author_title                        │
            │  4. year_author_page                    │
            │  5. year_volume_page                    │
            │  6. year_author_volume                  │
            │                                         │
            │  Early stop when score >= threshold     │
            └─────────────────────────────────────────┘
                                │
                                ▼
            ┌─────────────────────────────────────────┐
            │  4. OpenCitations SPARQL Query Phase    │
            │  - Async query execution                │
            │  - Rate limiting                        │
            │  - Token bucket algorithm               │
            │  - Error handling                       │
            │  - Max 3 retries with backoff           │
            └─────────────────────────────────────────┘
                                │
                                ▼
            ┌─────────────────────────────────────────┐
            │  5. Candidate Scoring Phase             │
            │                                         │
            │  Scoring Components:                    │
            │  ├─ DOI Exact Match: 15 pts             │
            │  ├─ Title Similarity: 14-10 pts         │
            │  ├─ Author Match: 7 pts                 │
            │  ├─ Year Match: 1 pt                    │
            │  ├─ Volume Match: 3 pts                 │
            │  └─ Page Match: 8 pts                   │
            │                                         │
            │  Threshold: 26/48 points (54.5%)        │
            └─────────────────────────────────────────┘
                                │
                                ▼
            ┌─────────────────────────────────────────┐
            │  6. Result Generation & Export Phase    │
            │  - CSV matched references               │
            │  - CSV unmatched references             │
            │  - HTML processing report               │
            │  - Statistics text file                 │
            │  - 5 specialized log files              │
            └─────────────────────────────────────────┘
```

---

## Installation

### Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (project & dependency manager)
- GROBID (optional, for processing fallback)
- Internet connection (for OpenCitations SPARQL endpoint)

### Setup — the virtual environment with `uv`

The project uses **[uv](https://docs.astral.sh/uv/)** to manage a local virtual
environment and its dependencies, declared in `pyproject.toml` and pinned in
`uv.lock`. You do **not** create or activate a `venv` by hand, and you do **not**
run `pip install` — `uv` does it for you.

From the repository root, create/sync the environment:

```bash
uv sync
```

This creates a `.venv/` directory (git-ignored) and installs the exact locked
dependencies (aiohttp, rapidfuzz, Unidecode, tqdm, requests, python-dotenv,
SPARQLWrapper). Re-run `uv sync` whenever `pyproject.toml`/`uv.lock` change.

Then run anything through **`uv run`** — it automatically uses `.venv` without
you activating it or setting `PYTHONPATH`:

```bash
# the matcher
uv run script/ReferenceMatchingTool.py <input> [options]

# the evaluation script
uv run script/evaluation.py <action> <input> [options]

# an arbitrary Python command inside the project env
uv run python -c "import aiohttp, rapidfuzz; print('env OK')"
```

> Note: fuzzy title matching uses **rapidfuzz** (the maintained successor to
> fuzzywuzzy); `uv sync` installs it, so there is nothing extra to set up.

If you prefer a classic activated shell you *can* do it, but it is optional:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
python script/ReferenceMatchingTool.py <input> [options]
```

Running the unit tests (uses a throwaway pytest, no change to the locked env):

```bash
uv run --with pytest pytest script/tests/
```

## Configuration

Operational settings can be provided through an external `.env` file, so you don't have to retype CLI flags on every run. Copy the template and edit it:

```bash
cp script/.env.example script/.env
```

The `.env` is loaded automatically when it sits in the directory you run the tool from (or a parent), or next to `ReferenceMatchingTool.py`. You can also point to one explicitly with `--env-file /path/to/.env`.

**Precedence** (strongest to weakest): explicit CLI flag → real environment variable → `.env` file → built-in default.

| `.env` variable | Equivalent CLI flag | Default | Description |
|---|---|---|---|
| `SPARQL_ENDPOINT` | `--endpoint` | `https://sparql.opencitations.net/meta` | OpenCitations Meta SPARQL endpoint |
| `MAX_CONCURRENT_REFERENCES` | `--max-concurrent-references` | `10` | References processed in parallel |
| `BURST_SIZE` | `--burst-size` | `10` | Max simultaneous SPARQL HTTP requests |
| `RATE_LIMIT` | `--rate-limit` | `2.5` | Average requests per second |
| `TIMEOUT` | `--timeout` | `600` | Per-query timeout (seconds) |
| `MAX_RETRIES` | `--max-retries` | `3` | Retries for failed queries |
| `THRESHOLD` | `--threshold` | `26` | Minimum matching score (0–48) |
| `BATCH_SIZE` | `--batch-size` | `3` | Files per batch (`--batch` mode) |
| `PAUSE_DURATION` | `--pause-duration` | `0` | Pause between batches (seconds) |
| `VALIDATE_OUTPUT` | `--validate-output` | `false` | Re-process a file once if its output looks empty |
| `QUERY_CACHE_SIZE` | `--query-cache-size` | `50000` | SPARQL results reused for identical queries (0 = off) |
| `ERROR_THRESHOLD` | `--error-threshold` | `10` | Max consecutive server errors before stopping |
| `LOG_LEVEL` | `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `USE_GROBID` | `--use-grobid` | `false` | Enable GROBID fallback |
| `GROBID_CONFIG_PATH` | `--grobid-config` | (auto-search) | Path to the GROBID config file |
| `USE_DOI` | `--use-doi` / `--no-doi` | `true` | Use DOI-based queries |

## Usage

### Basic Usage

#### Process Crossref JSON File
```bash
uv run script/ReferenceMatchingTool.py crossref_references.json \
    --output output_file.csv \
    --threshold 26 \
    --use-grobid \
    --grobid-config grobid_config.json
```
#### Process TEI XML File

```bash
uv run script/ReferenceMatchingTool.py references.tei.xml \
    --output output_file.csv \
    --threshold 26 \
    --use-grobid \
    --grobid-config grobid_config.json
```
#### Process Directory (Batch Mode)
```bash
uv run script/ReferenceMatchingTool.py input_directory/ \
    --batch \
    --output output_directory/ \
    --threshold 26 \
    --use-grobid
```
#### Disable DOI_based query Usage
```bash
uv run script/ReferenceMatchingTool.py crossref_references.json \
    --output matches.csv \
    --no-doi
```
#### Adjust Rate Limiting and Burst Size
```bash
uv run script/ReferenceMatchingTool.py crossref_references.json \
    --output matches.csv \
    --rate-limit 1.5 \
    --burst-size 5
```

### Command-Line Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `input` | str | **Required.** Path to input Crossref JSON or TEI XML file, or directory for batch processing | - |
| `--output, -o` | str | Output CSV file path (single mode) or directory (batch mode). Auto-generated if not specified | Auto-generated |
| `--threshold, -t` | int | Minimum matching score (0-48) required to consider a reference as matched. Lower = more permissive | 26 |
| `--use-grobid` | flag | Enable GROBID fallback to extract metadata from unstructured citation text when initial matching fails | False |
| `--grobid-config` | str | Path to GROBID configuration JSON file. If not specified, searches: current directory (`grobid_config.json`), `~/.grobid/config.json`, script directory, parent directories (up to 3 levels), and `GROBID_CONFIG_PATH` environment variable | None (auto-search) |
| `--batch, -b` | flag | Enable batch mode to process all JSON/XML files in the input directory concurrently | False |
| `--use-doi` | flag | Include DOI-based queries (year_and_doi, doi_title) in the matching strategy. Default enabled | True |
| `--no-doi` | flag | Disable DOI-based queries. Useful when DOI metadata is unreliable or missing | - |
| `--timeout` | int | Maximum time in seconds to wait for each SPARQL query response before timing out | 600 |
| `--max-retries` | int | Number of retry attempts for failed SPARQL queries (handles transient network errors) | 3 |
| `--batch-size` | int | Number of files processed simultaneously per batch. All files share **one** rate limiter, so `--rate-limit` is the real request rate of the whole run regardless of the batch size | 3 |
| `--pause-duration` | int | Delay in seconds between processing batches. With `--batch-size 1` it is applied after **every** file/work (10 s × 84k works ≈ 10 days of idle time), so it defaults to 0; the rate limiter already protects the server | 0 |
| `--query-cache-size` | int | Maximum number of SPARQL results kept in memory and reused when the exact same query is issued again — which happens systematically in the GROBID and "without year" passes, and whenever the same work is cited again. Reusing an identical query's result cannot change any score; it only saves requests. The run ends by printing how many queries were sent and how many were served from the cache. `0` disables it | 50000 |
| `--validate-output` | flag | After each file, check that its output CSV/stats are not near-empty and re-process it once if they are. Off by default because files with 0 references or 0 matches also produce a near-empty CSV and would be processed twice; turn it on if you suspect truncated outputs | False |
| `--error-threshold` | int | Maximum number of consecutive server errors (5xx) before stopping batch processing | 10 |
| `--log-level` | str | Verbosity of logging output: DEBUG (detailed), INFO (standard), WARNING, or ERROR (minimal) | INFO |
| `--rate-limit` | float | Maximum SPARQL queries per second for the whole run (one limiter shared by every file) to respect the OpenCitations limit of 180 requests/minute | 2.5 |
| `--burst-size` | int | Maximum number of concurrent requests allowed in token bucket before rate limiting kicks in | 10 |
| `--no-threshold-adjustment` | flag | Enforce the raw `--threshold` instead of the adaptive lowering to 90% when a score is close | (adjustment on) |
| `--force-restart` | flag | Batch mode: ignore any existing checkpoint in the output directory and reprocess every file from scratch | False |
| `--dump` | flag | Treat the input as a Crossref **dump** (`{"items":[...]}` JSON file, or a `.tar.gz` of such files) — each work is expanded to the matcher's input format and batch‑processed. Works with no references (no `reference-count` or no `reference` list — nothing to match) are not sent to the matcher: they are listed once in `works_without_references.tsv` (work, DOI, reference-count) and checkpointed as usual | False |
| `--limit` | int | With `--dump`, process only the first N works (0 = all). Handy for a quick test on a huge dump | 0 |

> **Checkpointing (batch mode):** progress is saved to `processing_checkpoint.pkl`
> **inside the `--output` directory**, so runs with different `-o` folders track
> their progress independently and can be resumed just by re-running the same
> command. Use `--force-restart` to redo a folder from scratch.

### Processing a Crossref dump (`--dump`)

The public Crossref data dump stores **many** works per file as
`{"items": [ {work}, ... ]}`, which is a different shape from the single‑work
API response (`{"message": {work}}`) the matcher reads. `--dump` bridges this: it
**streams** the dump one file at a time, wraps each work as `{"message": work}`
in memory, and matches it through the normal pipeline (one output CSV per work).
Because it streams — rather than expanding the whole dump to disk first — peak
disk/RAM stays bounded even for a multi‑terabyte dump. It accepts either a single
dump `.json` file or a `.tar.gz` archive of dump files, and is resumable (the
per‑output‑dir checkpoint skips works already done, so a re‑run continues where it
stopped).

```bash
# quick test on just the first work of a dump file
uv run script/ReferenceMatchingTool.py 0.json --dump --limit 1 -o out_dir --use-grobid

# a whole .tar.gz dump (all works)
uv run script/ReferenceMatchingTool.py crossref-data-2026-06.tar.gz --dump -o out_dir --use-grobid
```

`--limit N` caps the number of works (great for smoke‑testing a huge dump before
committing to a full run).

### Bulk DOI mode (`--bulk`)

The matcher asks one question per reference. `--bulk` asks one question for
**two thousand DOIs at a time**, and answers only the ingestion question — *is
this reference already on OpenCitations Meta?* — for the references that carry a
DOI. It lives in a separate module (`script/bulk_doi.py`), imported by the main
tool, so the two logics stay apart: bulk never scores, never calls GROBID and
never touches the matcher's code path.

Measured on the same 1,520,805 references: the matcher took 20.9 days and
millions of requests, bulk took 24 minutes and 425 requests.

**Three phases**

| Phase | What it does | Where |
|---|---|---|
| 1 | Streams the dump one member at a time (constant RAM) and splits references into *with DOI* / *without DOI* | local |
| 1.5 | Deduplicates the DOIs on disk (`sort -u`, in‑memory fallback) — the same work cited a thousand times becomes one DOI to check | local |
| 2 | Checks the unique DOIs against Meta with `VALUES` queries, rate‑limited and checkpointed (resumable) | network |

```bash
# whole dump, existence check only
uv run script/ReferenceMatchingTool.py crossref-data-2026-06.tar.gz --bulk -o out_dir

# also corroborate with the author surname
uv run script/ReferenceMatchingTool.py crossref-data-2026-06.tar.gz --bulk --bulk-authors -o out_dir

# stand-alone, without the main tool
uv run script/bulk_doi.py crossref-data-2026-06.tar.gz -o out_dir --authors
```

| Flag | Default | Meaning |
|---|---|---|
| `--bulk` | off | Enable bulk mode (implies a dump input) |
| `--bulk-chunk-size` | 2000 | DOIs per SPARQL request |
| `--bulk-max-per-min` | 170 | Request ceiling per minute (the endpoint allows 180) |
| `--bulk-authors` | off | Also fetch author surnames and corroborate (see below) |
| `--bulk-same-as DIR` | — | Process only the works the matcher already completed in `DIR`, and write `references_index.csv`, so the two methods can be compared with `compare_bulk_matcher.py` |

**Output files**

| File | Contents |
|---|---|
| `matched_dois.txt` | DOIs already on Meta — nothing left to do |
| `unmatched_dois.txt` | DOIs not on Meta: **candidates** for ingestion, to be validated |
| `no_doi_references.jsonl` | The references without a DOI, for the matcher |
| `bulk_report.txt` / `bulk_summary.json` | Counts and per‑phase timings |
| `check_errors.txt` | DOIs whose chunk failed twice (retries, then a second pass at the end of the run); never counted as absent |

#### Author corroboration (`--bulk-authors`)

A DOI found on Meta proves the *identifier* is there; it does not prove the
reference meant that work (the DOI printed in the reference can be wrong).
`--bulk-authors` adds the check in the **same** pass: for each chunk, a second
query fetches the author surnames of the DOIs found present, and they are
compared locally against the surname in the reference. Existence never depends
on that second query (a plain join; the same pattern inside an `OPTIONAL` is
~150× slower on QLever).

The comparison strips accents, punctuation and one‑letter initials, matches on
word sets (so `LA Follegatti-Romero` ≡ `follegatti romero`, `Bos Van den` ≡ `van
den bos`), ignores spacing (`Vanduyne` ≡ `van duyne`) and allows a tight
similarity on words of 5+ letters for typos (`Stenflos` ≡ `Stenflo`).

Measured on 3,477 real DOIs: **97.9%** of the applicable cases corroborate.
Costs roughly 3× the time and 7× the bandwidth of the plain existence check.

Two extra output files, and one rule that matters:

| File | Contents |
|---|---|
| `corroborated_dois.txt` | DOI on Meta **and** surname agrees — the safest bucket |
| `discordant_authors.tsv` | `doi <TAB> surname in the reference <TAB> surnames on Meta` |

> A discordant surname **never demotes a DOI**. The work is on Meta either way;
> the row is flagged for inspection, not turned into a candidate. The inference
> only runs one way: agreement strengthens a *present*, disagreement never
> produces an *absent*.

The report also separates the two non‑corroborable cases — the work has no
authors recorded on Meta, and the reference has no author to compare — because
they mean different things.

### Batch matching of a whole dump (`script/batch/`)

The tool sends a few SPARQL queries per reference, which is right for a
continuous flow of new references but too slow for a backlog: the June 2026
Crossref dump has 779,472,572 references without a DOI, 466.8 million distinct.
At the tool's best measured speed (0.166 s a reference, local QLever) that is
years. `script/batch/` applies **the same rules** in bulk: Meta's tables are
exported once to Parquet, every query of the tool becomes a join in
[DuckDB](https://duckdb.org/) over a batch of ~200,000 references, and the
candidates are scored with the tool's own `calculate_matching_score` and chosen
as in the tool's loop (query order, stop at 26, threshold adjustment to 23.4).

| | Seconds a reference | June 2026 dump |
|---|---|---|
| Tool, local QLever (20,000 dump references) | 0.166 | years |
| Batch matching, GROBID in blocks of 100 (200,000 references) | 0.0118 | ~2 months |

**Same results as the tool.** On the paper's dataset (`evaluation.py metrics`)
the metrics and Tables 4-7 are identical, with and without DOI queries. On
20,000 dump references, of the tool's 1,397 matches 1,395 are identical, one is
found through another query and one is lost: the index of title words misses a
key inside a word ("Thermo Solutal" against "Thermosolutal" on Meta). Other
known differences, none of which changed a match in those tests: no "without year"
pass (it would only redo `author_title`, which does not filter on the year), page
ranges compared part by part, and the best row taken for each candidate work.

Use the tool for new references as they arrive, and the batch scripts for a
backlog such as a whole dump.

**Pipeline**

```
Crossref dump ── bulk_doi.py --authors ──> matched_dois.txt, unmatched_dois.txt, discordant_authors.tsv,
     │                                     check_errors.txt, corroborated_dois.txt, no_doi_references.jsonl
     │
     ├─ no DOI:   no_doi_references.jsonl ── prepare_dump.py ──> batches ── match_bulk.py --batches ──> main run CSVs
     ├─ DOI to verify: extract_dois_to_verify.py ──> dois_to_verify.jsonl ── prepare_dump.py ──> batches
     │                 ── match_bulk.py --batches --drop-grobid-doi ──> DOI run CSVs
     ├─ (optional) recheck_absent_dois.py ──> repaired DOIs, checked again on Meta
     └─ extract_doi_occurrences.py ──> doi_occurrences.parquet (who cites each verified DOI)
                         all of the above ── merge_citations.py ──> merge_*.parquet, to_check.csv
```

The batch scripts need DuckDB and PyArrow on top of the project's
dependencies: run them with `uv run --with duckdb --with pyarrow`.

**1. Export Meta's tables** from a QLever endpoint holding OpenCitations Meta
(authors, titles, dates, volumes, pages, DOIs; same patterns as the tool's
queries):

```bash
QLEVER=http://localhost:7011 QLEVER_TOKEN=<access token> OUT=meta_tables \
    bash script/batch/export_meta_tables.sh all
```

**2. Bulk DOI check** of the dump with author corroboration
([Bulk DOI mode](#bulk-doi-mode---bulk)): it also writes the references without
a DOI (`no_doi_references.jsonl`).

**3. Deduplicate and split into batches.** Two references are the same if the
matcher sees them as the same (same fields after the tool's normalization, free
text included), so each distinct reference is matched once and its outcome holds
for all its occurrences. The batch of a reference comes from its hash, so every
batch is a random sample.

```bash
uv run --with duckdb --with pyarrow python script/batch/prepare_dump.py \
    bulk_out/no_doi_references.jsonl batches --batches 2300
```

**4. Match the batches.** Each batch writes `batch_KKKK_matches.csv` and
`batch_KKKK_matches_unmatched.csv` (the tool's columns, plus `citing_work` and
`citing_work_doi`; one row per occurrence). A batch with both CSVs is done, so
re-running the same command resumes after an interruption.

```bash
uv run --with duckdb --with pyarrow python script/batch/match_bulk.py batches results \
    --batches --tables meta_tables --grobid-config data/grobid/config.json --grobid-batch 100 \
    --min-candidate-score 21.6 --memory 60GB --threads 16 --processes 16
```

| Option | Default | Meaning |
|---|---|---|
| `--batches` | off | The input is the folder of `prepare_dump.py` (without it: a folder of Crossref JSON files, like the tool's batch mode) |
| `--tables` | required | Folder of Meta's tables |
| `--grobid-config` | none | GROBID on the unmatched references with free text (pass 2), as in the tool; without it, pass 1 only |
| `--grobid-batch` | 0 | Citations per GROBID request (`processCitationList`); 0 = one request a citation, like the tool. 100 cut GROBID's time on 200,000 references from 2,319 to 540 s with the same matches |
| `--grobid-threads` | 8 | Parallel GROBID requests |
| `--min-candidate-score` | 23.4 | The best candidate under the threshold is written to the unmatched CSV with its score; 21.6 makes it possible to apply threshold 24 later, 16.2 any threshold from 18 up (0.9 x the threshold). The accepted matches do not change |
| `--best-candidate` | off | No early stop at the first candidate reaching 26: the candidate with the most points over all the queries wins. On the held-out set it gives exactly the paper's held-out table (at 26: 17 wrong of 1,498 instead of 27; the early stop sometimes picks a Meta duplicate of the right work); no measurable time cost. With `--min-candidate-score 16.2`, one run gives the exact results of every threshold from 18 up |
| `--drop-grobid-doi` | off | Drop a DOI that GROBID finds in the text of a reference given without one: only for the discordant references of the DOI run, whose DOI was removed on purpose |
| `--memory` | 20GB | DuckDB's memory limit. If a pass runs out of memory it is split in half and redone (same outcome: references are independent) |
| `--threads` / `--processes` | 8 / 8 | DuckDB threads / processes computing the scores |
| `--temp` | system temp | DuckDB's temporary files. **Two DuckDB jobs running at the same time need two folders** (a shared one corrupts the files) |

On the June 2026 dump, a batch of ~200,000 distinct references (~310,000
occurrences) takes ~2,400 s with 16 threads and 16 processes.

**5. Verify the references with a DOI** that the bulk check could not settle:
absent (the DOI is not on Meta), discordant (the DOI is on Meta but the surname
differs: the DOI is removed, so the matcher judges on the other fields only) and
error (the check failed). They go through the same batch matching, in their own
folders:

```bash
uv run python script/batch/extract_dois_to_verify.py crossref-data-2026-06.tar.gz bulk_out dois_to_verify.jsonl
uv run --with duckdb --with pyarrow python script/batch/prepare_dump.py dois_to_verify.jsonl doi_batches --batches 80
uv run --with duckdb --with pyarrow python script/batch/match_bulk.py doi_batches doi_results \
    --batches --drop-grobid-doi --tables meta_tables --grobid-config data/grobid/config.json --grobid-batch 100 \
    --min-candidate-score 21.6
```

If the bulk check ran with a `normalize_doi` older than the current one (before
20/10/2026: no repair of Unicode dashes, `%2f`, trailing punctuation, spaces) or
had failed chunks, `recheck_absent_dois.py bulk_out recheck` checks the repaired
DOIs and the errors again on Meta, with the authors.

**6. Occurrences of the verified DOIs.** The bulk check keeps only unique DOIs;
the merge needs who cites them, so the dump is read once more (12 hours for the
2.05 billion occurrences of the June 2026 dump). Use `--legacy-normalization` for
a bulk run made before 20/10/2026.

```bash
uv run --with pyarrow python script/batch/extract_doi_occurrences.py crossref-data-2026-06.tar.gz doi_occurrences.parquet
```

**7. Merge** everything into one row per occurrence, with the cited work and a
reliability level:

```bash
uv run --with duckdb --with pyarrow python script/batch/merge_citations.py \
    dois_to_verify.jsonl doi_results merged --recheck recheck --no-doi results \
    --verified doi_occurrences.parquet --bulk bulk_out --memory 30GB
```

| Level | When |
|---|---|
| `high` | Two independent signals agree: the DOI and the matcher find the same work, or the DOI exists and the surname agrees |
| `medium` | One signal: the DOI exists (also with a disagreeing surname, if the matcher does not contradict it), or only the matcher |
| `to check` | The DOI leads to one work and the matcher to another: the DOI wins, the matcher's work stays in `matcher_doi` / `matcher_omid`; the rows are also written to `to_check.csv` |
| `none` | No link: these go to the triage for ingestion into Meta, printed at the end by best candidate (none; under 21.6, uncertain; from 21.6 up, probable duplicate) |

The output columns are `citing_work`, `citing_work_doi`, `ref_id`,
`category` (`verified`, `absent`, `discordant`, `error` or `no DOI`),
`original_doi`, `repaired_doi`, `cited_doi`, `cited_omid`, `level`, `source`,
`score`, `matcher_doi`, `matcher_omid`, `matcher_query` (the strategy that found
the candidate) and `authors` (the bulk's surname check: `agrees`, `disagrees` or
empty); `source` says which signals made the link (for example
`repaired DOI + surname`). The matcher's threshold is not fixed in the matching:
`--threshold T` counts as a match every candidate from 0.9 × T points up, and it
is exact only if `match_bulk.py` ran with `--min-candidate-score` 0.9 × T or
lower. `--corrections` applies human judgements last, one row per occurrence
(`citing_work`, `ref_id`, `verdict` = `right`, `wrong` or `other work`,
`cited_doi`, `cited_omid`, plus any notes): `right` makes the row `high`,
`wrong` removes the link, `other work` replaces the cited work, and `source`
gets "(confirmed / rejected / corrected by hand)"; the evidence columns are kept,
so the corrections survive every rerun of the merge. Each part is written to its own file (`merge_with_doi.parquet`,
`merge_no_doi.parquet`, `merge_verified.parquet`) to stay within memory on
billions of rows (2.08 billion rows took 16 minutes with 4 threads).

---

## Workflow

### Detailed Processing Workflow

```
┌────────────────────────────────────────────────────────────┐
│ Step 1: INPUT PROCESSING                                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│               Crossref JSON       TEI XML File             │
│                    │                    │                  │
│                    └────────────────────┘                  │
│                               ▼                            │
│                   Parse & Extract References               │
│                               │                            │
│             ┌─────────────────┴─────────────────┐          │
│             ▼                                   ▼          │
│       Crossref Format                      TEI Format      │
│       (JSON structure)                     (biblStruct)    │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ Step 2: REFERENCE NORMALIZATION                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  For each reference:                                       │
│  1. Clean title (remove punctuation, normalize case)       │
│  2. Normalize DOI (strip prefix, lowercase)                │
│  3. Extract authors (parse names, handle formats)          │
│  4. Validate year (check range 1700-current+1)             │
│  5. Normalize Unicode (remove accents, special chars)      │
│  6. Extract volume/page numbers                            │
│                                                            │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ Step 3: QUERY STRATEGY SELECTION                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Query Execution Order (sequential, early stop):           │
│                                                            │
│  1. year_and_doi (if DOI + year available)                 │
│  2. doi_title (if DOI + title available)                   │
│  3. author_title (if author + title available)             │
│  4. year_author_page (if year + author + page available)   │
│  5. year_volume_page (if year + volume + page available)   │
│  6. year_author_volume (if year + author + vol available)  │
│                                                            │
│  Early stop when: score >= threshold                       │
│  Grobid fallback: if initial match fails                   │
│  No-year attempt: if suspiscious year is found             │
│                                                            │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ Step 4: SPARQL QUERY EXECUTION                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  For selected query:                                       │
│  1. Construct SPARQL query with filters                    │
│  2. Apply rate limiting (2.5 req/sec)                      │
│  3. Execute query against OpenCitations endpoint           │
│  4. Handle errors:                                         │
│     - 429 (Rate Limit): Exponential backoff                │
│     - 5xx (Server Error): Retry with delay                 │
│     - Timeout: Retry with extended timeout                 │
│  5. Parse results (extract candidates)                     │
│                                                            │
│  Max retries: 3                                            │
│  Timeout: 600 seconds                                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ Step 5: CANDIDATE SCORING                                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  For each candidate from SPARQL results:                   │
│                                                            │
│  ┌─────────────────────────────────────────┐               │
│  │ DOI Matching (15 points max)            │               │
│  │ - Exact match: +15 pts                  │               │
│  └─────────────────────────────────────────┘               │
│                                                            │
│  ┌─────────────────────────────────────────┐               │
│  │ Title Similarity (14 points max)        │               │
│  │ - 100% match: +14 pts                   │               │
│  │ - 95-99%:     +13 pts                   │               │
│  │ - 90-94%:     +13 pts                   │               │
│  │ - 85-89%:     +12 pts                   │               │
│  │ - 80-84%:     +11 pts                   │               │
│  │ - 75-79%:     +10 pts                   │               │
│  │ - <75%:       +0 pts                    │               │
│  └─────────────────────────────────────────┘               │
│                                                            │
│  ┌─────────────────────────────────────────┐               │
│  │ Author Matching (7 points max)          │               │
│  │ - Surname match: +7 pts (tolerant to    │               │
│  │   accents, particles, word order and    │               │
│  │   small typos; queries also look up the │               │
│  │   capitalised and accent-free forms)    │               │
│  └─────────────────────────────────────────┘               │
│                                                            │
│  ┌─────────────────────────────────────────┐               │
│  │ Year Matching (1 point max)             │               │
│  │ - Exact year: +1 pt                     │               │
│  │ - Adjacent:   +0 pts                    │               │
│  └─────────────────────────────────────────┘               │
│                                                            │
│  ┌─────────────────────────────────────────┐               │
│  │ Volume Matching (3 points max)          │               │
│  │ - Exact match: +3 pts                   │               │
│  └─────────────────────────────────────────┘               │
│                                                            │
│  ┌─────────────────────────────────────────┐               │
│  │ Page Matching (8 points max)            │               │
│  │ - Start page match: +8 pts              │               │
│  └─────────────────────────────────────────┘               │
│                                                            │
│  TOTAL SCORE: Sum of all components (max 48 points)        │
│  THRESHOLD: 26 points (54.5% of maximum)                   │
│  ADJUSTED THRESHOLD: 90% of 26                             │
│                                                            │
│  Select early-winning candidate with score >= 90% of 26    │
│                                                            │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ Step 6: RESULT COMPILATION & OUTPUT                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Generate outputs:                                         │
│                                                            │
│  1. CSV Matched References                                 │
│     - reference_id, article_title                          │
│     - matched_title, score                                 │
│     - matched_doi, meta_id                                 │
│     - query_type                                           │
│                                                            │
│  2. CSV Unmatched References                               │
│     - All reference metadata                               │
│     - Best score achieved                                  │
│     - Score breakdown (original/grobid/no-year)            │
│     - GROBID attempt status                                │
│                                                            │
│  3. Statistics Text File                                   │
│     - Total references, match rate                         │
│     - Field availability stats                             │
│     - Query type distribution                              │
│     - GROBID fallback statistics                           │
│                                                            │
│  4. Log Files (5 specialized logs)                         │
│     - reference_matching_main.log                          │
│     - reference_matching_authors.log                       │
│     - reference_matching_queries.log                       │
│     - reference_matching_scores.log                        │
│     - reference_matching_errors.log                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---
### Scoring Logic Examples

#### Example 1: Perfect Match (48/48 points)

```
Reference Input:
  Title: "Deep Learning in Medical Imaging"
  DOI: 10.1234/example.2020.001
  Authors: ["Smith, J.", "Johnson, M."]
  Year: 2020
  Volume: 15
  Pages: 123-145

OpenCitations Candidate:
  Title: "Deep Learning in Medical Imaging"
  DOI: 10.1234/example.2020.001
  Authors: ["Smith, John", "Johnson, Mary"]
  Year: 2020
  Volume: 15
  Start Page: 123
  End Page: 145

Score Calculation:
  ✓ DOI exact match:        +15 points
  ✓ Title 100% match:       +14 points
  ✓ Author match (Smith):   +7 points
  ✓ Year exact:             +1 point
  ✓ Volume match:           +3 points
  ✓ Page match (123):       +8 points
  ─────────────────────────────────
  TOTAL:                    48 points ✓ MATCH
```

#### Example 2: Strong Match (31/48 points)

```
Reference Input:
  Title: "Machine learning techniques"
  Authors: ["Doe, A."]
  Year: 2019
  Volume: 12
  Pages: 45-67

OpenCitations Candidate:
  Title: "Machine Learning Techniques for Data Analysis"
  Authors: ["Doe, Alice", "Brown, Bob"]
  Year: 2019
  Volume: 12
  Start Page: 45

Score Calculation:
  ✗ DOI not available:      +0 points
  ✓ Title 85% match:        +12 points
  ✓ Author match (Doe):     +7 points
  ✓ Year exact:             +1 point
  ✓ Volume match:           +3 points
  ✓ Page match (45):        +8 points
  ─────────────────────────────────
  TOTAL:                    31 points ✓ MATCH
```

#### Example 3: Weak Match (12/48 points)

```
Reference Input:
  Title: "Quantum computing review"
  Year: 2021

OpenCitations Candidate:
  Title: "A Comprehensive Review of Quantum Computing"
  Year: 2021

Score Calculation:
  ✗ DOI not available:      +0 points
  ✓ Title 80% match:        +11 points
  ✗ No author data:         +0 points
  ✓ Year exact:             +1 point
  ✗ No volume:              +0 points
  ✗ No page:                +0 points
  ─────────────────────────────────
  TOTAL:                    12 points ✗ NO MATCH
```

---

## Output Files

### 1. CSV Results File

Tabular format for matched references:

```csv
reference_id,article_title,matched_title,score,matched_doi,meta_id,query_type
ref_1,"Machine Learning in Healthcare","Machine Learning in Healthcare Applications",35,"10.1234/mlh.2020","https://opencitations.net/meta/br/...",author_title
ref_3,"Neural Networks in Medicine","Neural Networks in Medical Imaging",42,"10.5678/nnm.2021","https://opencitations.net/meta/br/...",doi_title
```

### 2. Unmatched References CSV

References that didn't meet the threshold:

```csv
reference_id,year,volume,first_page,first_author_lastname,article_title,volume_title,journal_title,doi,unstructured,best_score,score_original,score_after_grobid,score_without_year,grobid_attempted,threshold_failed
ref_2,2019,12,45,Doe,"Deep Learning Review",,,10.9999/dlr.2019,,12,12,N/A,N/A,No,Yes
```

### 3. Statistics File

Text file with comprehensive statistics:

```txt
Total references: 25
Matches found: 18 (72.0%)
Errors: 0

References with author: 22/25
References with title: 25/25
References with DOI: 15/25
References with year: 24/25
References with volume: 20/25
References with page: 18/25

Query Type Distribution:
  author_title: 8 (44.4%)
  year_and_doi: 6 (33.3%)
  year_volume_page: 4 (22.2%)

GROBID fallbacks attempted: 3
GROBID successes: 2
```

### 4. HTML Processing Report

HTML report with comprehensive statistics, field contributions, query type distribution, and visualizations (generated as `processing_report.html`).

---

## Logging System

### Multi-File Logging Architecture

The tool uses 5 specialized log files for different aspects:

```
┌─────────────────────────────────────────────────────────────┐
│                      LOG FILES STRUCTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. reference_matching_main.log                             │
│     ├─ All processing events                                │
│     ├─ Initialization messages                              │
│     ├─ Progress updates                                     │
│     └─ General workflow logs                                │
│                                                             │
│  2. reference_matching_authors.log                          │
│     ├─ Author extraction details                            │
│     ├─ Name parsing and normalization                       │
│     ├─ Author matching results                              │
│     └─ Filter: Messages containing "AUTHOR" or "👤"         │ 
│                                                             │
│  3. reference_matching_queries.log                          │
│     ├─ SPARQL query construction                            │
│     ├─ Query execution details                              │
│     ├─ API response summaries                               │
│     └─ Filter: Messages containing "SPARQL", "QUERY",       │ 
│        "🔍", or "🔨"                                       │
│                                                             │
│  4. reference_matching_scores.log                           │
│     ├─ Score calculation details                            │
│     ├─ Field-by-field scoring                               │
│     ├─ Match/no-match decisions                             │
│     └─ Filter: Messages containing "SCORE", "MATCH", "🎯"   │
│                                                             │
│  5. reference_matching_errors.log                           │
│     ├─ All WARNING and ERROR messages                       │
│     ├─ Exception tracebacks                                 │
│     ├─ API failures                                         │
│     └─ Validation errors                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Log Rotation

- Each log file has a maximum size of 10 MB
- Keeps 5 backup files (rotating)
- UTF-8 encoding for international characters
- Automatic timestamp and line number tracking

### Sample Log Entries

```
2025-11-07 14:23:15 - INFO - [match_reference:1234] - Starting match for reference #1
2025-11-07 14:23:15 - DEBUG - [normalize_doi:567] - 🔨 Normalized DOI: 10.1234/example
2025-11-07 14:23:16 - INFO - [execute_sparql_query:890] - 🔍 QUERY: author_title
2025-11-07 14:23:17 - DEBUG - [calculate_score:1112] - 🎯 SCORE: DOI=15, Title=13, Author=7
2025-11-07 14:23:17 - INFO - [match_reference:1245] - ✓ MATCH found with score 35/48
```

---

## Error Handling

```
┌─────────────────────────────────────────────────────────────┐
│                    ERROR HANDLING MATRIX                    │
├──────────────────┬──────────────────────────────────────────┤
│ Error Type       │ Recovery Strategy                        │
├──────────────────┼──────────────────────────────────────────┤
│ Rate Limit (429) │ • Exponential backoff: min(60, 2^n * 5s) │
│                  │ • Reset token bucket to 0                │
│                  │ • Max 3 retries                          │
│                  │ • Log retry attempts and wait time       │
├──────────────────┼──────────────────────────────────────────┤
│ Server Error     │ • Retry with exponential backoff + jitter│
│ (500, 502, 503,  │ • Wait: 2^attempt + random(0, 1) seconds │
│ 504)             │ • Max 3 retries                          │
│                  │ • Log server status and response         │
├──────────────────┼──────────────────────────────────────────┤
│ Timeout          │ • Retry with fixed 2s delay              │
│                  │ • Same timeout value on each retry       │
│                  │ • Max 3 retries                          │
│                  │ • Log timeout occurrence                 │
├──────────────────┼──────────────────────────────────────────┤
│ Network Error    │ • Exponential backoff: 2^attempt seconds │
│ (ClientError)    │ • Max 3 retries                          │
│                  │ • Log network error details              │
│                  │ • Raise QueryExecutionError if persistent│
├──────────────────┼──────────────────────────────────────────┤
│ GROBID Failure   │ • Log extraction failure/error           │
│                  │ • Continue without GROBID enrichment     │
│                  │ • Mark as unmatched if all attempts fail │
├──────────────────┼──────────────────────────────────────────┤
│ JSON Parse Error │ • Try multiple encodings (utf-8, latin-1)│
│ (Input file)     │ • Log encoding and parse errors          │
│                  │ • Raise error if all encodings fail      │
│                  │ • No retry for malformed input files     │
├──────────────────┼──────────────────────────────────────────┤
│ JSON Parse Error │ • Caught by generic Exception handler    │
│ (SPARQL response)│ • Exponential backoff: 2^attempt seconds │
│                  │ • Max 3 retries                          │
│                  │ • Log error type and message             │
└──────────────────┴──────────────────────────────────────────┘
```
---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Low Match Rate

```
Symptom: Match rate < 30%
Possible causes:
  ✗ Incomplete reference metadata
  ✗ Non-standard citation formats
  ✗ References not in OpenCitations
  ✗ PDF extraction errors

Solutions:
  ✓ Check reference quality in input
  ✓ Verify DOIs are correct
  ✓ Enable GROBID fallback
  ✓ Review author extraction logs
  ✓ Try different query strategies
```

#### Issue 2: Rate Limiting Errors

```
Symptom: Frequent 429 errors
Possible causes:
  ✗ Too many concurrent requests
  ✗ Insufficient pause between batches

Solutions:
  ✓ Reduce requests_per_second
  ✓ Increase pause_duration
  ✓ Reduce batch_size
  ✓ Check rate limiting logs
```

#### Issue 3: GROBID Connection Failed

```
Symptom: "Cannot connect to GROBID server"
Possible causes:
  ✗ GROBID server not running
  ✗ Wrong server URL in config
  ✗ Network/firewall issues

Solutions:
  ✓ Start GROBID server: docker run -d -p 8070:8070 grobid/grobid
  ✓ Check grobid_config.json URL
  ✓ Test connection: curl http://localhost:8070/api/isalive
```

#### Issue 4: Encoding Errors

```
Symptom: UnicodeDecodeError or garbled text
Possible causes:
  ✗ Non-UTF-8 input files
  ✗ Special characters in titles

Solutions:
  ✓ Save input CSV as UTF-8
  ✓ Enable text normalization
  ✓ Check log files for details
  ✓ Use unidecode for accents
```

## Performance Tips

### Optimizing Match Rates

1. **Complete Metadata**: Input data with complete metadata have a higher chance to match
2. **Standardized Formats**: Input data that follows standard citation formats have a higher chance to match
3. **Enable GROBID**: Better extraction for difficult documents
4. **Adjust Threshold**: Lower threshold (e.g., 22) for more matches (precision/recall tradeoff)

### Optimizing Speed

1. **Batch Processing**: Default batch_size is 3, can be increased for higher throughput
2. **Concurrent Queries**: Increase max_concurrent_queries (carefully, could actually slow down the process due to multiple errors)
3. **Checkpoint Frequently**: Save progress every 25-50 references
4. **Skip Slow Queries**: Set shorter timeouts for faster queries (precision/recall tradeoff)

---
