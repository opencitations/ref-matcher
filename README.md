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
| `PAUSE_DURATION` | `--pause-duration` | `10` | Pause between batches (seconds) |
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
| `--batch-size` | int | Number of files processed simultaneously per batch. Each file gets its own rate limiter, so the **effective request rate ≈ `batch-size × rate-limit`** — lower it (e.g. 1) to reduce server load / 500s | 3 |
| `--pause-duration` | int | Delay in seconds between processing batches to avoid overwhelming the server | 10 |
| `--error-threshold` | int | Maximum number of consecutive server errors (5xx) before stopping batch processing | 10 |
| `--log-level` | str | Verbosity of logging output: DEBUG (detailed), INFO (standard), WARNING, or ERROR (minimal) | INFO |
| `--rate-limit` | float | Maximum SPARQL queries per second (per file) to respect OpenCitations API rate limits | 2.5 |
| `--burst-size` | int | Maximum number of concurrent requests allowed in token bucket before rate limiting kicks in | 10 |
| `--no-threshold-adjustment` | flag | Enforce the raw `--threshold` instead of the adaptive lowering to 90% when a score is close | (adjustment on) |
| `--force-restart` | flag | Batch mode: ignore any existing checkpoint in the output directory and reprocess every file from scratch | False |
| `--dump` | flag | Treat the input as a Crossref **dump** (`{"items":[...]}` JSON file, or a `.tar.gz` of such files) — each work is expanded to the matcher's input format and batch‑processed | False |
| `--limit` | int | With `--dump`, process only the first N works (0 = all). Handy for a quick test on a huge dump | 0 |

> **Checkpointing (batch mode):** progress is saved to `processing_checkpoint.pkl`
> **inside the `--output` directory**, so runs with different `-o` folders track
> their progress independently and can be resumed just by re-running the same
> command. Use `--force-restart` to redo a folder from scratch.

### Processing a Crossref dump (`--dump`)

The public Crossref data dump stores **many** works per file as
`{"items": [ {work}, ... ]}`, which is a different shape from the single‑work
API response (`{"message": {work}}`) the matcher reads. `--dump` bridges this: it
expands each work in the dump into the matcher's input format and runs the normal
batch pipeline (one output CSV per work). It accepts either a single dump
`.json` file or a `.tar.gz` archive of dump files.

```bash
# quick test on just the first work of a dump file
uv run script/ReferenceMatchingTool.py 0.json --dump --limit 1 -o out_dir --use-grobid

# a whole .tar.gz dump (all works)
uv run script/ReferenceMatchingTool.py crossref-data-2026-06.tar.gz --dump -o out_dir --use-grobid
```

`--limit N` caps the number of works (great for smoke‑testing a huge dump before
committing to a full run).

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
│  │ - Any exact surname match: +7 pts       │               │
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
│  │ - Start OR End page match: +8 pts       │               │
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
