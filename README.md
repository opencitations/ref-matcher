# Reference Matching Tool

Repository for a bibliographic reference matching tool designed to identify and align references between [Crossref](https://www.crossref.org/) and [OpenCitations Meta](https://opencitations.net/meta). It implements a heuristic-based approach, enabling the retrieval and validation of bibliographic metadata even in cases of incomplete or inconsistent citation records and generates comprehensive reports with detailed statistics.

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

- **Multi-source Reference Extraction**: Extracts references using Crossref API and GROBID fallback
- **Intelligent SPARQL Matching**: Uses multiple query strategies to find matches in OpenCitations
- **Sophisticated Scoring System**: Weighted scoring based on DOI, title, authors, year, volume, and pages
- **Comprehensive Logging**: Multi-file logging system with specialized logs for queries, authors, scores, and errors
- **Batch Processing**: Process multiple references with checkpointing and error recovery
- **HTML Reports**: Beautiful, interactive HTML reports with detailed statistics
- **Rate Limiting**: Built-in rate limiting to respect API constraints
- **Concurrent Processing**: Async operations for improved performance

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
         ┌──────────────┐    ┌──────────────┐
         │  Crossref    │    │    GROBID    │
         │     API      │    │   Fallback   │
         └──────────────┘    └──────────────┘
                    │                │
                    └────────┬───────┘
                             ▼
        ┌─────────────────────────────────────────┐
        │     2. Reference Normalization Phase    │
        │  - Clean titles, authors, DOIs          │
        │  - Normalize text (Unicode, accents)    │
        │  - Extract numeric fields (year, pages) │
        └─────────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────┐
        │    3. SPARQL Query Construction Phase   │
        │                                         │
        │  Strategy Selection (in order):         │
        │  ├─ DOI + Title (if DOI available)      │
        │  ├─ Year + DOI (if both available)      │
        │  ├─ Author + Title (primary)            │
        │  ├─ Year + Author + Page                │
        │  ├─ Year + Volume + Page                │
        │  └─ Year + Author + Volume              │
        └─────────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────┐
        │   4. OpenCitations SPARQL Query Phase   │
        │  - Execute queries with retry logic     │
        │  - Rate limiting (2.5 req/sec)          │
        │  - Error handling (429, 5xx)            │
        └─────────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────┐
        │      5. Candidate Scoring Phase         │
        │                                         │
        │  Scoring Components:                    │
        │  ├─ DOI Exact Match: 15 pts             │
        │  ├─ Title Similarity: 10-14 pts         │
        │  ├─ Author Match: 7 pts                 │
        │  ├─ Year Match: 1 pt                    │
        │  ├─ Volume Match: 3 pts                 │
        │  └─ Page Match: 8 pts                   │
        │                                         │
        │  Threshold: 26/48 points (54%)          │
        └─────────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────┐
        │    6. Result Generation & Export Phase  │
        │  - JSON results with match details      │
        │  - CSV summary reports                  │
        │  - HTML interactive dashboard           │
        │  - Detailed log files                   │
        └─────────────────────────────────────────┘
```

---

## Installation

### Requirements

- Python 3.8+
- GROBID server (optional, for PDF processing fallback)
- Internet connection (for OpenCitations SPARQL endpoint)

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### MatcherConfig Class

The tool uses a configuration class with sensible defaults:

```python
@dataclass
class MatcherConfig:
    # Timeouts and retries
    default_timeout: int = 600
    max_retries: int = 3
    
    # Year validation
    min_year: int = 1700
    max_year: int = current_year + 1
    
    # Scoring weights
    doi_exact_score: int = 15
    author_exact_match_score: int = 7
    title_exact_score: int = 14
    title_95_score: int = 13
    title_90_score: int = 13
    title_85_score: int = 12
    title_80_score: int = 11
    title_75_score: int = 10
    year_exact_score: int = 1
    volume_match_score: int = 3
    page_match_score: int = 8
    
    # Matching threshold
    matching_threshold: int = 26  # out of 48 max points
    
    # Rate limiting
    requests_per_second: float = 2.5
    burst_size: int = 10
    
    # Batch processing
    default_batch_size: int = 3
    checkpoint_interval: int = 10
```

### GROBID Configuration

Create a `grobid_config.json` file:

```json
{
  "grobid_server": "http://localhost:8070",
  "batch_size": 1000,
  "sleep_time": 5,
  "timeout": 60,
  "coordinates": ["persName"]
}
```

---

## Usage

### Basic Usage

#### Single Reference Matching

```python
from ReferenceMatchingToolBackupMod import ReferenceMatchingTool

# Initialize tool
tool = ReferenceMatchingTool()

# Match a single reference
reference = {
    "title": "Machine learning in bioinformatics",
    "year": "2020",
    "authors": ["Smith, J.", "Doe, A."],
    "doi": "10.1234/example"
}

result = tool.match_reference(reference)
print(f"Match found: {result['match_found']}")
print(f"Score: {result['match_score']}")
print(f"OpenCitations URI: {result['opencitations_uri']}")
```

#### Batch Processing from CSV

```python
# Process references from CSV file
tool.process_references_from_csv(
    input_csv="references.csv",
    output_json="results.json"
)
```

#### Process PDF with DOI

```python
# Extract and match references from a PDF
results = tool.match_references_from_pdf(
    doi="10.1234/article.doi",
    output_prefix="my_paper"
)
```

### Command Line Interface

```bash
# Process a single PDF by DOI
python ReferenceMatchingToolBackupMod.py \
    --doi "10.1234/article.doi" \
    --output-prefix "results"

# Process multiple PDFs from a directory
python ReferenceMatchingToolBackupMod.py \
    --pdf-dir "/path/to/pdfs" \
    --output-dir "/path/to/output"

# Custom batch size and pause
python ReferenceMatchingToolBackupMod.py \
    --doi "10.1234/article.doi" \
    --batch-size 5 \
    --pause-duration 15
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--doi` | DOI of the article to process | None |
| `--pdf-path` | Direct path to PDF file | None |
| `--pdf-dir` | Directory containing PDFs | None |
| `--output-prefix` | Prefix for output files | "reference_matching" |
| `--output-dir` | Directory for output files | Current directory |
| `--batch-size` | Number of refs per batch | 3 |
| `--pause-duration` | Seconds to pause between batches | 10 |
| `--use-grobid-fallback` | Enable GROBID fallback | True |
| `--grobid-config` | Path to GROBID config | "grobid_config.json" |

---

## Workflow

### Detailed Processing Workflow

```
┌────────────────────────────────────────────────────────────┐
│ Step 1: INPUT PROCESSING                                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  CSV Input           PDF Input (DOI)      PDF Input (File) │
│      │                    │                      │         │
│      └────────────────────┼──────────────────────┘         │
│                           ▼                                │
│              Parse & Extract References                    │
│                           │                                │
│         ┌─────────────────┼─────────────────┐              │
│         ▼                 ▼                 ▼              │
│   Via Crossref      Via GROBID        Manual CSV           │ 
│   (Primary)         (Fallback)        (Direct)             │
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
│  Decision Tree:                                            │
│                                                            │
│  Has DOI + Title? ──Yes──> Use: DOI_TITLE query            │
│       │                                                    │
│       No                                                   │
│       │                                                    │
│  Has Year + DOI? ──Yes──> Use: YEAR_AND_DOI query          │
│       │                                                    │
│       No                                                   │
│       │                                                    │
│  Has Author + Title? ──Yes──> Use: AUTHOR_TITLE query      │
│       │                                                    │
│       No                                                   │
│       │                                                    │
│  Has Year + Author + Page? ──Yes──> Use: Y_A_P query       │
│       │                                                    │
│       No                                                   │
│       │                                                    │
│  Has Year + Volume + Page? ──Yes──> Use: Y_V_P query       │
│       │                                                    │
│       No                                                   │
│       │                                                    │
│  Has Year + Author + Vol? ──Yes──> Use: Y_A_V query        │
│       │                                                    │
│       No ──────> SKIP (insufficient metadata)              │
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
│  THRESHOLD: 26 points (54% of maximum)                     │
│                                                            │
│  Select candidate with highest score >= 26                 │
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
│  1. JSON Results File                                      │
│     - All reference details                                │
│     - Match scores and URIs                                │
│     - Query types used                                     │
│     - Timestamp and metadata                               │
│                                                            │
│  2. CSV Summary File                                       │
│     - Reference ID, Title                                  │
│     - Match Found (Yes/No)                                 │
│     - Match Score                                          │
│     - OpenCitations URI                                    │
│     - Query Type                                           │
│                                                            │
│  3. HTML Report                                            │
│     - Interactive dashboard                                │
│     - Statistics and charts                                │
│     - Field contribution analysis                          │
│     - Links to log files                                   │
│                                                            │
│  4. Log Files (5 specialized logs)                         │
│     - Main processing log                                  │
│     - Author extraction log                                │
│     - SPARQL query log                                     │
│     - Score calculation log                                │
│     - Error log                                            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Scoring System

### Scoring Components (Maximum: 48 points)

The scoring system is designed to balance multiple metadata fields:

```
┌─────────────────────────────────────────────────────────┐
│                    SCORING BREAKDOWN                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Component          │ Max Points │ Weight │ Description │
│ ────────────────────┼────────────┼────────┼─────────────│
│  DOI Match          │     15     │  31.2% │ Strongest   │
│  Title Similarity   │     14     │  29.2% │ Very Strong │
│  Page Match         │      8     │  16.7% │ Strong      │
│  Author Match       │      7     │  14.6% │ Moderate    │
│  Volume Match       │      3     │   6.2% │ Weak        │
│  Year Match         │      1     │   2.1% │ Very Weak   │
│ ────────────────────┴────────────┴────────┴─────────────│
│  TOTAL              │     48     │ 100.0% │             │
│                                                         │
│  THRESHOLD: 26 points (54.2% of maximum)                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

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

### 1. JSON Results File

Complete matching results with all metadata:

```json
{
  "metadata": {
    "total_references": 25,
    "matched": 18,
    "unmatched": 7,
    "match_rate": 72.0,
    "processing_time": "00:05:32"
  },
  "references": [
    {
      "ref_id": 1,
      "original_title": "Machine Learning in Healthcare",
      "normalized_title": "machine learning in healthcare",
      "doi": "10.1234/mlh.2020",
      "year": 2020,
      "authors": ["Smith, J.", "Doe, A."],
      "match_found": true,
      "match_score": 35,
      "opencitations_uri": "https://opencitations.net/id/...",
      "query_type": "author_title",
      "matched_candidate": {
        "title": "Machine Learning in Healthcare Applications",
        "doi": "10.1234/mlh.2020",
        "authors": ["Smith, John", "Doe, Alice"],
        "year": 2020,
        "volume": "15",
        "pages": "123-145"
      },
      "score_breakdown": {
        "doi_score": 15,
        "title_score": 13,
        "author_score": 7,
        "year_score": 1,
        "volume_score": 0,
        "page_score": 0
      }
    }
  ]
}
```

### 2. CSV Summary File

Tabular format for easy analysis:

```csv
ref_id,title,match_found,match_score,opencitations_uri,query_type
1,"Machine Learning in Healthcare",Yes,35,"https://opencitations.net/id/...",author_title
2,"Deep Learning Review",No,0,"",""
3,"Neural Networks in Medicine",Yes,42,"https://opencitations.net/id/...",doi_title
```

### 3. HTML Report

Interactive dashboard with:
- **Overview Statistics**: Match rate, total references, processing time
- **Query Type Breakdown**: Which query strategies were used
- **Field Contribution Analysis**: How each metadata field contributed to matches
- **Match Score Distribution**: Histogram of score distribution
- **Author Statistics**: Author extraction and matching success
- **Volume/Page Statistics**: Availability and match rates
- **GROBID Fallback Stats**: Success rate of GROBID processing
- **Log File Links**: Quick access to specialized logs

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

### Error Types and Recovery Strategies

```
┌─────────────────────────────────────────────────────────────┐
│                    ERROR HANDLING MATRIX                    │
├──────────────────┬──────────────────────────────────────────┤
│ Error Type       │ Recovery Strategy                        │
├──────────────────┼──────────────────────────────────────────┤
│ Rate Limit (429) │ • Exponential backoff                    │
│                  │ • Wait time: 2^attempt seconds           │
│                  │ • Max 3 retries                          │
│                  │ • Log retry attempts                     │
├──────────────────┼──────────────────────────────────────────┤
│ Server Error     │ • Retry with delay                       │
│ (500, 502, 503)  │ • Increase timeout                       │
│                  │ • Max 3 retries                          │
│                  │ • Log server response                    │
├──────────────────┼──────────────────────────────────────────┤
│ Timeout          │ • Extend timeout by 50%                  │
│                  │ • Retry with new timeout                 │
│                  │ • Log timeout duration                   │
├──────────────────┼──────────────────────────────────────────┤
│ Network Error    │ • Check connection                       │
│                  │ • Retry after 5 seconds                  │
│                  │ • Log network state                      │
├──────────────────┼──────────────────────────────────────────┤
│ Invalid Data     │ • Skip reference                         │
│                  │ • Log validation error                   │
│                  │ • Continue with next                     │
├──────────────────┼──────────────────────────────────────────┤
│ GROBID Failure   │ • Log extraction failure                 │
│                  │ • Mark as unmatched                      │
│                  │ • Continue processing                    │
├──────────────────┼──────────────────────────────────────────┤
│ JSON Parse Error │ • Log malformed response                 │
│                  │ • Retry query                            │
│                  │ • Skip if persistent                     │
└──────────────────┴──────────────────────────────────────────┘
```

### Custom Exceptions

```python
ReferenceMatchingError       # Base exception
├─ QueryExecutionError       # SPARQL query failures
│  ├─ RateLimitError         # 429 responses
│  └─ ServerError            # 5xx responses
└─ ValidationError           # Data validation errors
```

---

## Advanced Features

### 1. Batch Processing with Checkpoints

Process large sets of references with automatic progress saving:

```python
tool.process_references_from_csv(
    input_csv="large_dataset.csv",
    output_json="results.json",
    batch_size=10,              # Process 10 refs at a time
    pause_duration=15,          # Pause 15s between batches
    checkpoint_interval=50      # Save progress every 50 refs
)
```

**Checkpoint Recovery:**
If processing is interrupted, the tool automatically resumes from the last checkpoint.

### 2. Rate Limiting

Prevents overwhelming the OpenCitations API:

```python
# Token bucket algorithm
Requests per second: 2.5
Burst capacity: 10 requests
Refill rate: 1 token per 0.4 seconds
```

### 3. Concurrent Processing

Uses async operations for improved performance:

```python
# Concurrent SPARQL queries
max_concurrent_queries: 5
timeout_per_query: 600 seconds
connection_pool_size: 10
```

### 4. GROBID Fallback

Automatically uses GROBID if Crossref fails:

```python
# Fallback chain
1. Try Crossref API (fast, structured)
   ↓ (if fails)
2. Try GROBID server (slower, PDF parsing)
   ↓ (if fails)
3. Mark as extraction failure
```

### 5. Text Normalization Pipeline

Sophisticated text cleaning for better matching:

```python
# Normalization steps
1. Unicode normalization (NFD)
2. Accent removal (unidecode)
3. Lowercase conversion
4. Punctuation removal
5. Whitespace normalization
6. HTML entity decoding
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
  ✓ Disable GROBID: --use-grobid-fallback false
```

#### Issue 4: Memory Issues with Large Datasets

```
Symptom: Out of memory errors
Possible causes:
  ✗ Processing too many refs at once
  ✗ Large PDF files

Solutions:
  ✓ Reduce batch_size to 1-3
  ✓ Increase checkpoint_interval
  ✓ Process PDFs separately
  ✓ Split large CSV files
```

#### Issue 5: Encoding Errors

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

1. **Provide Complete Metadata**: Include DOI, authors, year, volume, and pages
2. **Use Standardized Formats**: Follow standard citation formats
3. **Clean Input Data**: Remove formatting artifacts before processing
4. **Enable GROBID**: Better PDF extraction for difficult documents
5. **Adjust Threshold**: Lower threshold (e.g., 22) for more matches (precision/recall tradeoff)

### Optimizing Speed

1. **Batch Processing**: Use batch_size=5-10 for optimal throughput
2. **Concurrent Queries**: Increase max_concurrent_queries (carefully)
3. **Local GROBID**: Run GROBID server locally for faster PDF processing
4. **Checkpoint Frequently**: Save progress every 25-50 references
5. **Skip Slow Queries**: Set shorter timeouts for faster queries

---

## 📄 License

This tool is provided as-is for academic and research purposes.

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional query strategies
- Machine learning-based scoring
- Support for additional databases
- Parallel processing optimization
- UI/dashboard improvements

---

## Support

For issues, questions, or feature requests:
1. Review log files for error details
2. Consult OpenCitations documentation
3. Raise an issue with detailed logs

---

## Version History

### Current Version
- Multi-file logging system
- DOI-based scoring (15 points)
- Enhanced HTML reports
- Async query execution
- Improved error handling
- GROBID fallback support
- Checkpoint recovery

---

**Happy Matching! 🎯**
