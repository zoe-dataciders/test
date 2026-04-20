# Biohacking Research Paper Search

A Python package that searches bioRxiv, medRxiv, and arXiv for research papers on topics of interest, ranks them by relevance using hybrid machine learning scoring, and outputs results as CSV, Parquet, or Delta Lake tables.

## Features

- **Multi-source search**: Queries bioRxiv, medRxiv, and arXiv simultaneously
- **Hybrid ranking**: Combines BM25 lexical matching with semantic embeddings (SentenceTransformers)
- **Optional cross-encoder reranking**: Final precision boost with cross-encoder models
- **Batch mode**: Search multiple topics in one run
- **Flexible output**: Save results as CSV, Parquet, Delta Lake, or all formats
- **Azure ML Studio compatible**: Works both locally and in Azure ML compute environments

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `pandas`: Data manipulation
- `requests`: HTTP requests to paper APIs
- `sentence-transformers`: Semantic embeddings and cross-encoder models
- `pyarrow`: Parquet format support
- `deltalake`: Delta Lake format support

## Usage

### Local: Single Topic
```bash
python -m biohacking_research \
  --topics "gene therapy" \
  --from-date 2024-01-01 \
  --to-date 2024-12-31
```

### Local: Multiple Topics
```bash
python -m biohacking_research \
  --topics "gene therapy,CRISPR,longevity" \
  --from-date 2024-01-01 \
  --to-date 2024-12-31 \
  --output-dir ./results \
  --output-format all
```

### Azure ML Studio

In Azure ML Studio, set up a command job with:
- **Code**: Point to the repository root
- **Command**: 
  ```bash
  python -m biohacking_research \
    --topics "gene therapy,CRISPR" \
    --from-date 2024-01-01 \
    --to-date 2024-12-31 \
    --output-dir ${{outputs.results}} \
    --output-format all
  ```

## Command-Line Arguments

### Required
- `--topics`: Comma-separated list of topics, e.g., `"topic1,topic2,topic3"`
- `--from-date`: Start date in YYYY-MM-DD format
- `--to-date`: End date in YYYY-MM-DD format

### Optional
- `--max-results-per-source` (default: 100): Maximum papers per source
- `--bm25-weight` (default: 0.45): Weight for lexical scoring (0-1)
- `--semantic-weight` (default: 0.55): Weight for semantic similarity (0-1)
- `--use-cross-encoder`: Enable cross-encoder reranking (slower but more precise)
- `--output-dir` (default: `outputs`): Directory to save results
- `--output-format` (default: `all`): Output format — `csv`, `parquet`, `delta`, or `all`

## Output Format

Results are saved with this structure:
```
{output-dir}/
├── {topic1}.csv
├── {topic1}.parquet
├── {topic1}/                    # Delta Lake table
│   ├── _delta_log/
│   └── ...
├── {topic2}.csv
├── {topic2}.parquet
├── {topic2}/
│   └── ...
└── batch_search_metadata.json   # Summary metadata
```

Each result file contains columns:
- `name of research paper`: Paper title and link
- `date published`: Publication date (YYYY-MM-DD)
- `relevance`: Relevance score (0-1)

## Project Structure

```
biohacking_research/
├── __init__.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── __main__.py              # Package entry point
    ├── batch_search.py          # Multi-topic batch runner
    ├── paper_search.py          # Core search interface
    ├── models.py                # Data classes
    ├── ranker.py                # Ranking engine (BM25 + embeddings + cross-encoder)
    ├── searcher.py              # API interaction
    └── utils.py                 # Helper functions
```

## How It Works

1. **Fetch**: Queries each source API for papers matching your topic and date range
2. **Deduplicate**: Removes duplicate papers across sources
3. **Score**: 
   - BM25 lexical matching on title and abstract
   - Semantic embeddings using SentenceTransformer
   - Hybrid score = (BM25 weight × BM25 score) + (semantic weight × embedding similarity)
4. **Optionally Rerank**: Cross-encoder model refines top results
5. **Sort**: Returns papers ranked by relevance (highest first)
6. **Save**: Writes output in your chosen format(s)

## Configuration

### Ranking Weights
- `--bm25-weight` and `--semantic-weight` must sum to 1.0 (normalized automatically)
- Default (0.45 BM25, 0.55 semantic) balances keyword matching with semantic understanding
- Adjust based on your use case:
  - Higher BM25: For exact keyword matches (e.g., "CRISPR gene editing")
  - Higher semantic: For conceptual searches (e.g., "longevity mechanisms")

### Cross-Encoder
- Slower (~10x) but more accurate relevance judgment
- Only runs on top 25 results by default
- Useful for high-precision final filtering


