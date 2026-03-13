# Medical RAG System — Phase 1: Document Pipeline & Knowledge Base

## Architecture

```
User Query
    │
    ▼
Safety Pre-Filter ──── blocked ──► Block + Disclaimer
    │
    ▼
Query Processor (embed + classify)
    │
    ├─── Dense Retrieval (BioLORD embeddings + Chroma)
    │
    ├─── Sparse Retrieval (BM25)
    │
    ▼
RRF Fusion (α=0.65 dense / 0.35 sparse)
    │
    ▼
Cross-Encoder Reranking (ms-marco-MiniLM)
    │
    ▼
LLM Generation (GPT-4o, grounded on retrieved context)
    │
    ▼
Post-Processing (disclaimer injection, hallucination check)
    │
    ▼
Final Response (answer + citations + disclaimer)
```

## Phase 1 Files

| File | Purpose |
|---|---|
| `config/settings.py` | All tunable parameters with rationale |
| `pipeline/ingestion.py` | Fetch & parse trusted medical documents |
| `pipeline/chunker.py` | Sentence-aware chunking with overlap |
| `pipeline/embedder.py` | BioLORD embeddings + Chroma vector store |
| `pipeline/retriever.py` | Hybrid dense+sparse+rerank retrieval |
| `tests/eval_retrieval.py` | Retrieval quality evaluation suite |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full Phase 1 pipeline
python pipeline/embedder.py   # runs ingestion → chunking → embedding

# Evaluate retrieval quality
python tests/eval_retrieval.py
```

## Key Retrieval Parameters (config/settings.py)

| Parameter | Default | Effect on retrieval |
|---|---|---|
| `CHUNK_SIZE` | 400 tokens | Larger = more context per chunk, lower precision |
| `CHUNK_OVERLAP` | 80 tokens | Higher = fewer boundary information losses |
| `RETRIEVAL_ALPHA` | 0.65 | Higher = more semantic, lower = more keyword |
| `RETRIEVAL_TOP_K` | 20 | More candidates before reranking = higher recall |
| `RERANK_TOP_N` | 5 | Final chunks passed to LLM |

## Evaluation Targets

| Metric | Target |
|---|---|
| Hit Rate @5 | > 90% |
| MRR | > 0.70 |
| Precision @3 | > 60% |
| Keyword Coverage | > 75% |

## Document Sources

- **MedlinePlus (NIH)** — Consumer health fact sheets
- **CDC** — Disease fact sheets and statistics
- **WHO** — International health guidelines
- **PubMed abstracts** — Peer-reviewed research (via NCBI API)
- **Mayo Clinic** — Clinical overviews

All sources are validated against `TRUSTED_SOURCES` in settings.py before ingestion.
