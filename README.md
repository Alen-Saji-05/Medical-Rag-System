# MedQuery — Responsible Medical RAG System

A production-grade Retrieval-Augmented Generation system for grounded medical question-answering. Every answer is sourced from curated, authoritative medical documents, cited, and checked for faithfulness before delivery.

---

## Architecture

```
User Query
    │
    ▼
Pre-Generation Safety Filter ── blocked ──► Crisis / Refusal response
    │ safe
    ▼
Hybrid Retrieval
    ├── Dense (BioLORD-2023-C embeddings + ChromaDB)
    ├── Sparse (BM25 keyword matching)
    └── RRF Fusion (α=0.65) → Cross-Encoder Reranking
    │
    ▼
Grounded Prompt Builder (numbered [SOURCE N] blocks, token budget)
    │
    ▼
LLM Generation (Groq · llama-3.3-70b-versatile)
    │
    ▼
Post-Generation Processing
    ├── Hallucination Guard (NLI entailment check per sentence)
    └── Disclaimer injection
    │
    ▼
Response (answer + citations + faithfulness score)
    │
    ▼
Audit Logger (queries.jsonl · flagged.jsonl · feedback.jsonl)
```

---

## Project Structure

```
medical_rag/
├── config/
│   └── settings.py          All tunable parameters with rationale
│
├── pipeline/
│   ├── ingestion.py         Fetch & parse documents from trusted sources
│   ├── chunker.py           Sentence-aware chunking with overlap
│   ├── embedder.py          BioLORD embeddings + ChromaDB vector store
│   ├── retriever.py         Hybrid dense+sparse+rerank retrieval
│   ├── safety.py            Pre/post generation safety filters
│   ├── prompt_builder.py    Grounded prompt construction with token budget
│   ├── generator.py         MedicalRAGPipeline orchestrator + LLM client
│   └── audit_log.py         Append-only JSONL audit trail + feedback log
│
├── api/
│   └── main.py              FastAPI REST API (serves UI + all endpoints)
│
├── ui/
│   └── index.html           Single-file web UI (disclaimer modal, citations,
│                            faithfulness meter, feedback, specialty filters)
│
├── monitoring/
│   ├── dashboard.py         Terminal + HTML metrics dashboard
│   └── corpus_refresh.py    Incremental document refresh with drift detection
│
├── tests/
│   ├── eval_retrieval.py    Phase 1: hit rate, MRR, precision, coverage
│   ├── test_generation.py   Phase 2: safety, prompt, post-processor unit tests
│   ├── eval_e2e.py          Phase 4: RAGAS-style end-to-end metrics
│   ├── adversarial.py       Phase 4: jailbreak & robustness tests
│   └── run_all.py           Master test runner (all suites)
│
├── logs/                    Runtime output (auto-created)
│   ├── queries.jsonl        Every query + response metadata
│   ├── flagged.jsonl        Human review queue
│   ├── feedback.jsonl       User thumbs-up/down events
│   └── eval_*.json          Saved evaluation reports
│
├── cli.py                   Unified CLI for all commands
├── setup.sh                 One-shot environment setup
└── requirements.txt
```

---

## Quick Start

### 1. Setup

```bash
bash setup.sh
```

Or manually:

```bash
# Fix PyMuPDF conflict (if needed)
pip uninstall frontend -y
pip install -r requirements.txt

# Create directories
mkdir -p data/raw data/processed data/chunks embeddings/chroma_db logs
```

### 2. Set API key

Get a free key at https://console.groq.com

```bash
# Linux / Mac
export GROQ_API_KEY="gsk_..."

# Windows PowerShell
$env:GROQ_API_KEY="gsk_..."
```

### 3. Build the knowledge base

```bash
python pipeline/embedder.py
```

This fetches documents from NIH, CDC, WHO, and Mayo Clinic, chunks them, embeds with BioLORD, and stores in ChromaDB.

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000** — the disclaimer modal loads first, then the full chat UI.

---

## CLI Reference

```bash
# Chat
python cli.py                                 # interactive chat
python cli.py --query "What is diabetes?"     # single question
python cli.py --stream --verbose              # streaming + full metadata

# Evaluation
python cli.py --eval retrieval                # retrieval quality (no API key needed)
python cli.py --eval e2e                      # RAGAS-style end-to-end metrics
python cli.py --eval adversarial              # jailbreak + robustness tests
python cli.py --eval all --quick              # all suites, 3-question subset

# Monitoring
python cli.py --monitor                       # terminal metrics dashboard
python cli.py --monitor --html                # generate logs/dashboard.html
python cli.py --monitor --watch 30            # auto-refresh every 30s

# Corpus
python cli.py --corpus-status                 # show stale documents
python cli.py --corpus-refresh                # re-fetch and re-embed stale docs
python cli.py --corpus-refresh --force        # force-refresh all documents
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the web UI |
| `GET` | `/health` | Health check + pipeline status |
| `POST` | `/ask` | Single question → grounded answer |
| `POST` | `/chat` | Multi-turn conversation |
| `GET` | `/sources` | Knowledge base statistics |
| `POST` | `/feedback` | Submit thumbs-up/down on a response |
| `GET` | `/admin/stats` | Aggregate metrics (volume, faithfulness, latency) |
| `GET` | `/admin/review` | Human review queue (flagged responses) |

Interactive API docs: http://localhost:8000/docs

---

## Configuration (config/settings.py)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `EMBEDDING_MODEL` | `BioLORD-2023-C` | Biomedical sentence embedding |
| `CHUNK_SIZE` | `400` | Tokens per chunk — larger = more context, lower precision |
| `CHUNK_OVERLAP` | `80` | Overlap — higher = fewer boundary losses |
| `RETRIEVAL_ALPHA` | `0.65` | Dense/sparse blend — higher = more semantic |
| `RETRIEVAL_TOP_K` | `20` | Candidates before reranking — higher = better recall |
| `RERANK_TOP_N` | `5` | Final chunks to LLM — lower = higher precision |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model — swap for `llama-3.1-8b-instant` for speed |
| `LLM_TEMPERATURE` | `0.1` | Low = factual/consistent |
| `NLI_ENTAILMENT_THRESHOLD` | `0.5` | Below this → sentence flagged as unsupported |

---

## Evaluation Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Hit Rate @5 | ≥ 90% | Relevant chunk in top-5 results |
| MRR | ≥ 0.70 | Mean reciprocal rank of first relevant result |
| Faithfulness | ≥ 70% | NLI-verified claim support |
| Answer Relevancy | ≥ 55% | Answer addresses the question |
| Context Recall | ≥ 60% | Retrieved context covers necessary info |
| Adversarial Pass Rate | ≥ 85% | Safety filter robustness |

---

## Trusted Sources

| Domain | Authority |
|--------|-----------|
| `ncbi.nlm.nih.gov` | PubMed / MEDLINE |
| `who.int` | WHO guidelines |
| `cdc.gov` | CDC fact sheets |
| `medlineplus.gov` | NIH consumer health |
| `mayoclinic.org` | Mayo Clinic clinical overviews |
| `nih.gov` | NIH publications |

---

## Safety Design

**Pre-generation filters** (pattern-based, zero latency):
- Emergency keywords → immediate 911 response, no RAG
- Harmful intent → refusal + crisis hotline (988)
- Out-of-scope → polite redirect

**Post-generation checks**:
- NLI entailment guard — every sentence checked against retrieved context
- Disclaimer injected on every response without exception
- Low faithfulness + flagged sentences → routed to human review queue

**Adversarial resistance** (tested):
- Prompt injection / jailbreak attempts
- Lethal dose / harmful framing queries
- Leading questions (e.g. vaccine misinformation)
- Hallucination bait (fictional drugs / conditions)
- Out-of-scope queries

---

## Corpus Refresh

Documents are flagged stale after 30 days. The refresh cycle:
1. Re-fetches the source URL
2. Detects content drift via SHA-256 hash + word-count ratio
3. Re-chunks and upserts only changed documents (incremental)
4. Logs all changes to `logs/corpus_changes.jsonl`

```bash
python cli.py --corpus-refresh
```

---

## LLM Options (Groq)

| Model | Speed | Quality | Context |
|-------|-------|---------|---------|
| `llama-3.3-70b-versatile` | Medium | Best | 128k |
| `llama-3.1-8b-instant` | Fastest | Good | 128k |
| `mixtral-8x7b-32768` | Medium | Good | 32k |

Change in `config/settings.py` → `LLM_MODEL`.
