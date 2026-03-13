"""
config/settings.py
Central configuration for the Medical RAG pipeline.
All retrieval-quality parameters are documented with rationale.
"""

from pydantic import BaseModel, Field
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# ─── Embedding model ──────────────────────────────────────────────────────────
# BioLORD-2023-C: biomedical-specific, outperforms general models on MedQA/BioASQ
# Fallback: "text-embedding-3-large" (OpenAI) for production scalability
EMBEDDING_MODEL = "FremyCompany/BioLORD-2023-C"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 32          # Balance GPU memory vs throughput

# ─── Chunking strategy ────────────────────────────────────────────────────────
# Retrieval accuracy is highly sensitive to chunk size and overlap.
# 400 tokens  → enough context for a clinical paragraph without diluting relevance.
# 80-token overlap → prevents boundary-cut information loss between adjacent chunks.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
CHUNK_MIN_LENGTH = 100             # Drop chunks shorter than this (noise)

# ─── Hybrid retrieval ─────────────────────────────────────────────────────────
# Dense (vector) search: semantic similarity — good for paraphrase matching
# Sparse (BM25) search: exact keyword matching — good for medical terms/drug names
# Alpha controls the blend: 0.0 = pure BM25, 1.0 = pure dense
RETRIEVAL_ALPHA = 0.65             # Slightly favour dense; tweak per eval metrics
RETRIEVAL_TOP_K = 20               # Retrieve more candidates before reranking
RERANK_TOP_N = 5                   # Final top-N after cross-encoder reranking

# ─── Reranker ────────────────────────────────────────────────────────────────
# Cross-encoder reads query + chunk together → more accurate than bi-encoder alone
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_SCORE_THRESHOLD = 0.1    # Discard chunks below this relevance score

# ─── Vector store ─────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "medical_knowledge"
CHROMA_PERSIST_DIR = str(EMBEDDINGS_DIR / "chroma_db")

# ─── LLM generation ───────────────────────────────────────────────────────────
# Groq inference - fast, free tier at console.groq.com
# Options: "llama-3.3-70b-versatile" (best), "llama-3.1-8b-instant" (fastest)
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1              # Low temp → factual, consistent answers
LLM_MAX_TOKENS = 1024
LLM_CONTEXT_WINDOW = 8000          # Max tokens for retrieved context in prompt

# ─── Hallucination guard ──────────────────────────────────────────────────────
# NLI model checks whether generated claims are entailed by retrieved context.
# Sentences scoring below this threshold are flagged as potentially unsupported.
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_ENTAILMENT_THRESHOLD = 0.5    # Below this → flag claim as unsupported

# ─── Safety classifier ────────────────────────────────────────────────────────
# Patterns that trigger immediate safety responses before any RAG processing
EMERGENCY_PATTERNS = [
    "chest pain", "can't breathe", "cannot breathe", "heart attack",
    "stroke", "unconscious", "not breathing", "overdose", "suicidal",
    "suicide", "severe bleeding", "allergic reaction", "anaphylaxis",
]
HARMFUL_PATTERNS = [
    "how to overdose", "lethal dose", "how to kill", "self harm",
    "how much to take to die",
]
OUT_OF_SCOPE_PATTERNS = [
    "stock price", "weather", "sports", "movie", "recipe",
    "legal advice", "financial advice",
]

# ─── Conversation ─────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 6              # How many prior turns to include in context

# ─── Safety ───────────────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ This information is for educational purposes only and does not constitute "
    "medical advice. Always consult a qualified healthcare professional for diagnosis, "
    "treatment, or any medical decisions."
)
EMERGENCY_MESSAGE = (
    "🚨 If you or someone else is experiencing a medical emergency, "
    "call emergency services (911 in the US) immediately. Do not rely on this tool."
)

# ─── Trusted source domains ───────────────────────────────────────────────────
TRUSTED_SOURCES = [
    "ncbi.nlm.nih.gov",    # PubMed / MEDLINE
    "who.int",              # WHO guidelines
    "cdc.gov",              # CDC fact sheets
    "medlineplus.gov",      # MedlinePlus consumer health
    "mayoclinic.org",       # Mayo Clinic
    "nih.gov",              # NIH publications
]
