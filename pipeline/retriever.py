"""
pipeline/retriever.py
Hybrid retrieval engine — the core of RAG retrieval accuracy.

Three-stage retrieval pipeline (critical for medical QA quality):

Stage 1 — Dense retrieval (vector similarity)
  Finds semantically similar chunks even when the exact medical terms differ.
  "chest discomfort" will find chunks about "myocardial infarction pain".

Stage 2 — Sparse retrieval (BM25)
  Finds exact keyword matches. Critical for drug names, ICD codes, lab values.
  "metformin HCl 500mg" must exactly match — semantic search alone misses it.

Stage 3 — Cross-encoder reranking
  Takes the top-K from stages 1+2, re-scores each (query, chunk) pair jointly.
  Far more accurate than bi-encoder but too slow to run on all chunks.
  This is where most retrieval quality gains come from.

Reciprocal Rank Fusion (RRF) blends dense and sparse rankings before reranking.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    RETRIEVAL_ALPHA, RETRIEVAL_TOP_K, RERANK_TOP_N,
    RERANKER_MODEL, RERANKER_SCORE_THRESHOLD,
    CHROMA_COLLECTION, CHROMA_PERSIST_DIR, CHUNKS_DIR
)


# ─── Result model ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A retrieved and reranked chunk with full provenance."""
    chunk_id: str
    content: str
    rerank_score: float
    dense_rank: Optional[int]
    bm25_rank: Optional[int]
    rrf_score: float

    # Metadata
    title: str
    source_name: str
    source_url: str
    source_authority: str
    specialty: str
    evidence_level: str
    pub_date: str


# ─── RRF fusion ───────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_ids: list[str],
    sparse_ids: list[str],
    alpha: float = RETRIEVAL_ALPHA,
    k: int = 60,
) -> dict[str, float]:
    """
    Combine dense and sparse ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = alpha * 1/(k + dense_rank(d))
                               + (1-alpha) * 1/(k + sparse_rank(d))

    RRF is robust to score-scale differences between dense and sparse results,
    which is why it outperforms naive score combination.

    k=60 is the standard RRF constant (Cormack et al., 2009).
    """
    scores: dict[str, float] = {}

    for rank, doc_id in enumerate(dense_ids, 1):
        scores[doc_id] = scores.get(doc_id, 0) + alpha * (1.0 / (k + rank))

    for rank, doc_id in enumerate(sparse_ids, 1):
        scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1.0 / (k + rank))

    return scores  # Sorted externally by caller


# ─── BM25 index ───────────────────────────────────────────────────────────────

class BM25Index:
    """
    In-memory BM25 index over the chunk corpus.
    Rebuilt from the chunks JSONL on initialization.

    BM25 is critical for medical terms: drug names, anatomical terms,
    procedure codes — where exact match matters more than semantics.
    """

    def __init__(self, chunks_path: Optional[str] = None):
        chunks_path = chunks_path or str(CHUNKS_DIR / "chunks.jsonl")
        self._chunks: list[dict] = []
        self._bm25 = None
        self._load(chunks_path)

    def _load(self, path: str):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank-bm25 not installed. Run: pip install rank-bm25")
            return

        try:
            with open(path) as f:
                self._chunks = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            logger.warning(f"Chunks file not found: {path}. BM25 disabled.")
            return

        tokenized = [c["content"].lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(self._chunks)} chunks.")

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[str]:
        """Returns list of chunk_ids ranked by BM25 score."""
        if self._bm25 is None or not self._chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Sort by score descending, return top_k chunk_ids
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [self._chunks[i]["chunk_id"] for i, _ in ranked if _ > 0]

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        for c in self._chunks:
            if c["chunk_id"] == chunk_id:
                return c
        return None


# ─── Dense retriever ──────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Chroma-backed dense vector retrieval.
    Supports optional metadata filters (specialty, evidence_level, etc.)
    """

    def __init__(self):
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self._collection = client.get_collection(CHROMA_COLLECTION)
        return self._collection

    def search(
        self,
        query_embedding: list[float],
        top_k: int = RETRIEVAL_TOP_K,
        where: Optional[dict] = None,
    ) -> list[str]:
        """
        Returns list of chunk_ids ranked by cosine similarity.

        where: Optional Chroma metadata filter, e.g.:
               {"specialty": "cardiology"}
               {"evidence_level": {"$in": ["guideline", "rct"]}}
        """
        collection = self._get_collection()
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count()),
            "include": ["distances", "metadatas"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        ids = results["ids"][0] if results["ids"] else []
        return ids


# ─── Cross-encoder reranker ───────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks candidate (query, chunk) pairs using a cross-encoder.

    Why this matters for retrieval accuracy:
    Bi-encoders embed query and document separately → efficient but lossy.
    Cross-encoders jointly encode (query, document) → slower but far more accurate.
    By applying cross-encoder only to top-K candidates, we get accuracy without
    the computational cost of running it over the full corpus.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading reranker: {self.model_name}")
                self._model = CrossEncoder(self.model_name, max_length=512)
            except Exception as e:
                logger.warning(f"Reranker load failed: {e}. Reranking disabled.")

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int = RERANK_TOP_N,
    ) -> list[tuple[dict, float]]:
        """
        Score each (query, chunk) pair and return top_n by score.
        Returns list of (chunk_dict, score) tuples.
        """
        self._load()
        if self._model is None or not chunks:
            # Fallback: return chunks as-is with dummy scores
            return [(c, 0.5) for c in chunks[:top_n]]

        pairs = [(query, c["content"]) for c in chunks]
        scores = self._model.predict(pairs).tolist()

        ranked = sorted(
            zip(chunks, scores), key=lambda x: x[1], reverse=True
        )

        # Apply minimum score threshold
        ranked = [(c, s) for c, s in ranked if s >= RERANKER_SCORE_THRESHOLD]
        return ranked[:top_n]


# ─── Hybrid retrieval engine ──────────────────────────────────────────────────

class HybridRetriever:
    """
    Three-stage hybrid retrieval engine:
    Dense → Sparse → RRF fusion → Cross-encoder reranking

    This is the primary interface for the RAG pipeline.
    """

    def __init__(self):
        from pipeline.embedder import BiomedicalEmbedder
        self.embedder = BiomedicalEmbedder()
        self.dense = DenseRetriever()
        self.bm25 = BM25Index()
        self.reranker = CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        top_n: int = RERANK_TOP_N,
        specialty_filter: Optional[str] = None,
        evidence_filter: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Full retrieval pipeline for a single query.

        Args:
            query:            The medical question.
            top_k:            Candidates to retrieve before reranking.
            top_n:            Final results after reranking.
            specialty_filter: Optional specialty to restrict retrieval.
            evidence_filter:  Optional list of evidence levels to restrict.

        Returns:
            List of RetrievedChunk ordered by reranker score (best first).
        """
        # Build optional Chroma filter
        where = self._build_filter(specialty_filter, evidence_filter)

        # Stage 1: Dense retrieval
        query_embedding = self.embedder.embed_single(query)
        dense_ids = self.dense.search(query_embedding, top_k=top_k, where=where)

        # Stage 2: Sparse (BM25) retrieval
        bm25_ids = self.bm25.search(query, top_k=top_k)

        # Stage 3: RRF fusion
        rrf_scores = reciprocal_rank_fusion(dense_ids, bm25_ids, alpha=RETRIEVAL_ALPHA)
        # Sort fused candidates by RRF score
        fused_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        fused_ids = fused_ids[:top_k]

        # Fetch full chunk data for candidates
        candidate_chunks = self._fetch_chunks(fused_ids)

        if not candidate_chunks:
            logger.warning(f"No candidates retrieved for query: {query[:80]}")
            return []

        # Stage 4: Cross-encoder reranking
        ranked = self.reranker.rerank(query, candidate_chunks, top_n=top_n)

        # Build result objects
        results = []
        dense_rank_map = {cid: i for i, cid in enumerate(dense_ids, 1)}
        bm25_rank_map = {cid: i for i, cid in enumerate(bm25_ids, 1)}

        for chunk, score in ranked:
            cid = chunk["chunk_id"]
            results.append(RetrievedChunk(
                chunk_id=cid,
                content=chunk["content"],
                rerank_score=round(score, 4),
                dense_rank=dense_rank_map.get(cid),
                bm25_rank=bm25_rank_map.get(cid),
                rrf_score=round(rrf_scores.get(cid, 0), 6),
                title=chunk.get("title", ""),
                source_name=chunk.get("source_name", ""),
                source_url=chunk.get("source_url", ""),
                source_authority=chunk.get("source_authority", ""),
                specialty=chunk.get("specialty", ""),
                evidence_level=chunk.get("evidence_level", ""),
                pub_date=chunk.get("pub_date", ""),
            ))

        logger.debug(
            f"Retrieved {len(results)} chunks for: '{query[:60]}...' "
            f"(top score: {results[0].rerank_score if results else 'N/A'})"
        )
        return results

    def _build_filter(
        self,
        specialty: Optional[str],
        evidence_levels: Optional[list[str]],
    ) -> Optional[dict]:
        filters = []
        if specialty:
            filters.append({"specialty": specialty})
        if evidence_levels:
            filters.append({"evidence_level": {"$in": evidence_levels}})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    def _fetch_chunks(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch full chunk data for a list of IDs (from BM25 index as source of truth)."""
        result = []
        for cid in chunk_ids:
            chunk = self.bm25.get_chunk(cid)
            if chunk:
                result.append(chunk)
        return result


if __name__ == "__main__":
    # Quick smoke test
    retriever = HybridRetriever()
    results = retriever.retrieve("What are the symptoms of type 2 diabetes?", top_n=3)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r.rerank_score:.3f} | {r.source_name}")
        print(f"     {r.content[:200]}...")
