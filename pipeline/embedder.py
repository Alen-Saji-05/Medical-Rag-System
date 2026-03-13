"""
pipeline/embedder.py
Embedding generation and vector store construction.

Retrieval accuracy directly depends on:
1. Embedding model quality (biomedical vs. general-purpose)
2. Batching strategy (GPU efficiency without OOM)
3. Metadata storage alongside vectors (for filtered retrieval)
4. Collection structure (allows incremental upserts without full rebuilds)
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from loguru import logger
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    CHROMA_COLLECTION, CHROMA_PERSIST_DIR, CHUNKS_DIR
)


# ─── Embedding model wrapper ──────────────────────────────────────────────────

class BiomedicalEmbedder:
    """
    Wraps a biomedical sentence-transformer model.

    BioLORD-2023-C is trained on biomedical ontologies and literature,
    giving it far stronger performance on clinical/medical queries
    than general-purpose models like all-MiniLM or even ada-002.

    Falls back to a general model if biomedical model is unavailable.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None  # Lazy load

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info("Biomedical embedding model loaded.")
            except Exception as e:
                logger.warning(
                    f"Could not load {self.model_name}: {e}. "
                    "Falling back to all-MiniLM-L6-v2."
                )
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,  # Normalize → cosine similarity = dot product
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ─── Vector store builder ─────────────────────────────────────────────────────

class VectorStoreBuilder:
    """
    Builds and manages the Chroma vector store.

    Design decisions for retrieval accuracy:
    - cosine similarity (l2_normalize=True on embeddings)
    - Full metadata stored per chunk → supports filtered retrieval
    - Upsert semantics: re-running doesn't duplicate chunks
    - Batched inserts to handle large corpora
    """

    METADATA_KEYS = [
        "doc_id", "title", "source_name", "source_url",
        "source_authority", "specialty", "evidence_level",
        "pub_date", "doc_type", "word_count",
        "has_numbers", "has_list", "chunk_index", "total_chunks",
    ]

    def __init__(self):
        self.embedder = BiomedicalEmbedder()
        self._collection = None

    def _get_collection(self):
        """Lazy-init Chroma collection with persistence."""
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self._collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},  # Cosine distance
            )
            logger.info(
                f"Chroma collection '{CHROMA_COLLECTION}' ready. "
                f"Current count: {self._collection.count()}"
            )
        return self._collection

    def build_from_chunks(
        self,
        chunks_path: Optional[str] = None,
        batch_size: int = 64,
    ) -> int:
        """
        Load chunks from JSONL, embed them, and upsert into Chroma.
        Returns the total number of vectors in the store after insertion.
        """
        chunks_path = chunks_path or str(CHUNKS_DIR / "chunks.jsonl")
        chunks = self._load_chunks(chunks_path)

        if not chunks:
            logger.error("No chunks to embed. Run chunking pipeline first.")
            return 0

        collection = self._get_collection()
        existing_ids = set(collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            logger.info("All chunks already in vector store. Nothing to add.")
            return collection.count()

        logger.info(
            f"Embedding {len(new_chunks)} new chunks "
            f"(skipping {len(chunks) - len(new_chunks)} existing)"
        )

        # Process in batches
        for i in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding"):
            batch = new_chunks[i : i + batch_size]
            texts = [c["content"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            metadatas = [self._extract_metadata(c) for c in batch]
            embeddings = self.embedder.embed(texts)

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        final_count = collection.count()
        logger.info(f"Vector store updated. Total vectors: {final_count}")
        return final_count

    def _load_chunks(self, path: str) -> list[dict]:
        chunks = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        except FileNotFoundError:
            logger.error(f"Chunks file not found: {path}")
        return chunks

    def _extract_metadata(self, chunk: dict) -> dict:
        """
        Extract only the fields Chroma accepts as metadata.
        Chroma metadata must be: str, int, float, or bool only.
        """
        meta = {}
        for key in self.METADATA_KEYS:
            val = chunk.get(key, "")
            if isinstance(val, (str, int, float, bool)):
                meta[key] = val
            else:
                meta[key] = str(val)
        return meta

    def collection_stats(self) -> dict:
        """Return summary statistics about the vector store."""
        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            return {"total_chunks": 0}

        all_meta = collection.get(include=["metadatas"])["metadatas"]

        specialties = {}
        sources = {}
        evidence_levels = {}

        for meta in all_meta:
            specialties[meta.get("specialty", "unknown")] = (
                specialties.get(meta.get("specialty", "unknown"), 0) + 1
            )
            sources[meta.get("source_name", "unknown")] = (
                sources.get(meta.get("source_name", "unknown"), 0) + 1
            )
            evidence_levels[meta.get("evidence_level", "unknown")] = (
                evidence_levels.get(meta.get("evidence_level", "unknown"), 0) + 1
            )

        return {
            "total_chunks": count,
            "by_specialty": specialties,
            "by_source": sources,
            "by_evidence_level": evidence_levels,
        }


# ─── End-to-end Phase 1 runner ────────────────────────────────────────────────

def run_phase1():
    """
    Run the full Phase 1 pipeline:
    ingestion → chunking → embedding → vector store
    """
    from pipeline.ingestion import IngestionPipeline, create_sample_manifest
    from pipeline.chunker import ChunkingPipeline

    logger.info("═" * 50)
    logger.info("PHASE 1: Document Pipeline & Knowledge Base")
    logger.info("═" * 50)

    # Step 1: Ingestion
    logger.info("\n[1/3] Ingestion")
    raw_dir = Path(CHUNKS_DIR).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = str(raw_dir / "manifest.json")
    create_sample_manifest(manifest_path)
    ingestor = IngestionPipeline()
    docs = ingestor.ingest_from_manifest(manifest_path)
    logger.info(f"  Ingested: {len(docs)} documents")

    # Step 2: Chunking
    logger.info("\n[2/3] Chunking")
    chunker = ChunkingPipeline()
    chunks = chunker.run()
    logger.info(f"  Created: {len(chunks)} chunks")

    # Step 3: Embedding + vector store
    logger.info("\n[3/3] Embedding & Vector Store")
    builder = VectorStoreBuilder()
    total = builder.build_from_chunks()
    logger.info(f"  Vector store size: {total}")

    # Stats
    stats = builder.collection_stats()
    logger.info("\n📊 Knowledge Base Stats:")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  By specialty: {stats.get('by_specialty', {})}")
    logger.info(f"  By source:    {stats.get('by_source', {})}")

    logger.info("\n✅ Phase 1 complete.")
    return stats


if __name__ == "__main__":
    run_phase1()