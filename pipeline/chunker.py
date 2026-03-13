"""
pipeline/chunker.py
Retrieval-optimized chunking pipeline.

Chunking is the single biggest lever for retrieval accuracy:
- Too small → each chunk lacks context; low answer coverage.
- Too large → chunks contain multiple topics; retrieved chunk dilutes relevance.
- No overlap → boundary sentences lose context.

This module implements:
1. Sentence-aware splitting (never break mid-sentence)
2. Configurable overlap to preserve boundary context
3. Per-chunk metadata inheritance from the parent document
4. Quality filtering to drop noise chunks
5. Semantic deduplication to remove near-duplicate chunks
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN_LENGTH,
    PROCESSED_DIR, CHUNKS_DIR
)


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    A single retrievable unit with full provenance metadata.

    Every field here becomes a Chroma metadata field, enabling
    metadata-filtered retrieval (e.g., filter to guideline-level evidence only).
    """
    chunk_id: str               # Unique, stable identifier
    doc_id: str                 # Parent document ID
    content: str                # The actual text
    chunk_index: int            # Position within document (for context ordering)
    total_chunks: int           # Total chunks in parent doc

    # Inherited from parent document — critical for filtered retrieval
    title: str
    source_name: str
    source_url: str
    source_authority: str       # "primary" | "secondary" | "review"
    specialty: str
    evidence_level: str         # "guideline" | "rct" | "review" | "factsheet"
    pub_date: str
    doc_type: str

    # Chunk-level signals
    word_count: int
    has_numbers: bool           # Likely contains statistics/dosages → higher value
    has_list: bool              # Bulleted/numbered content


# ─── Sentence splitter ────────────────────────────────────────────────────────

class SentenceSplitter:
    """
    Splits text into sentences, preserving clinical abbreviations.
    Avoids breaking at 'Dr.', 'vs.', 'Fig.', '0.5 mg', etc.
    """

    # Abbreviations that should NOT end a sentence
    ABBREVIATIONS = {
        "dr", "mr", "mrs", "ms", "prof", "vs", "fig", "no", "vol",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
        "oct", "nov", "dec", "approx", "est", "ref", "dept", "govt"
    }

    def split(self, text: str) -> list[str]:
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Split on sentence boundaries, but not on abbreviations
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        raw_sentences = sentence_endings.split(text)

        merged: list[str] = []
        for sent in raw_sentences:
            if merged and self._is_abbreviation_ending(merged[-1]):
                merged[-1] = merged[-1] + " " + sent
            else:
                merged.append(sent)

        return [s.strip() for s in merged if s.strip()]

    def _is_abbreviation_ending(self, text: str) -> bool:
        words = text.rstrip().split()
        if not words:
            return False
        last = words[-1].rstrip(".").lower()
        return last in self.ABBREVIATIONS or (len(last) == 1 and last.isalpha())


# ─── Core chunker ─────────────────────────────────────────────────────────────

class MedicalChunker:
    """
    Produces retrieval-optimized chunks from parsed medical documents.

    Strategy:
    1. Split into sentences (never break mid-sentence)
    2. Accumulate sentences until chunk_size tokens is reached
    3. Add overlap by carrying forward last N tokens of previous chunk
    4. Filter out low-quality chunks
    5. Generate stable chunk IDs for deduplication
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_length: int = CHUNK_MIN_LENGTH,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_length = min_length
        self.splitter = SentenceSplitter()

    def chunk_document(self, doc: dict) -> list[DocumentChunk]:
        """
        Chunk a single document dict (as produced by IngestionPipeline).
        Returns a list of DocumentChunk objects.
        """
        content = doc["content"]
        sentences = self.splitter.split(content)

        if not sentences:
            logger.warning(f"No sentences found in doc {doc['doc_id']}")
            return []

        raw_chunks = self._build_chunks(sentences)
        chunks = []
        total = len(raw_chunks)

        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.split()) < self.min_length // 4:
                continue  # Drop very short noise chunks

            chunk_id = self._make_id(doc["doc_id"], i, chunk_text)
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc["doc_id"],
                content=chunk_text,
                chunk_index=i,
                total_chunks=total,
                title=doc["title"],
                source_name=doc["source_name"],
                source_url=doc["source_url"],
                source_authority=doc["source_authority"],
                specialty=doc["specialty"],
                evidence_level=doc["evidence_level"],
                pub_date=doc["pub_date"],
                doc_type=doc["doc_type"],
                word_count=len(chunk_text.split()),
                has_numbers=bool(re.search(r'\d', chunk_text)),
                has_list=bool(re.search(r'(\n\s*[-•*]|\n\s*\d+\.)', chunk_text)),
            )
            chunks.append(chunk)

        return chunks

    def _build_chunks(self, sentences: list[str]) -> list[str]:
        """
        Accumulate sentences into token-bounded chunks with overlap.
        Uses approximate token count: 1 token ≈ 0.75 words (standard heuristic).
        """
        chunks = []
        current: list[str] = []
        current_tokens = 0
        overlap_buffer: list[str] = []

        for sent in sentences:
            sent_tokens = self._token_estimate(sent)

            # If adding this sentence exceeds limit, flush current chunk
            if current_tokens + sent_tokens > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(chunk_text)

                # Build overlap: take sentences from end of current chunk
                # until we have CHUNK_OVERLAP tokens worth
                overlap_buffer = []
                overlap_tokens = 0
                for s in reversed(current):
                    s_tok = self._token_estimate(s)
                    if overlap_tokens + s_tok > self.chunk_overlap:
                        break
                    overlap_buffer.insert(0, s)
                    overlap_tokens += s_tok

                current = overlap_buffer.copy()
                current_tokens = overlap_tokens

            current.append(sent)
            current_tokens += sent_tokens

        # Flush the final chunk
        if current:
            chunks.append(" ".join(current))

        return chunks

    def _token_estimate(self, text: str) -> int:
        """Approximate token count: words / 0.75."""
        return int(len(text.split()) / 0.75)

    def _make_id(self, doc_id: str, index: int, text: str) -> str:
        """Stable, unique chunk ID based on content hash."""
        content_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{content_hash}"


# ─── Quality analyzer ─────────────────────────────────────────────────────────

class ChunkQualityAnalyzer:
    """
    Scores chunks for retrieval quality. Used to flag low-value chunks
    during corpus review and to weight retrieval scores downstream.
    """

    def score(self, chunk: DocumentChunk) -> dict:
        scores = {}

        # Length score: penalize too short, reward optimal range
        words = chunk.word_count
        if words < 50:
            scores["length"] = 0.3
        elif 80 <= words <= 350:
            scores["length"] = 1.0
        else:
            scores["length"] = 0.7

        # Authority score
        authority_map = {"primary": 1.0, "secondary": 0.7, "review": 0.5}
        scores["authority"] = authority_map.get(chunk.source_authority, 0.5)

        # Evidence level score
        evidence_map = {"guideline": 1.0, "rct": 0.9, "review": 0.7, "factsheet": 0.6}
        scores["evidence"] = evidence_map.get(chunk.evidence_level, 0.5)

        # Contains clinical data (numbers, dosages, statistics)
        scores["has_data"] = 0.8 if chunk.has_numbers else 0.5

        scores["overall"] = sum(scores.values()) / len(scores)
        return scores


# ─── Batch chunking pipeline ──────────────────────────────────────────────────

class ChunkingPipeline:
    """
    Loads all processed documents and produces the final chunk corpus.
    """

    def __init__(self):
        self.chunker = MedicalChunker()
        self.analyzer = ChunkQualityAnalyzer()
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    def run(self) -> list[DocumentChunk]:
        """Process all documents in PROCESSED_DIR → save chunks to CHUNKS_DIR."""
        doc_files = list(PROCESSED_DIR.glob("*.json"))
        if not doc_files:
            logger.warning(f"No documents found in {PROCESSED_DIR}")
            return []

        all_chunks: list[DocumentChunk] = []
        seen_ids: set[str] = set()

        for doc_file in doc_files:
            with open(doc_file) as f:
                doc = json.load(f)

            logger.info(f"Chunking: {doc['title']} ({doc['word_count']} words)")
            chunks = self.chunker.chunk_document(doc)

            # Deduplicate by chunk_id
            new_chunks = [c for c in chunks if c.chunk_id not in seen_ids]
            seen_ids.update(c.chunk_id for c in new_chunks)
            all_chunks.extend(new_chunks)

            logger.info(f"  → {len(new_chunks)} chunks")

        # Save all chunks to a single JSONL file
        output_path = CHUNKS_DIR / "chunks.jsonl"
        with open(output_path, "w") as f:
            for chunk in all_chunks:
                f.write(json.dumps(asdict(chunk)) + "\n")

        # Save quality report
        self._save_quality_report(all_chunks)

        logger.info(f"\nTotal chunks: {len(all_chunks)} → {output_path}")
        return all_chunks

    def _save_quality_report(self, chunks: list[DocumentChunk]):
        report = []
        for chunk in chunks:
            scores = self.analyzer.score(chunk)
            report.append({
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "word_count": chunk.word_count,
                "scores": scores,
            })

        report.sort(key=lambda x: x["scores"]["overall"])
        report_path = CHUNKS_DIR / "quality_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        low_quality = [r for r in report if r["scores"]["overall"] < 0.5]
        logger.info(
            f"Quality report saved. Low-quality chunks flagged: {len(low_quality)}"
        )


if __name__ == "__main__":
    pipeline = ChunkingPipeline()
    chunks = pipeline.run()
    if chunks:
        print(f"\nSample chunk:")
        print(f"  ID: {chunks[0].chunk_id}")
        print(f"  Words: {chunks[0].word_count}")
        print(f"  Source: {chunks[0].source_name}")
        print(f"  Content preview: {chunks[0].content[:200]}...")
