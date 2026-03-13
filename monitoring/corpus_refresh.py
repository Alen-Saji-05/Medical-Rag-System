"""
monitoring/corpus_refresh.py
Corpus refresh manager — versioned document updates with change tracking.

Handles the monthly document corpus refresh cycle:
  1. Check which documents need updating (stale by date or URL change)
  2. Re-ingest updated documents
  3. Detect content drift (significant changes to existing chunks)
  4. Re-embed only changed documents (incremental rebuild)
  5. Write a change log for audit trail

Run:
  python monitoring/corpus_refresh.py --check      # show stale documents
  python monitoring/corpus_refresh.py --refresh    # run full refresh
  python monitoring/corpus_refresh.py --status     # show corpus health
"""

import json
import hashlib
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DIR, CHUNKS_DIR


# ── Config ────────────────────────────────────────────────────────────────────
REFRESH_INTERVAL_DAYS = 30     # Flag documents older than this
CHANGE_LOG_PATH = Path(__file__).parent.parent / "logs" / "corpus_changes.jsonl"
VERSION_FILE   = Path(__file__).parent.parent / "logs" / "corpus_version.json"


# ── Document staleness checker ─────────────────────────────────────────────────

class CorpusHealthChecker:
    """Scans the processed document store and reports health metrics."""

    def check(self) -> dict:
        docs = list(PROCESSED_DIR.glob("*.json"))
        if not docs:
            return {"status": "empty", "total": 0}

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=REFRESH_INTERVAL_DAYS)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        stale, fresh, missing_date = [], [], []

        for doc_path in docs:
            with open(doc_path) as f:
                doc = json.load(f)

            ingested_at = doc.get("ingested_at", "")
            pub_date    = doc.get("pub_date", "")

            if not ingested_at:
                missing_date.append(doc.get("doc_id", doc_path.stem))
                continue

            try:
                ingested_dt = datetime.fromisoformat(ingested_at.replace("Z", "+00:00"))
                if ingested_dt < cutoff:
                    stale.append({
                        "doc_id":     doc.get("doc_id"),
                        "title":      doc.get("title"),
                        "source_url": doc.get("source_url"),
                        "ingested_at": ingested_at,
                        "days_old":   (now - ingested_dt).days,
                    })
                else:
                    fresh.append(doc.get("doc_id"))
            except ValueError:
                missing_date.append(doc.get("doc_id"))

        # Chunk stats
        chunk_count = 0
        chunks_file = CHUNKS_DIR / "chunks.jsonl"
        if chunks_file.exists():
            with open(chunks_file) as f:
                chunk_count = sum(1 for line in f if line.strip())

        return {
            "status": "ok",
            "total_documents": len(docs),
            "fresh_documents": len(fresh),
            "stale_documents": len(stale),
            "stale_threshold_days": REFRESH_INTERVAL_DAYS,
            "total_chunks": chunk_count,
            "stale_items": stale,
            "missing_date": missing_date,
            "checked_at": now.isoformat(),
        }

    def print_status(self):
        health = self.check()
        print("\n" + "═" * 55)
        print("  CORPUS HEALTH REPORT")
        print("═" * 55)
        print(f"  Status            : {health['status']}")
        print(f"  Total documents   : {health['total_documents']}")
        print(f"  Fresh (<{REFRESH_INTERVAL_DAYS}d old)   : {health['fresh_documents']}")
        print(f"  Stale (≥{REFRESH_INTERVAL_DAYS}d old)   : {health['stale_documents']}")
        print(f"  Total chunks      : {health['total_chunks']}")
        if health["stale_items"]:
            print(f"\n  Stale documents:")
            for item in health["stale_items"]:
                print(f"    - [{item['days_old']}d] {item['title'][:50]}")
        print("═" * 55 + "\n")
        return health


# ── Content drift detector ─────────────────────────────────────────────────────

class ContentDriftDetector:
    """
    Detects when a re-fetched document has changed significantly from
    the stored version. Uses SHA-256 hash of content for exact detection
    and word-count ratio for approximate change magnitude.
    """

    def compare(self, stored_content: str, new_content: str) -> dict:
        stored_hash = hashlib.sha256(stored_content.encode()).hexdigest()
        new_hash    = hashlib.sha256(new_content.encode()).hexdigest()
        changed = stored_hash != new_hash

        if not changed:
            return {"changed": False, "change_magnitude": 0.0, "hash": new_hash}

        # Estimate change magnitude
        stored_words = len(stored_content.split())
        new_words    = len(new_content.split())
        magnitude = abs(new_words - stored_words) / max(stored_words, 1)

        return {
            "changed": True,
            "change_magnitude": round(magnitude, 3),
            "stored_word_count": stored_words,
            "new_word_count": new_words,
            "stored_hash": stored_hash[:12],
            "new_hash": new_hash[:12],
        }


# ── Refresh runner ─────────────────────────────────────────────────────────────

class CorpusRefreshManager:
    """
    Orchestrates incremental corpus refresh:
    - Re-fetches stale documents
    - Detects content drift
    - Re-embeds only changed documents
    - Logs all changes for audit trail
    """

    def __init__(self):
        self.checker = CorpusHealthChecker()
        self.drift   = ContentDriftDetector()
        CHANGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def run_refresh(self, force: bool = False) -> dict:
        """
        Run the full refresh cycle. Returns a summary of changes made.
        force=True re-ingests all documents regardless of staleness.
        """
        from pipeline.ingestion import DocumentFetcher, HTMLParser, PDFParser

        health = self.checker.check()
        targets = health["stale_items"] if not force else self._all_docs()

        if not targets:
            logger.info("All documents are fresh. No refresh needed.")
            return {"refreshed": 0, "changed": 0, "errors": 0}

        logger.info(f"Refreshing {len(targets)} document(s)...")
        fetcher = DocumentFetcher()
        changed_doc_ids = []
        errors = 0

        for item in targets:
            doc_id  = item["doc_id"]
            url     = item.get("source_url", "")
            title   = item.get("title", "unknown")

            if url.startswith("local://"):
                logger.info(f"  Skipping local PDF: {title}")
                continue

            logger.info(f"  Re-fetching: {title[:50]}")
            new_content = fetcher.fetch(url)
            if not new_content:
                logger.warning(f"  Failed to fetch: {url}")
                errors += 1
                continue

            # Load stored document
            doc_path = PROCESSED_DIR / f"{doc_id}.json"
            if not doc_path.exists():
                logger.warning(f"  Stored doc not found: {doc_id}")
                errors += 1
                continue

            with open(doc_path) as f:
                stored_doc = json.load(f)

            drift = self.drift.compare(stored_doc["content"], new_content)

            if drift["changed"]:
                logger.info(
                    f"  Content changed (magnitude={drift['change_magnitude']:.1%}) — updating"
                )
                stored_doc["content"]     = new_content
                stored_doc["word_count"]  = len(new_content.split())
                stored_doc["ingested_at"] = datetime.now(timezone.utc).isoformat()

                with open(doc_path, "w") as f:
                    json.dump(stored_doc, f, indent=2)

                self._log_change(doc_id, title, drift)
                changed_doc_ids.append(doc_id)
            else:
                logger.info(f"  No content change — updating timestamp only")
                stored_doc["ingested_at"] = datetime.now(timezone.utc).isoformat()
                with open(doc_path, "w") as f:
                    json.dump(stored_doc, f, indent=2)

        # Re-chunk and re-embed only changed documents
        if changed_doc_ids:
            logger.info(f"\nRe-embedding {len(changed_doc_ids)} changed document(s)...")
            self._re_embed(changed_doc_ids)

        self._update_version(len(changed_doc_ids))

        summary = {
            "refreshed": len(targets),
            "changed": len(changed_doc_ids),
            "errors": errors,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"Refresh complete: {summary}")
        return summary

    def _all_docs(self) -> list[dict]:
        docs = []
        for path in PROCESSED_DIR.glob("*.json"):
            with open(path) as f:
                doc = json.load(f)
            docs.append({
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "source_url": doc.get("source_url"),
                "ingested_at": doc.get("ingested_at"),
            })
        return docs

    def _re_embed(self, doc_ids: list[str]):
        """Re-chunk and upsert changed documents into Chroma."""
        try:
            from pipeline.chunker import MedicalChunker
            from pipeline.embedder import VectorStoreBuilder

            chunker = MedicalChunker()
            builder = VectorStoreBuilder()
            new_chunks = []

            for doc_id in doc_ids:
                doc_path = PROCESSED_DIR / f"{doc_id}.json"
                if not doc_path.exists():
                    continue
                with open(doc_path) as f:
                    doc = json.load(f)
                chunks = chunker.chunk_document(doc)
                new_chunks.extend(chunks)
                logger.info(f"  Re-chunked {doc_id}: {len(chunks)} chunks")

            if new_chunks:
                # Write updated chunks to JSONL
                from dataclasses import asdict
                chunks_path = CHUNKS_DIR / "chunks.jsonl"
                with open(chunks_path, "a") as f:
                    for chunk in new_chunks:
                        f.write(json.dumps(asdict(chunk)) + "\n")

                # Upsert into vector store
                builder.build_from_chunks(str(chunks_path))
                logger.info(f"  Vector store updated with {len(new_chunks)} new chunks")

        except Exception as e:
            logger.error(f"Re-embedding failed: {e}")

    def _log_change(self, doc_id: str, title: str, drift: dict):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "doc_id": doc_id,
            "title": title,
            **drift,
        }
        with open(CHANGE_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _update_version(self, changes: int):
        version = {"last_refresh": datetime.now(timezone.utc).isoformat(),
                   "documents_changed": changes}
        with open(VERSION_FILE, "w") as f:
            json.dump(version, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus refresh manager")
    parser.add_argument("--check",   action="store_true", help="Show corpus health")
    parser.add_argument("--refresh", action="store_true", help="Run refresh cycle")
    parser.add_argument("--force",   action="store_true", help="Force refresh all docs")
    parser.add_argument("--status",  action="store_true", help="Show corpus stats")
    args = parser.parse_args()

    if args.check or args.status:
        CorpusHealthChecker().print_status()
    elif args.refresh:
        manager = CorpusRefreshManager()
        summary = manager.run_refresh(force=args.force)
        print(f"\nRefresh summary: {summary}")
    else:
        parser.print_help()
