"""
pipeline/audit_log.py
Audit logger — persists every query, response, safety flag, and feedback event.

Why this matters for Phase 3:
- Regulatory / ethical accountability: every AI-assisted health response is logged
- Human review queue: flagged and low-faithfulness responses surface automatically
- Feedback loop: thumbs-up/down data drives future corpus and prompt improvements
- Transparency: users and administrators can inspect what the system said and why

Storage: append-only JSONL files — simple, grep-able, and easy to ship to any
downstream system (BigQuery, Elasticsearch, S3) without schema migration.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))

# ── Storage paths ─────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
QUERY_LOG   = LOG_DIR / "queries.jsonl"
FLAGGED_LOG = LOG_DIR / "flagged.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class QueryLogEntry:
    """One entry per RAG pipeline invocation."""
    event_id: str
    timestamp: str
    query: str
    safety_category: str       # safe / emergency / harmful / out_of_scope
    context_chunks_used: int
    context_truncated: bool
    faithfulness: float
    has_warnings: bool
    flagged_sentences: list[str]
    latency_ms: int
    model_used: str
    citations_count: int
    session_id: Optional[str] = None


@dataclass
class FeedbackEntry:
    """User thumbs-up / thumbs-down on a specific response."""
    event_id: str
    timestamp: str
    query_event_id: str        # Links back to QueryLogEntry
    feedback_type: str         # "positive" | "negative"
    query_preview: str         # First 100 chars of query for context
    session_id: Optional[str] = None
    comment: Optional[str] = None


# ── Logger ─────────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Thread-safe append-only audit logger.
    All writes are atomic single-line JSONL appends.
    """

    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Query logging ──────────────────────────────────────────────────────

    def log_query(
        self,
        query: str,
        safety_category: str,
        context_chunks_used: int,
        context_truncated: bool,
        faithfulness: float,
        has_warnings: bool,
        flagged_sentences: list[str],
        latency_ms: int,
        model_used: str,
        citations_count: int,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Log a completed pipeline invocation.
        Returns the event_id for downstream linking (e.g. feedback).
        """
        event_id = str(uuid.uuid4())
        entry = QueryLogEntry(
            event_id=event_id,
            timestamp=_now(),
            query=query,
            safety_category=safety_category,
            context_chunks_used=context_chunks_used,
            context_truncated=context_truncated,
            faithfulness=faithfulness,
            has_warnings=has_warnings,
            flagged_sentences=flagged_sentences,
            latency_ms=latency_ms,
            model_used=model_used,
            citations_count=citations_count,
            session_id=session_id,
        )
        _append(QUERY_LOG, asdict(entry))

        # Mirror to flagged log if the response needs human review
        if has_warnings or safety_category != "safe" or faithfulness < 0.6:
            self._log_flagged(entry)

        return event_id

    def _log_flagged(self, entry: QueryLogEntry):
        """Write to the human review queue."""
        record = asdict(entry)
        record["review_reason"] = self._review_reason(entry)
        _append(FLAGGED_LOG, record)
        logger.warning(
            f"Query flagged for review | "
            f"reason={record['review_reason']} | "
            f"faithfulness={entry.faithfulness:.2f} | "
            f"category={entry.safety_category}"
        )

    def _review_reason(self, entry: QueryLogEntry) -> str:
        reasons = []
        if entry.safety_category != "safe":
            reasons.append(f"safety:{entry.safety_category}")
        if entry.faithfulness < 0.6:
            reasons.append(f"low_faithfulness:{entry.faithfulness:.2f}")
        if entry.flagged_sentences:
            reasons.append(f"flagged_sentences:{len(entry.flagged_sentences)}")
        if entry.context_truncated:
            reasons.append("context_truncated")
        return ", ".join(reasons) if reasons else "warnings"

    # ── Feedback logging ───────────────────────────────────────────────────

    def log_feedback(
        self,
        query_event_id: str,
        feedback_type: str,
        query_preview: str,
        session_id: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        entry = FeedbackEntry(
            event_id=event_id,
            timestamp=_now(),
            query_event_id=query_event_id,
            feedback_type=feedback_type,
            query_preview=query_preview[:100],
            session_id=session_id,
            comment=comment,
        )
        _append(FEEDBACK_LOG, asdict(entry))
        logger.info(f"Feedback logged: {feedback_type} for event {query_event_id}")
        return event_id

    # ── Review queue ────────────────────────────────────────────────────────

    def get_review_queue(self, limit: int = 50) -> list[dict]:
        """Return most recent flagged entries for the human review dashboard."""
        return _tail(FLAGGED_LOG, limit)

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self, last_n: int = 1000) -> dict:
        """Return aggregate stats over the last N queries."""
        entries = _tail(QUERY_LOG, last_n)
        if not entries:
            return {"total_queries": 0}

        total = len(entries)
        safe = sum(1 for e in entries if e.get("safety_category") == "safe")
        flagged = sum(1 for e in entries if e.get("has_warnings"))
        avg_faith = sum(e.get("faithfulness", 1) for e in entries) / total
        avg_latency = sum(e.get("latency_ms", 0) for e in entries) / total

        categories = {}
        for e in entries:
            cat = e.get("safety_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        feedback = _tail(FEEDBACK_LOG, last_n)
        pos = sum(1 for f in feedback if f.get("feedback_type") == "positive")
        neg = sum(1 for f in feedback if f.get("feedback_type") == "negative")

        return {
            "total_queries": total,
            "safe_queries": safe,
            "flagged_queries": flagged,
            "flag_rate": round(flagged / total, 3),
            "avg_faithfulness": round(avg_faith, 3),
            "avg_latency_ms": round(avg_latency),
            "safety_breakdown": categories,
            "feedback": {
                "positive": pos,
                "negative": neg,
                "total": pos + neg,
                "satisfaction_rate": round(pos / (pos + neg), 3) if (pos + neg) > 0 else None,
            },
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _append(path: Path, record: dict):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Audit log write failed ({path}): {e}")

def _tail(path: Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-n:] if l.strip()]
    except Exception as e:
        logger.error(f"Audit log read failed ({path}): {e}")
        return []


# ── Singleton ──────────────────────────────────────────────────────────────────
audit_logger = AuditLogger()
