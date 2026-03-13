"""
pipeline/safety.py
Safety layer — runs BEFORE and AFTER LLM generation.

Pre-generation checks:
  1. Emergency detection  → bypass RAG, return emergency message immediately
  2. Harmful intent       → refuse and explain why
  3. Out-of-scope         → politely redirect

Post-generation checks:
  4. Hallucination guard  → NLI entailment check against retrieved context
  5. Disclaimer injection → appended to every response without exception
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    DISCLAIMER, EMERGENCY_MESSAGE,
    EMERGENCY_PATTERNS, HARMFUL_PATTERNS, OUT_OF_SCOPE_PATTERNS,
    NLI_MODEL, NLI_ENTAILMENT_THRESHOLD,
)


# ─── Safety classification ────────────────────────────────────────────────────

class QueryCategory(Enum):
    SAFE         = "safe"
    EMERGENCY    = "emergency"
    HARMFUL      = "harmful"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class SafetyResult:
    category: QueryCategory
    safe: bool
    reason: Optional[str] = None
    response: Optional[str] = None   # Pre-built response for unsafe queries


# ─── Pre-generation filter ────────────────────────────────────────────────────

class PreGenerationFilter:
    """
    Fast pattern-based check before any retrieval or LLM call.
    Order matters: emergency check runs first to minimise latency for
    genuine crisis queries.
    """

    def check(self, query: str) -> SafetyResult:
        q = query.lower().strip()

        # 1. Emergency — respond immediately, no RAG
        if self._matches(q, EMERGENCY_PATTERNS):
            logger.warning(f"Emergency pattern detected: '{query[:60]}'")
            return SafetyResult(
                category=QueryCategory.EMERGENCY,
                safe=False,
                reason="Emergency keyword detected",
                response=(
                    f"{EMERGENCY_MESSAGE}\n\n"
                    "Please call emergency services or go to the nearest emergency room now.\n\n"
                    f"{DISCLAIMER}"
                ),
            )

        # 2. Harmful intent — refuse clearly
        if self._matches(q, HARMFUL_PATTERNS):
            logger.warning(f"Harmful intent detected: '{query[:60]}'")
            return SafetyResult(
                category=QueryCategory.HARMFUL,
                safe=False,
                reason="Potentially harmful query",
                response=(
                    "I'm not able to provide information that could be used to cause harm.\n\n"
                    "If you or someone you know is in crisis, please contact:\n"
                    "• Emergency services: 911\n"
                    "• National Crisis Hotline: 988 (call or text)\n"
                    "• Crisis Text Line: Text HOME to 741741\n\n"
                    f"{DISCLAIMER}"
                ),
            )

        # 3. Out of scope — redirect politely
        if self._matches(q, OUT_OF_SCOPE_PATTERNS):
            return SafetyResult(
                category=QueryCategory.OUT_OF_SCOPE,
                safe=False,
                reason="Query outside medical domain",
                response=(
                    "This assistant is designed specifically for medical and health information. "
                    "Your question appears to be outside that scope.\n\n"
                    "Please ask a health-related question and I'll do my best to help.\n\n"
                    f"{DISCLAIMER}"
                ),
            )

        return SafetyResult(category=QueryCategory.SAFE, safe=True)

    def _matches(self, query: str, patterns: list[str]) -> bool:
        return any(p in query for p in patterns)


# ─── Hallucination guard ──────────────────────────────────────────────────────

class HallucinationGuard:
    """
    Post-generation NLI check.

    Splits the generated answer into sentences, then checks each sentence
    against the retrieved context using a Natural Language Inference model.

    A sentence is flagged if the NLI model scores it as CONTRADICTION or
    NEUTRAL (not entailed) below the confidence threshold.

    This is a best-effort check — NLI models are imperfect, especially on
    medical text. Flagged sentences are highlighted, not removed.
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading NLI model: {NLI_MODEL}")
                self._model = CrossEncoder(NLI_MODEL)
                logger.info("NLI hallucination guard loaded.")
            except Exception as e:
                logger.warning(f"NLI model failed to load: {e}. Guard disabled.")

    def check(self, answer: str, context: str) -> dict:
        """
        Returns:
          {
            "flagged_sentences": [...],   # Sentences not supported by context
            "supported_sentences": [...],
            "overall_faithfulness": float  # 0-1, fraction of sentences supported
          }
        """
        self._load()
        sentences = self._split_sentences(answer)

        if not sentences:
            return {"flagged_sentences": [], "supported_sentences": [],
                    "overall_faithfulness": 1.0}

        if self._model is None:
            # Guard disabled — return all as supported
            return {
                "flagged_sentences": [],
                "supported_sentences": sentences,
                "overall_faithfulness": 1.0,
            }

        # Score each (context, sentence) pair
        # NLI label order: contradiction=0, entailment=1, neutral=2
        pairs = [(context, sent) for sent in sentences]
        scores = self._model.predict(pairs, apply_softmax=True)

        flagged = []
        supported = []

        for sent, score in zip(sentences, scores):
            entailment_score = float(score[1])  # Index 1 = entailment
            if entailment_score >= NLI_ENTAILMENT_THRESHOLD:
                supported.append(sent)
            else:
                flagged.append(sent)
                logger.debug(
                    f"Flagged sentence (entailment={entailment_score:.2f}): "
                    f"'{sent[:80]}...'"
                )

        faithfulness = len(supported) / len(sentences) if sentences else 1.0

        return {
            "flagged_sentences": flagged,
            "supported_sentences": supported,
            "overall_faithfulness": round(faithfulness, 3),
        }

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter for post-processing."""
        # Split on . ! ? followed by whitespace and capital letter
        raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in raw if len(s.strip()) > 20]


# ─── Post-generation processor ────────────────────────────────────────────────

class PostGenerationProcessor:
    """
    Runs after LLM generation:
    1. Hallucination check
    2. Disclaimer injection
    3. Flagged-sentence annotation (optional, for transparency)
    """

    def __init__(self, annotate_flagged: bool = False):
        self.guard = HallucinationGuard()
        self.annotate_flagged = annotate_flagged

    def process(self, answer: str, context: str) -> dict:
        """
        Returns processed response dict:
          {
            "answer": str,                    # Final answer with disclaimer
            "faithfulness": float,
            "flagged_sentences": list[str],
            "has_warnings": bool,
          }
        """
        # Run hallucination check
        nli_result = self.guard.check(answer, context)
        faithfulness = nli_result["overall_faithfulness"]
        flagged = nli_result["flagged_sentences"]

        # Annotate flagged sentences if enabled
        final_answer = answer
        if self.annotate_flagged and flagged:
            for sent in flagged:
                annotated = f"{sent} ⚠️"
                final_answer = final_answer.replace(sent, annotated)

        # Always inject disclaimer
        final_answer = f"{final_answer}\n\n---\n{DISCLAIMER}"

        # Low faithfulness warning
        has_warnings = faithfulness < 0.7 or len(flagged) > 0

        return {
            "answer": final_answer,
            "faithfulness": faithfulness,
            "flagged_sentences": flagged,
            "has_warnings": has_warnings,
        }
