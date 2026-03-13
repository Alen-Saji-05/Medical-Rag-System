"""
pipeline/prompt_builder.py
Constructs the LLM prompt from retrieved chunks.

The prompt design is critical for RAG quality:
- System prompt defines the assistant's role, constraints, and citation format
- Each retrieved chunk is clearly labelled with source metadata
- The model is explicitly instructed to cite sources by [1], [2], etc.
- Uncertainty is required: the model must say "the sources don't mention" rather
  than hallucinate when context is insufficient
- Conversation history is included for coherent multi-turn dialogue
"""

from dataclasses import dataclass
from typing import Optional
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import LLM_CONTEXT_WINDOW, MAX_HISTORY_TURNS


# ─── Data models ──────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    role: str    # "user" or "assistant"
    content: str


@dataclass
class BuiltPrompt:
    system: str
    messages: list[dict]       # OpenAI-format messages
    context_used: int          # Number of chunks included
    context_truncated: bool    # Whether any chunks were dropped due to length


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a medical knowledge assistant that answers health questions using only the provided reference documents. You are NOT a doctor and cannot provide personalised medical advice.

## Your responsibilities
- Answer questions accurately using ONLY the information in the provided [SOURCE] blocks below
- Cite every factual claim with the source number in brackets, e.g. [1], [2]
- If multiple sources support a claim, cite all of them: [1][3]
- Be clear about what the evidence level of each source is (guideline, fact sheet, research, etc.)

## When information is missing
- If the provided sources do not contain enough information to answer the question, say exactly: "The available sources don't contain sufficient information about this. I recommend consulting a healthcare professional or visiting a trusted source like MedlinePlus (medlineplus.gov)."
- Never speculate or add information beyond what the sources state
- Never invent citations

## Format rules
- Use plain, clear language — avoid jargon unless explaining it
- For lists of symptoms, treatments, or steps: use bullet points
- Keep answers focused and concise — do not pad with disclaimers within the answer body (a disclaimer is appended automatically)
- Do not start your answer with "Based on the provided sources" — just answer directly

## Absolute limits
- Never recommend specific drug dosages or prescriptions
- Never diagnose a condition
- Never contradict the provided sources
- If a question involves a medical emergency, direct to emergency services immediately"""


# ─── Context formatter ────────────────────────────────────────────────────────

class ContextFormatter:
    """
    Formats retrieved chunks into numbered [SOURCE] blocks for the prompt.

    Each block includes:
    - Source number (for citation)
    - Document title and authority
    - Evidence level (guideline / factsheet / review / rct)
    - Publication date
    - The chunk content itself

    This structured format helps the LLM cite accurately and lets the
    user verify claims against the original source.
    """

    def format(self, chunks: list) -> tuple[str, list[dict]]:
        """
        Returns:
          - context_text: formatted string to embed in the prompt
          - citations: list of citation dicts for the response metadata
        """
        if not chunks:
            return "No relevant sources were retrieved.", []

        parts = []
        citations = []

        for i, chunk in enumerate(chunks, 1):
            source_name = getattr(chunk, "source_name", "Unknown")
            title = getattr(chunk, "title", "Unknown")
            evidence = getattr(chunk, "evidence_level", "unknown")
            pub_date = getattr(chunk, "pub_date", "unknown")
            url = getattr(chunk, "source_url", "")
            content = getattr(chunk, "content", "")
            score = getattr(chunk, "rerank_score", 0.0)

            block = (
                f"[SOURCE {i}]\n"
                f"Title: {title}\n"
                f"From: {source_name} | Evidence: {evidence} | Date: {pub_date}\n"
                f"Relevance score: {score:.3f}\n"
                f"---\n"
                f"{content}\n"
            )
            parts.append(block)
            citations.append({
                "number": i,
                "title": title,
                "source_name": source_name,
                "url": url,
                "evidence_level": evidence,
                "pub_date": pub_date,
                "rerank_score": score,
            })

        context_text = "\n\n".join(parts)
        return context_text, citations


# ─── Prompt builder ───────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles the full prompt for the LLM from:
    - Retrieved context chunks
    - Conversation history
    - Current user query

    Token budget management:
    We keep a running estimate of token usage and drop the lowest-scoring
    chunks first if the context would exceed LLM_CONTEXT_WINDOW.
    This ensures the most relevant evidence always fits.
    """

    def __init__(self):
        self.formatter = ContextFormatter()

    def build(
        self,
        query: str,
        retrieved_chunks: list,
        history: Optional[list[ConversationTurn]] = None,
        max_context_tokens: int = LLM_CONTEXT_WINDOW,
    ) -> BuiltPrompt:
        """
        Build the full prompt. Returns a BuiltPrompt with:
        - system: system prompt string
        - messages: list of {role, content} dicts in OpenAI format
        - context_used: how many chunks were included
        - context_truncated: whether chunks were dropped
        """
        history = history or []

        # Fit chunks within token budget
        chunks_to_use, truncated = self._fit_chunks(
            retrieved_chunks, query, max_context_tokens
        )

        # Format context into labelled source blocks
        context_text, citations = self.formatter.format(chunks_to_use)

        # Build the user message: context + question
        user_content = (
            f"## Reference sources\n\n"
            f"{context_text}\n\n"
            f"---\n\n"
            f"## Question\n{query}"
        )

        # Assemble message list: history + current turn
        messages = []
        recent_history = history[-(MAX_HISTORY_TURNS * 2):]  # Keep last N turns
        for turn in recent_history:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_content})

        return BuiltPrompt(
            system=SYSTEM_PROMPT,
            messages=messages,
            context_used=len(chunks_to_use),
            context_truncated=truncated,
        ), citations

    def _fit_chunks(
        self, chunks: list, query: str, max_tokens: int
    ) -> tuple[list, bool]:
        """
        Drop lowest-scoring chunks if total context exceeds token budget.
        Chunks are already sorted best-first from the reranker.
        Rough token estimate: 1 token ≈ 4 characters.
        """
        # Reserve tokens for system prompt + query + response
        reserved = 2000
        budget = max_tokens - reserved

        used = []
        total_chars = 0
        truncated = False

        for chunk in chunks:
            content = getattr(chunk, "content", "")
            chunk_chars = len(content) + 200  # +200 for metadata overhead
            if total_chars + chunk_chars > budget * 4:
                truncated = True
                logger.debug(
                    f"Context budget reached at {len(used)} chunks "
                    f"({total_chars // 4} tokens estimated). "
                    f"Dropping {len(chunks) - len(used)} lower-scored chunks."
                )
                break
            used.append(chunk)
            total_chars += chunk_chars

        return used, truncated
