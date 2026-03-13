"""
pipeline/generator.py
LLM generation with retry, streaming, and structured response output.

Connects the retrieved context (Phase 1) to the LLM to produce
grounded, cited, faithful answers.

Key design decisions:
- Temperature 0.1 → maximally factual / deterministic
- Stream support → better UX for longer answers
- Exponential backoff retry → handles API rate limits gracefully
- Structured RAGResponse output → carries faithfulness and citation metadata
  downstream to the API layer
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Iterator
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
)


# ─── Response model ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Full response from the RAG pipeline.
    Carries the answer plus all metadata needed for the UI and audit log.
    """
    answer: str                          # Final answer with disclaimer
    query: str                           # Original user query
    citations: list[dict]                # Source list with URLs and evidence levels
    context_chunks_used: int             # How many chunks fed to LLM
    context_truncated: bool              # Were any chunks dropped?
    faithfulness: float                  # NLI faithfulness score (0–1)
    flagged_sentences: list[str]         # Sentences not supported by context
    has_warnings: bool                   # Any safety or faithfulness warnings
    safety_category: str                 # "safe" / "emergency" / "harmful" / "out_of_scope"
    latency_ms: int                      # Total pipeline latency
    model_used: str = LLM_MODEL


# ─── Groq client wrapper ─────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around the Groq chat completions API.
    Groq is OpenAI-API-compatible so only the client init and base_url differ.
    Get a free API key at: https://console.groq.com
    Handles: API key loading, retries, streaming, and error normalisation.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0   # seconds, doubled each retry

    def __init__(self, api_key: Optional[str] = None):
        self._client = None
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self._api_key:
            logger.warning(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com\n"
                "  Windows:  $env:GROQ_API_KEY='gsk_...'\n"
                "  Linux/Mac: export GROQ_API_KEY='gsk_...'"
            )

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    def complete(
        self,
        system: str,
        messages: list[dict],
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        """
        Non-streaming completion with exponential backoff retry.
        Returns the assistant's reply as a plain string.
        """
        client = self._get_client()
        full_messages = [{"role": "system", "content": system}] + messages

        delay = self.RETRY_DELAY
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            except Exception as e:
                error_type = type(e).__name__
                if attempt == self.MAX_RETRIES:
                    logger.error(f"LLM call failed after {self.MAX_RETRIES} attempts: {e}")
                    raise
                logger.warning(
                    f"LLM attempt {attempt}/{self.MAX_RETRIES} failed ({error_type}). "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2

    def stream(
        self,
        system: str,
        messages: list[dict],
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> Iterator[str]:
        """
        Streaming completion. Yields text chunks as they arrive.
        Usage:
            for chunk in client.stream(system, messages):
                print(chunk, end="", flush=True)
        """
        client = self._get_client()
        full_messages = [{"role": "system", "content": system}] + messages

        stream = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ─── RAG pipeline orchestrator ────────────────────────────────────────────────

class MedicalRAGPipeline:
    """
    Full RAG pipeline orchestrator for Phase 2.

    Query → Safety check → Retrieval → Prompt building
           → LLM generation → Post-processing → RAGResponse

    This is the single entry point for the API layer (Phase 4).
    """

    def __init__(self, api_key: Optional[str] = None):
        # Lazy-load heavy components
        self._retriever = None
        self._llm = LLMClient(api_key=api_key)
        self._safety_pre = None
        self._safety_post = None
        self._prompt_builder = None

    def _load_components(self):
        """Lazy-load all pipeline components on first call."""
        if self._retriever is None:
            from pipeline.retriever import HybridRetriever
            from pipeline.safety import PreGenerationFilter, PostGenerationProcessor
            from pipeline.prompt_builder import PromptBuilder

            logger.info("Loading RAG pipeline components...")
            self._retriever = HybridRetriever()
            self._safety_pre = PreGenerationFilter()
            self._safety_post = PostGenerationProcessor()
            self._prompt_builder = PromptBuilder()
            logger.info("Pipeline ready.")

    def ask(
        self,
        query: str,
        history: Optional[list] = None,
        specialty_filter: Optional[str] = None,
        evidence_filter: Optional[list[str]] = None,
        stream: bool = False,
    ) -> RAGResponse:
        """
        Ask the pipeline a medical question.

        Args:
            query:            The user's question.
            history:          Prior conversation turns (ConversationTurn list).
            specialty_filter: Restrict retrieval to a medical specialty.
            evidence_filter:  Restrict to specific evidence levels
                              e.g. ["guideline", "rct"]
            stream:           If True, print streamed output (for CLI use).

        Returns:
            RAGResponse with answer, citations, faithfulness, and metadata.
        """
        start_ms = int(time.time() * 1000)
        self._load_components()

        # ── Stage 1: Pre-generation safety check ──────────────────────────────
        safety_result = self._safety_pre.check(query)
        if not safety_result.safe:
            return RAGResponse(
                answer=safety_result.response,
                query=query,
                citations=[],
                context_chunks_used=0,
                context_truncated=False,
                faithfulness=1.0,
                flagged_sentences=[],
                has_warnings=True,
                safety_category=safety_result.category.value,
                latency_ms=int(time.time() * 1000) - start_ms,
            )

        # ── Stage 2: Retrieval ─────────────────────────────────────────────────
        logger.info(f"Retrieving context for: '{query[:70]}'")
        retrieved = self._retriever.retrieve(
            query,
            specialty_filter=specialty_filter,
            evidence_filter=evidence_filter,
        )

        if not retrieved:
            from config.settings import DISCLAIMER
            return RAGResponse(
                answer=(
                    "I wasn't able to find relevant information in the medical knowledge base "
                    "for your question. Please try rephrasing, or consult a healthcare "
                    f"professional directly.\n\n---\n{DISCLAIMER}"
                ),
                query=query,
                citations=[],
                context_chunks_used=0,
                context_truncated=False,
                faithfulness=1.0,
                flagged_sentences=[],
                has_warnings=False,
                safety_category="safe",
                latency_ms=int(time.time() * 1000) - start_ms,
            )

        # ── Stage 3: Prompt construction ──────────────────────────────────────
        built_prompt, citations = self._prompt_builder.build(
            query=query,
            retrieved_chunks=retrieved,
            history=history or [],
        )

        # ── Stage 4: LLM generation ───────────────────────────────────────────
        logger.info(
            f"Generating answer | chunks={built_prompt.context_used} "
            f"truncated={built_prompt.context_truncated}"
        )
        if stream:
            # Stream to stdout and collect full answer
            print("\nAssistant: ", end="", flush=True)
            answer_parts = []
            for chunk in self._llm.stream(built_prompt.system, built_prompt.messages):
                print(chunk, end="", flush=True)
                answer_parts.append(chunk)
            print()
            raw_answer = "".join(answer_parts)
        else:
            raw_answer = self._llm.complete(built_prompt.system, built_prompt.messages)

        # ── Stage 5: Post-generation safety + disclaimer ──────────────────────
        # Build context string for faithfulness check
        context_for_nli = " ".join(
            getattr(c, "content", "") for c in retrieved
        )
        post_result = self._safety_post.process(raw_answer, context_for_nli)

        latency = int(time.time() * 1000) - start_ms
        logger.info(
            f"Response generated | faithfulness={post_result['faithfulness']:.2f} "
            f"latency={latency}ms flagged={len(post_result['flagged_sentences'])}"
        )

        return RAGResponse(
            answer=post_result["answer"],
            query=query,
            citations=citations,
            context_chunks_used=built_prompt.context_used,
            context_truncated=built_prompt.context_truncated,
            faithfulness=post_result["faithfulness"],
            flagged_sentences=post_result["flagged_sentences"],
            has_warnings=post_result["has_warnings"],
            safety_category="safe",
            latency_ms=latency,
        )

    def ask_stream(self, query: str, history: Optional[list] = None) -> RAGResponse:
        """Convenience wrapper for streaming mode."""
        return self.ask(query, history=history, stream=True)
