"""
api/main.py
FastAPI REST API for the Medical RAG system.

Endpoints:
  POST /ask          — Ask a single question
  POST /chat         — Multi-turn conversation
  GET  /health       — Health check + pipeline status
  GET  /sources      — Knowledge base statistics

Run with:
  uvicorn api.main:app --reload --port 8000
"""

import os
import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pipeline.generator import MedicalRAGPipeline, RAGResponse
from pipeline.prompt_builder import ConversationTurn
from config.settings import DISCLAIMER


# ─── App lifecycle ────────────────────────────────────────────────────────────

pipeline: Optional[MedicalRAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting Medical RAG API...")
    pipeline = MedicalRAGPipeline(api_key=os.getenv("GROQ_API_KEY"))
    pipeline._load_components()
    logger.info("Pipeline loaded. API ready.")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="Medical Knowledge RAG API",
    description=(
        "Responsible medical question-answering using Retrieval-Augmented Generation. "
        "All responses are grounded in curated medical documents and include source citations."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000,
                       description="Medical question to answer")
    specialty_filter: Optional[str] = Field(
        None, description="Restrict retrieval to a specialty (e.g. 'cardiology')"
    )
    evidence_filter: Optional[list[str]] = Field(
        None, description="Restrict to evidence levels e.g. ['guideline', 'rct']"
    )

    model_config = {"json_schema_extra": {
        "example": {"query": "What are the symptoms of type 2 diabetes?"}
    }}


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ..., min_length=1, description="Conversation history ending with the user's question"
    )
    specialty_filter: Optional[str] = None
    evidence_filter: Optional[list[str]] = None

    model_config = {"json_schema_extra": {
        "example": {
            "messages": [
                {"role": "user", "content": "What causes high blood pressure?"},
                {"role": "assistant", "content": "High blood pressure is caused by..."},
                {"role": "user", "content": "What lifestyle changes help?"},
            ]
        }
    }}


class CitationOut(BaseModel):
    number: int
    title: str
    source_name: str
    url: str
    evidence_level: str
    pub_date: str


class AskResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    citations: list[CitationOut]
    context_chunks_used: int
    faithfulness: float
    has_warnings: bool
    safety_category: str
    latency_ms: int
    disclaimer: str = DISCLAIMER


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check and pipeline status."""
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "timestamp": time.time(),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Ask a single medical question.
    Returns a grounded answer with cited sources and faithfulness score.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    try:
        result: RAGResponse = pipeline.ask(
            query=request.query,
            specialty_filter=request.specialty_filter,
            evidence_filter=request.evidence_filter,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return AskResponse(
        request_id=str(uuid.uuid4()),
        query=result.query,
        answer=result.answer,
        citations=[CitationOut(**c) for c in result.citations],
        context_chunks_used=result.context_chunks_used,
        faithfulness=result.faithfulness,
        has_warnings=result.has_warnings,
        safety_category=result.safety_category,
        latency_ms=result.latency_ms,
    )


@app.post("/chat", response_model=AskResponse)
async def chat(request: ChatRequest):
    """
    Multi-turn conversation endpoint.
    Pass the full message history; the last user message is treated as the question.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    messages = request.messages
    if not messages or messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message must be from the user."
        )

    # Current query is the last user message
    query = messages[-1].content

    # Prior turns become conversation history
    history = [
        ConversationTurn(role=m.role, content=m.content)
        for m in messages[:-1]
    ]

    try:
        result: RAGResponse = pipeline.ask(
            query=query,
            history=history,
            specialty_filter=request.specialty_filter,
            evidence_filter=request.evidence_filter,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(
        request_id=str(uuid.uuid4()),
        query=query,
        answer=result.answer,
        citations=[CitationOut(**c) for c in result.citations],
        context_chunks_used=result.context_chunks_used,
        faithfulness=result.faithfulness,
        has_warnings=result.has_warnings,
        safety_category=result.safety_category,
        latency_ms=result.latency_ms,
    )


@app.get("/sources")
async def sources():
    """Return knowledge base statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    try:
        from pipeline.embedder import VectorStoreBuilder
        builder = VectorStoreBuilder()
        stats = builder.collection_stats()
        return {"knowledge_base": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
