"""
api/main.py  — Phase 3 update
FastAPI REST API with audit logging, feedback, and monitoring endpoints.

New in Phase 3:
  POST /feedback     — Submit thumbs-up/down on a response
  GET  /admin/stats  — Aggregate pipeline metrics
  GET  /admin/review — Human review queue (flagged responses)

Run with:
  uvicorn api.main:app --reload --port 8000
"""

import os
import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pipeline.generator import MedicalRAGPipeline, RAGResponse
from pipeline.prompt_builder import ConversationTurn
from pipeline.audit_log import audit_logger
from config.settings import DISCLAIMER


pipeline: Optional[MedicalRAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting Medical RAG API (Phase 3)...")
    pipeline = MedicalRAGPipeline(api_key=os.getenv("GROQ_API_KEY"))
    pipeline._load_components()
    logger.info("Pipeline loaded. API ready.")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="Medical Knowledge RAG API",
    description="Responsible medical QA with RAG. Phase 3: UI + audit logging + feedback.",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UI_DIR = Path(__file__).parent.parent / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    specialty_filter: Optional[str] = None
    evidence_filter: Optional[list[str]] = None
    session_id: Optional[str] = None
    model_config = {"json_schema_extra": {"example": {"query": "What are symptoms of diabetes?"}}}


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    specialty_filter: Optional[str] = None
    evidence_filter: Optional[list[str]] = None
    session_id: Optional[str] = None


class CitationOut(BaseModel):
    number: int
    title: str
    source_name: str
    url: str
    evidence_level: str
    pub_date: str


class AskResponse(BaseModel):
    request_id: str
    event_id: str
    query: str
    answer: str
    citations: list[CitationOut]
    context_chunks_used: int
    faithfulness: float
    has_warnings: bool
    safety_category: str
    latency_ms: int
    disclaimer: str = DISCLAIMER


class FeedbackRequest(BaseModel):
    event_id: str
    feedback_type: str = Field(..., pattern="^(positive|negative)$")
    query_preview: str = Field("", max_length=200)
    session_id: Optional[str] = None
    comment: Optional[str] = Field(None, max_length=500)


def _build_response(result: RAGResponse, event_id: str) -> AskResponse:
    return AskResponse(
        request_id=str(uuid.uuid4()),
        event_id=event_id,
        query=result.query,
        answer=result.answer,
        citations=[CitationOut(**c) for c in result.citations],
        context_chunks_used=result.context_chunks_used,
        faithfulness=result.faithfulness,
        has_warnings=result.has_warnings,
        safety_category=result.safety_category,
        latency_ms=result.latency_ms,
    )


def _log_result(result: RAGResponse, session_id: Optional[str] = None) -> str:
    return audit_logger.log_query(
        query=result.query,
        safety_category=result.safety_category,
        context_chunks_used=result.context_chunks_used,
        context_truncated=result.context_truncated,
        faithfulness=result.faithfulness,
        has_warnings=result.has_warnings,
        flagged_sentences=result.flagged_sentences,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
        citations_count=len(result.citations),
        session_id=session_id,
    )


@app.get("/")
async def root():
    ui_index = UI_DIR / "index.html"
    if ui_index.exists():
        return FileResponse(str(ui_index))
    return {"message": "Medical RAG API v3. UI at /ui/index.html"}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0",
            "pipeline_loaded": pipeline is not None, "timestamp": time.time()}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    try:
        result = pipeline.ask(query=request.query,
                              specialty_filter=request.specialty_filter,
                              evidence_filter=request.evidence_filter)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    event_id = _log_result(result, request.session_id)
    return _build_response(result, event_id)


@app.post("/chat", response_model=AskResponse)
async def chat(request: ChatRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    messages = request.messages
    if not messages or messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from the user.")
    query = messages[-1].content
    history = [ConversationTurn(role=m.role, content=m.content) for m in messages[:-1]]
    try:
        result = pipeline.ask(query=query, history=history,
                              specialty_filter=request.specialty_filter,
                              evidence_filter=request.evidence_filter)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    event_id = _log_result(result, request.session_id)
    return _build_response(result, event_id)


@app.get("/sources")
async def sources():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    try:
        from pipeline.embedder import VectorStoreBuilder
        return {"knowledge_base": VectorStoreBuilder().collection_stats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        fid = audit_logger.log_feedback(
            query_event_id=request.event_id,
            feedback_type=request.feedback_type,
            query_preview=request.query_preview,
            session_id=request.session_id,
            comment=request.comment,
        )
        return {"status": "ok", "feedback_id": fid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/stats")
async def admin_stats(last_n: int = 1000):
    """Aggregate metrics: query volume, flag rates, faithfulness, latency, feedback."""
    return audit_logger.get_stats(last_n=last_n)


@app.get("/admin/review")
async def admin_review(limit: int = 50):
    """Human review queue — most recent flagged / low-faithfulness responses."""
    queue = audit_logger.get_review_queue(limit=limit)
    return {"count": len(queue), "items": queue}
