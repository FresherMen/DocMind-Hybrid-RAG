"""
FastAPI Backend — DocMind RAG API
Endpoints: /ingest, /query, /compare, /eval, /docs/list
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.rag_engine import RAGEngine, RAGResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────

engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine on startup if API key is set in environment."""
    global engine
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        engine = RAGEngine(
            openai_api_key=api_key,
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_store"),
        )
        logger.info("RAG engine initialized from environment.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="DocMind RAG API",
    description="Production RAG system with hybrid retrieval, reranking, and RAGAS-style eval",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class InitRequest(BaseModel):
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    top_k: int = Field(default=6, ge=1, le=20)
    use_compression: bool = True
    persist_dir: str = "./chroma_store"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: Optional[int] = None


class CompareRequest(BaseModel):
    question: str


class ChunkOut(BaseModel):
    content: str
    source: str
    page: int
    dense_score: float
    bm25_score: float
    final_score: float
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[ChunkOut]
    confidence: float
    reasoning: str
    query: str
    expanded_queries: List[str]
    latency_ms: int
    eval_metrics: Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def require_engine():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized. POST /init first.")


def serialize_response(resp: RAGResponse) -> QueryResponse:
    return QueryResponse(
        answer=resp.answer,
        sources=[
            ChunkOut(
                content=s.content,
                source=s.source,
                page=s.page,
                dense_score=s.dense_score,
                bm25_score=s.bm25_score,
                final_score=s.final_score,
                chunk_id=s.chunk_id,
            )
            for s in resp.sources
        ],
        confidence=resp.confidence,
        reasoning=resp.reasoning,
        query=resp.query,
        expanded_queries=resp.expanded_queries,
        latency_ms=resp.latency_ms,
        eval_metrics=resp.eval_metrics,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "DocMind RAG API is running. Go to /docs for documentation."}


@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": engine is not None}


@app.post("/init", summary="Initialize the RAG engine")
async def init_engine(req: InitRequest):
    global engine
    engine = RAGEngine(
        openai_api_key=req.openai_api_key,
        llm_model=req.llm_model,
        top_k=req.top_k,
        use_compression=req.use_compression,
        persist_dir=req.persist_dir,
    )
    return {"status": "initialized", "model": req.llm_model}


@app.post("/ingest", summary="Upload and index documents")
async def ingest(files: List[UploadFile] = File(...)):
    require_engine()
    results = []
    for uf in files:
        suffix = Path(uf.filename).suffix
        if suffix.lower() not in (".pdf", ".txt", ".md"):
            raise HTTPException(400, f"Unsupported file type: {suffix}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await uf.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = engine.ingest_file(tmp_path)
            results.append(vars(result))
        except Exception as e:
            raise HTTPException(500, str(e))
        finally:
            os.unlink(tmp_path)

    return {"ingested": results, "total_docs": len(engine.doc_registry)}


@app.post("/query", response_model=QueryResponse, summary="Ask a question (hybrid RAG)")
async def query(req: QueryRequest):
    require_engine()
    if req.top_k:
        engine.top_k = req.top_k
    try:
        resp = engine.query(req.question)
        return serialize_response(resp)
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/compare", summary="Compare question across all loaded documents")
async def compare(req: CompareRequest):
    require_engine()
    try:
        return engine.compare_across_docs(req.question)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/documents", summary="List all indexed documents")
async def list_documents():
    require_engine()
    return {"documents": engine.get_registry()}


@app.delete("/documents", summary="Clear all indexed documents")
async def clear_documents():
    require_engine()
    engine.clear()
    return {"status": "cleared"}


@app.get("/eval/metrics", summary="Retrieval quality metrics for a query")
async def eval_metrics(question: str):
    """Run a query and return only RAGAS-style eval metrics."""
    require_engine()
    resp = engine.query(question)
    return {
        "question": question,
        "eval_metrics": resp.eval_metrics,
        "confidence": resp.confidence,
        "latency_ms": resp.latency_ms,
        "sources_retrieved": len(resp.sources),
    }

# Hot reload trigger for dependency installation

