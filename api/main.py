"""
api/main.py
-----------
FastAPI application exposing the semantic search system.

Startup:
    uvicorn api.main:app --reload

Endpoints:
    POST   /query          — semantic search with cache
    GET    /cache/stats    — cache performance metrics
    DELETE /cache          — flush cache
    GET    /health         — liveness check
    GET    /               — API info
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so relative imports work regardless
# of working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from services.query_service import QueryService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application configuration (via environment variables)
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
TOP_K: int = int(os.getenv("TOP_K", "10"))
REBUILD_INDEX: bool = os.getenv("REBUILD_INDEX", "false").lower() == "true"

# Set MAX_DOCS=500 in .env for faster startup during development
_max_docs_env = os.getenv("MAX_DOCS", "")
MAX_DOCS: Optional[int] = int(_max_docs_env) if _max_docs_env.isdigit() else None


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

# Single shared instance of the query service
_service: Optional[QueryService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Initialises the query service on startup and cleanly shuts down.
    Using lifespan (vs. deprecated @app.on_event) is the current best practice.
    """
    global _service
    logger.info("Starting up semantic search system…")
    _service = QueryService(
        similarity_threshold=SIMILARITY_THRESHOLD,
        top_k=TOP_K,
        rebuild=REBUILD_INDEX,
        max_docs=MAX_DOCS,
    )
    _service.initialise()
    logger.info("System ready. API accepting requests.")
    yield
    # Shutdown: persist cache so warm entries survive restarts
    if _service and _service.is_ready:
        _service._cache.save()
        logger.info("Cache persisted. Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic Search API",
    description=(
        "Production-ready semantic search over the 20 Newsgroups corpus. "
        "Features: sentence-transformer embeddings, FAISS vector DB, "
        "Gaussian Mixture fuzzy clustering, and a custom semantic cache."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language search query.",
        example="Explain gun control policies",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class SearchResult(BaseModel):
    doc_id: int
    score: float
    category: str
    text_snippet: str
    cluster_distribution: Dict[str, float]


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: List[SearchResult]
    dominant_cluster: int
    latency_ms: float


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class FlushResponse(BaseModel):
    message: str
    stats: CacheStatsResponse


class HealthResponse(BaseModel):
    status: str
    system_ready: bool
    similarity_threshold: float
    top_k: int


# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------

def get_service() -> QueryService:
    """Retrieve the shared service instance, raising 503 if not ready."""
    if _service is None or not _service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is initialising. Please retry in a moment.",
        )
    return _service


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
async def root() -> dict:
    """API welcome message and quick reference."""
    return {
        "name": "Semantic Search API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Semantic search with caching",
            "GET  /cache/stats": "Cache performance metrics",
            "DELETE /cache": "Flush cache",
            "GET  /health": "Liveness check",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Liveness probe — returns 200 if the system is ready."""
    return HealthResponse(
        status="ok" if (_service and _service.is_ready) else "initialising",
        system_ready=bool(_service and _service.is_ready),
        similarity_threshold=SIMILARITY_THRESHOLD,
        top_k=TOP_K,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Search"],
    summary="Semantic search with cache",
    responses={
        200: {"description": "Search results (cache hit or miss)"},
        422: {"description": "Invalid request body"},
        503: {"description": "System initialising"},
    },
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Execute a semantic search query.

    **Processing steps:**
    1. Embed the query using all-MiniLM-L6-v2.
    2. Determine cluster membership via the fitted GMM.
    3. Check the semantic cache — return immediately on hit.
    4. On miss: search the FAISS vector DB, store result in cache.

    **Cache hit criteria:** cosine similarity ≥ similarity_threshold.
    """
    svc = get_service()
    try:
        raw = svc.query(request.query)
    except Exception as exc:
        logger.exception("Query error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(exc)}",
        )

    # Parse results into typed models
    results = [SearchResult(**r) for r in raw["result"]]

    return QueryResponse(
        query=raw["query"],
        cache_hit=raw["cache_hit"],
        matched_query=raw.get("matched_query"),
        similarity_score=raw.get("similarity_score"),
        result=results,
        dominant_cluster=raw["dominant_cluster"],
        latency_ms=raw["latency_ms"],
    )


@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    tags=["Cache"],
    summary="Cache performance metrics",
)
async def cache_stats() -> CacheStatsResponse:
    """
    Return current cache performance metrics.

    - **hit_rate** = hit_count / (hit_count + miss_count)
    - **total_entries** = number of distinct queries stored
    """
    svc = get_service()
    stats = svc.cache_stats()
    return CacheStatsResponse(**stats)


@app.delete(
    "/cache",
    response_model=FlushResponse,
    tags=["Cache"],
    summary="Flush the semantic cache",
)
async def flush_cache() -> FlushResponse:
    """
    Remove all entries from the semantic cache and reset all counters.
    Useful for testing and threshold experiments.
    """
    svc = get_service()
    updated_stats = svc.flush_cache()
    return FlushResponse(
        message="Cache flushed successfully.",
        stats=CacheStatsResponse(**updated_stats),
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "error": str(exc)},
    )
