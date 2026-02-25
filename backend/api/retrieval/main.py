"""Retrieval API - FastAPI service for querying documents with RAG.

This API provides endpoints for:
- Querying the RAG pipeline with natural language
- Health checking

The API uses the shared storage volume to access the FAISS index
and communicates with the LLM service for generating responses.

Routes:
    - POST /query: Query the RAG pipeline
    - GET /health: Health check endpoint
"""

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add backend/src to path for imports
APP_DIR = Path("/app")
if APP_DIR.exists():
    sys.path.insert(0, str(APP_DIR / "backend" / "src"))
else:
    # Local development paths
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

from config import find_config_path, load_config
from pipelines import get_retrieval_pipeline


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str


class ContextItem(BaseModel):
    """A single retrieved context item."""

    text: str
    source: str
    distance: float


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    response: str
    context: list[ContextItem]


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str
    service: str
    pipeline_loaded: bool


app = FastAPI(
    title="RecRAG Retrieval API",
    description="API for querying documents using Retrieval-Augmented Generation",
    version="1.0.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy loaded)
_pipeline: Any = None


def get_pipeline() -> Any:
    """Get or initialize the retrieval pipeline.

    Returns:
        The RetrievalPipeline instance.

    Raises:
        HTTPException: If pipeline initialization fails.
    """
    global _pipeline
    if _pipeline is None:
        try:
            config_path = find_config_path()
            _pipeline = get_retrieval_pipeline(config_path)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize pipeline: {str(e)}",
            )
    return _pipeline


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Query the RAG pipeline with a question.

    The query is:
    1. Embedded using the configured embedding model
    2. Searched against the FAISS vector store
    3. Passed to the LLM with retrieved context to generate an answer

    Args:
        request: QueryRequest containing the question.

    Returns:
        QueryResponse with the LLM's answer and retrieved context.

    Raises:
        HTTPException: If query fails or pipeline not ready.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty",
        )

    try:
        pipeline = get_pipeline()
        result = pipeline.query(request.query)

        # Convert context items to response format
        context = [
            ContextItem(
                text=doc.text,
                source=doc.source,
                distance=doc.distance,
            )
            for doc in result["context"]
        ]

        return QueryResponse(
            response=result["response"],
            context=context,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}",
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns the health status of the service and whether
    the pipeline has been initialized.

    Returns:
        HealthResponse with service status.
    """
    pipeline_loaded = _pipeline is not None
    if not pipeline_loaded:
        try:
            # Try to load pipeline to verify it's ready
            get_pipeline()
            pipeline_loaded = True
        except Exception:
            pipeline_loaded = False

    return HealthResponse(
        status="healthy" if pipeline_loaded else "starting",
        service="retrieval-api",
        pipeline_loaded=pipeline_loaded,
    )
