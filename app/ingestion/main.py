import logging

import asyncio

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from anyio import open_file
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from auth import verify_api_key
from config import find_config_path, get_frontend_origins, load_config
from structured_logging import StructuredLoggingMiddleware
from metrics import MetricsMiddleware, metrics_endpoint
from models.api import (
    EmbeddingConfigPatch,
    ExtractionMode,
    FileListResponse,
    ReindexRequest,
    ReindexResponse,
    StatusResponse,
    UploadResponse,
    IndexStatusRequest,
    IndexStatusResponse,
    TargetedIngestRequest,
)
from runtime import IngestionRuntime
from utils.status import read_status, write_status

logger = logging.getLogger(__name__)

_IS_PRODUCTION = os.getenv("ENVIRONMENT", "").lower() == "production"

_APP_DIR = Path("/app")


def _reset_directory(path: Path) -> None:
    """Remove all contents of a directory and recreate it (called via asyncio.to_thread)."""
    import shutil

    if path.exists():
        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)


_REPO_ROOT = _APP_DIR if _APP_DIR.exists() else Path(__file__).resolve().parents[2]

DATA_DIR = _REPO_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
STATE_DIR = _REPO_ROOT / "state"
CONFIG_PATH = _REPO_ROOT / "config.toml"


# ── FastAPI app ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Clear old ingestion status on startup."""
    from utils.status import get_status_file

    runtime = IngestionRuntime()
    app_instance.state.ingestion_runtime = runtime

    status_file = get_status_file(STATE_DIR)
    if status_file.exists():
        try:
            status_file.unlink()
        except OSError:
            pass
    await runtime.warm()
    yield
    await runtime.shutdown()


def get_ingestion_runtime(request: Request) -> IngestionRuntime:
    runtime = getattr(request.app.state, "ingestion_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Ingestion runtime unavailable")
    return runtime


logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")

app = FastAPI(
    title="RecRAG Ingestion API",
    description="API for uploading documents and checking ingestion status",
    version="1.0.0",
    docs_url=None if _IS_PRODUCTION else "/docs",
    redoc_url=None if _IS_PRODUCTION else "/redoc",
    openapi_url=None if _IS_PRODUCTION else "/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_frontend_origins(
        load_config(find_config_path(CONFIG_PATH if CONFIG_PATH.exists() else None))
    ),
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "RecRAG-API-Key"],
    expose_headers=["X-Request-ID"],
)

# Add structured logging middleware
app.add_middleware(StructuredLoggingMiddleware)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)


@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    runtime: IngestionRuntime = Depends(get_ingestion_runtime),
    files: list[UploadFile] = File(...),
    extraction_mode: str = Form(default="text_only"),
    vision_provider: str | None = Form(default=None),
    vision_model: str | None = Form(default=None),
    _: None = Depends(verify_api_key),
) -> UploadResponse:
    """Upload PDF files for ingestion.

    Args:
        files: PDF files to upload.
        extraction_mode: "text_only" (default) or "vision_assisted".
        vision_provider: Vision provider (openai, ollama, nim) when using vision mode.
        vision_model: Vision model name when using vision mode.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required")

    # Validate extraction mode
    try:
        mode = ExtractionMode(extraction_mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid extraction_mode: {extraction_mode}. "
            f"Must be one of: {[e.value for e in ExtractionMode]}",
        )

    current = read_status(STATE_DIR)
    if current.get("status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="Ingestion already in progress. Please wait for it to complete.",
        )

    staged_files: list[tuple[str, bytes]] = []
    seen_names: set[str] = set()
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        filename = upload.filename

        if filename in seen_names:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate filename in batch: {filename}",
            )

        try:
            content = await upload.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

        staged_files.append((filename, content))
        seen_names.add(filename)

    try:
        # Async-safe directory reset: remove and recreate using asyncio.to_thread
        await asyncio.to_thread(PDF_DIR.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(_reset_directory, PDF_DIR)

        for filename, content in staged_files:
            file_path = PDF_DIR / filename
            async with await open_file(file_path, "wb") as f:
                await f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save batch: {e}")

    write_status(STATE_DIR, "idle", files_processed=0, extraction_mode=mode.value)
    background_tasks.add_task(
        runtime.run_ingestion,
        STATE_DIR,
        extraction_mode=mode,
        vision_provider=vision_provider,
        vision_model=vision_model,
    )

    return UploadResponse(
        success=True,
        files_uploaded=len(staged_files),
        message=f"Batch uploaded. Ingestion has started (mode={mode.value}).",
        extraction_mode=mode.value,
    )


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    status = read_status(STATE_DIR)
    return StatusResponse(
        status=status.get("status", "idle"),
        started_at=status.get("started_at"),
        completed_at=status.get("completed_at"),
        files_processed=status.get("files_processed"),
        error_message=status.get("error_message"),
        extraction_mode=status.get("extraction_mode"),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "ingestion-api"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return metrics_endpoint()


@app.get("/files", response_model=FileListResponse)
async def list_files() -> FileListResponse:
    """Return a list of uploaded PDF files."""
    files = []
    if PDF_DIR.exists():
        for file_path in PDF_DIR.glob("*.pdf"):
            if file_path.is_file():
                files.append(file_path.name)
    return FileListResponse(files=sorted(files))


@app.delete("/documents/{filename}")
async def delete_document(
    filename: str,
    _: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete an uploaded PDF and remove its vectors from the store.

    Args:
        filename: The PDF filename to delete (e.g. "report.pdf").
    """
    file_path = PDF_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Delete from filesystem
    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

    # Delete from vector store
    config_path = find_config_path(CONFIG_PATH if CONFIG_PATH.exists() else None)
    config = load_config(config_path)

    from pipelines.base import (
        create_embedder_from_config,
        create_vector_store_from_config,
    )

    try:
        embedder = create_embedder_from_config(config)
        vector_store = create_vector_store_from_config(config, config_path, embedder)
        deleted = vector_store.delete_document(filename)
    except Exception as e:
        logger.warning("Vector store cleanup failed for %s: %s", filename, e)
        deleted = 0

    return {
        "deleted": True,
        "filename": filename,
        "vectors_removed": deleted,
    }


@app.post("/config")
async def set_config(
    patch: EmbeddingConfigPatch,
    _: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Swap the default embedding used by future ingestion jobs."""
    try:
        config_path = find_config_path(CONFIG_PATH if CONFIG_PATH.exists() else None)
        config = load_config(config_path)
        runtime = app.state.ingestion_runtime
        embed_provider, embed_model = await runtime.update_embedding_default(
            config=config,
            config_path=config_path,
            embedding={
                "provider": patch.embedding.provider,
                "model": patch.embedding.model,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {e}")

    return {
        "applied": True,
        "embedding": {"provider": embed_provider, "model": embed_model},
    }


@app.post("/status/index", response_model=IndexStatusResponse)
async def check_index_status(
    request: IndexStatusRequest,
    _: None = Depends(verify_api_key),
) -> IndexStatusResponse:
    from pipelines.base import get_collection_name, get_milvus_uri
    from stores import VectorStore

    config_path = find_config_path(CONFIG_PATH if CONFIG_PATH.exists() else None)
    config = load_config(config_path)

    embed_model = (
        request.embedding.model
        if request.embedding
        else config.get("embedding", {}).get("model", "text-embedding-3-small")
    )
    vision_model = request.vision.model if request.vision else None

    collection_name = get_collection_name(config, embed_model, vision_model)
    uri = get_milvus_uri(config, config_path)

    try:
        vs = VectorStore(
            dimension=1,  # Dimension doesn't matter just to check count
            collection_name=collection_name,
            uri=uri,
        )
        return IndexStatusResponse(has_documents=vs.count > 0)
    except Exception as e:
        logger.warning(f"Error checking index status: {e}")
        return IndexStatusResponse(has_documents=False)


@app.post("/ingest/target", response_model=ReindexResponse)
async def targeted_ingest(
    background_tasks: BackgroundTasks,
    request: TargetedIngestRequest,
    runtime: IngestionRuntime = Depends(get_ingestion_runtime),
    _: None = Depends(verify_api_key),
) -> ReindexResponse:
    """Trigger a targeted ingest for a specific permutation."""
    current = read_status(STATE_DIR)
    if current.get("status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="Ingestion already in progress. Please wait for it to complete.",
        )

    background_tasks.add_task(
        runtime.run_targeted_ingestion,
        STATE_DIR,
        extraction_mode=request.extraction_mode,
        vision_provider=request.vision_provider,
        vision_model=request.vision_model,
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
    )

    return ReindexResponse(
        started=True,
        message="Targeted ingestion started. Poll /status for progress.",
        extraction_mode=request.extraction_mode.value,
    )


@app.post("/reindex", response_model=ReindexResponse)
async def reindex(
    background_tasks: BackgroundTasks,
    request: ReindexRequest | None = None,
    runtime: IngestionRuntime = Depends(get_ingestion_runtime),
    _: None = Depends(verify_api_key),
) -> ReindexResponse:
    """Trigger a full re-index of all previously uploaded documents.

    Args:
        request: Optional request body with extraction_mode and vision settings.

    Rejected if ingestion is already running.
    """
    current = read_status(STATE_DIR)
    if current.get("status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="Ingestion already in progress. Please wait for it to complete.",
        )

    # Use defaults if no request body provided
    mode = request.extraction_mode if request else ExtractionMode.TEXT_ONLY
    vision_provider = request.vision_provider if request else None
    vision_model = request.vision_model if request else None

    background_tasks.add_task(
        runtime.run_ingestion,
        STATE_DIR,
        extraction_mode=mode,
        vision_provider=vision_provider,
        vision_model=vision_model,
    )

    return ReindexResponse(
        started=True,
        message=f"Re-indexing started (mode={mode.value}). Poll /status for progress.",
        extraction_mode=mode.value,
    )
