import logging
import os
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import find_config_path, get_frontend_origins, load_config
from utils.status import read_status, write_status
from services.ingestion import (
    run_ingestion_background,
    run_reindex_background,
    swap_runtime_config,
)
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

logger = logging.getLogger(__name__)

_IS_PRODUCTION = os.getenv("ENVIRONMENT", "").lower() == "production"

_APP_DIR = Path("/app")
_REPO_ROOT = _APP_DIR if _APP_DIR.exists() else Path(__file__).resolve().parents[2]

DATA_DIR = _REPO_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
STORAGE_DIR = _REPO_ROOT / "storage"
CONFIG_PATH = _REPO_ROOT / "config.toml"


# ── FastAPI app ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Clear old ingestion status on startup."""
    from utils.status import get_status_file

    status_file = get_status_file(STORAGE_DIR)
    if status_file.exists():
        try:
            status_file.unlink()
        except OSError:
            pass
    yield


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
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    extraction_mode: str = Form(default="text_only"),
    vision_provider: str | None = Form(default=None),
    vision_model: str | None = Form(default=None),
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

    current = read_status(STORAGE_DIR)
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
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        safe_filename = Path(upload.filename).name
        if safe_filename in seen_names:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate filename in batch: {safe_filename}",
            )

        try:
            content = await upload.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

        staged_files.append((safe_filename, content))
        seen_names.add(safe_filename)

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    try:
        shutil.rmtree(PDF_DIR)
        PDF_DIR.mkdir(parents=True, exist_ok=True)

        for safe_filename, content in staged_files:
            file_path = PDF_DIR / safe_filename
            with open(file_path, "wb") as f:
                f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save batch: {e}")

    write_status(STORAGE_DIR, "idle", files_processed=0, extraction_mode=mode.value)
    background_tasks.add_task(
        run_ingestion_background,
        STORAGE_DIR,
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
    status = read_status(STORAGE_DIR)
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


@app.get("/files", response_model=FileListResponse)
async def list_files() -> FileListResponse:
    """Return a list of uploaded PDF files."""
    files = []
    if PDF_DIR.exists():
        for file_path in PDF_DIR.glob("*.pdf"):
            if file_path.is_file():
                files.append(file_path.name)
    return FileListResponse(files=sorted(files))


@app.post("/config")
async def set_config(patch: EmbeddingConfigPatch) -> dict[str, Any]:
    """Swap the embedding model used by the ingestion pipeline at runtime.

    Rebuilds the embedder and vector store on the existing pipeline singleton
    so subsequent uploads are indexed with the new model.
    """
    try:
        config_path = find_config_path(CONFIG_PATH if CONFIG_PATH.exists() else None)
        config = load_config(config_path)
        embed_provider, embed_model = swap_runtime_config(
            config,
            config_path,
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
async def check_index_status(request: IndexStatusRequest) -> IndexStatusResponse:
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
) -> ReindexResponse:
    """Trigger a targeted ingest for a specific permutation."""
    current = read_status(STORAGE_DIR)
    if current.get("status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="Ingestion already in progress. Please wait for it to complete.",
        )

    # Use run_reindex_background but we need to pass embedding overrides
    from services.ingestion import run_targeted_ingestion_background

    background_tasks.add_task(
        run_targeted_ingestion_background,
        STORAGE_DIR,
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
) -> ReindexResponse:
    """Trigger a full re-index of all previously uploaded documents.

    Args:
        request: Optional request body with extraction_mode and vision settings.

    Rejected if ingestion is already running.
    """
    current = read_status(STORAGE_DIR)
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
        run_reindex_background,
        STORAGE_DIR,
        extraction_mode=mode,
        vision_provider=vision_provider,
        vision_model=vision_model,
    )

    return ReindexResponse(
        started=True,
        message=f"Re-indexing started (mode={mode.value}). Poll /status for progress.",
        extraction_mode=mode.value,
    )
