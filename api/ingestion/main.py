import logging
import shutil
import threading
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import find_config_path, load_config
from pipelines import IngestionPipeline
from adapters import create_embedder
from stores import VectorStore
from pipelines.base import get_collection_name, get_milvus_uri
from utils.status import now, read_status, write_status

logger = logging.getLogger(__name__)

# ── Path resolution ───────────────────────────────────────────────────────────
# /app is the container root; outside Docker resolve from the repo root.
_APP_DIR = Path("/app")
_REPO_ROOT = _APP_DIR if _APP_DIR.exists() else Path(__file__).resolve().parents[2]

DATA_DIR = _REPO_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
STORAGE_DIR = _REPO_ROOT / "storage"
CONFIG_PATH = _REPO_ROOT / "config.toml"

# ── Pipeline (lazy-initialised once at first ingestion) ───────────────────────
_pipeline: Any = None
_pipeline_lock = threading.Lock()
_ingestion_lock = threading.Lock()


def _get_pipeline() -> Any:
    """Return the shared IngestionPipeline, building it on first call."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                config_path = find_config_path(
                    CONFIG_PATH if CONFIG_PATH.exists() else None
                )
                config = load_config(config_path)
                _pipeline = IngestionPipeline.from_config(config, config_path)
                logger.info("Ingestion pipeline initialised")
    return _pipeline


# ── Background ingestion tasks ────────────────────────────────────────────────


def _run_ingestion_background() -> None:
    """Run full-corpus ingestion and update the status file."""
    with _ingestion_lock:
        started_at = now()
        write_status(STORAGE_DIR, "processing", started_at=started_at)
        logger.info("Ingestion started for uploaded corpus")

        try:
            pipeline = _get_pipeline()
            results = pipeline.process_documents_streaming(force=True)
            completed_at = now()
            docs = results.get("documents", 0)
            write_status(
                STORAGE_DIR,
                "complete",
                started_at=started_at,
                completed_at=completed_at,
                files_processed=docs,
            )
            logger.info(
                "Ingestion complete — docs: %d, chunks: %d",
                docs,
                results.get("chunks", 0),
            )
        except Exception as exc:
            completed_at = now()
            logger.error("Ingestion failed: %s", exc)
            write_status(
                STORAGE_DIR,
                "error",
                started_at=started_at,
                completed_at=completed_at,
                error_message=str(exc),
            )


def _run_reindex_background() -> None:
    """Force re-index all documents using the current embedder configuration."""
    with _ingestion_lock:
        started_at = now()
        write_status(STORAGE_DIR, "processing", started_at=started_at)
        logger.info("Re-index started (force=True)")

        try:
            pipeline = _get_pipeline()
            results = pipeline.process_documents_streaming(force=True)
            completed_at = now()
            docs = results.get("documents", 0)
            write_status(
                STORAGE_DIR,
                "complete",
                started_at=started_at,
                completed_at=completed_at,
                files_processed=docs,
            )
            logger.info(
                "Re-index complete — docs: %d, chunks: %d",
                docs,
                results.get("chunks", 0),
            )
        except Exception as exc:
            completed_at = now()
            logger.error("Re-index failed: %s", exc)
            write_status(
                STORAGE_DIR,
                "error",
                started_at=started_at,
                completed_at=completed_at,
                error_message=str(exc),
            )


# ── Pydantic models ───────────────────────────────────────────────────────────


class StatusResponse(BaseModel):
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    files_processed: int | None = None
    error_message: str | None = None


class UploadResponse(BaseModel):
    success: bool
    files_uploaded: int
    message: str


class AdapterConfig(BaseModel):
    provider: str
    model: str


class EmbeddingConfigPatch(BaseModel):
    embedding: AdapterConfig


class ReindexResponse(BaseModel):
    started: bool
    message: str


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="RecRAG Ingestion API",
    description="API for uploading documents and checking ingestion status",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required")

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

    write_status(STORAGE_DIR, "idle", files_processed=0)
    background_tasks.add_task(_run_ingestion_background)

    return UploadResponse(
        success=True,
        files_uploaded=len(staged_files),
        message="Batch uploaded. Ingestion has started.",
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
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "ingestion-api"}


@app.post("/config")
async def set_config(patch: EmbeddingConfigPatch) -> dict[str, Any]:
    """Swap the embedding model used by the ingestion pipeline at runtime.

    Rebuilds the embedder and vector store on the existing pipeline singleton
    so subsequent uploads are indexed with the new model.
    """
    global _pipeline

    try:
        config_path = find_config_path(CONFIG_PATH if CONFIG_PATH.exists() else None)
        config = load_config(config_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")

    embed_section = config.get("embedding", {})
    embed_kwargs: dict[str, Any] = {
        k: v for k, v in embed_section.items() if k not in ("provider", "model")
    }

    try:
        new_embedder = create_embedder(
            patch.embedding.provider,
            model=patch.embedding.model,
            **embed_kwargs,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create embedder: {e}")

    new_collection = get_collection_name(config, new_embedder.model)
    uri = get_milvus_uri(config, config_path)
    try:
        new_vs = VectorStore(
            dimension=new_embedder.dimension,
            collection_name=new_collection,
            uri=uri,
            metric_type=config.get("storage", {}).get("metric_type", "L2"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initialise vector store: {e}"
        )

    with _pipeline_lock:
        pipeline = _get_pipeline()
        pipeline.embedder = new_embedder
        pipeline.vector_store = new_vs

    return {
        "applied": True,
        "embedding": {
            "provider": patch.embedding.provider,
            "model": patch.embedding.model,
        },
    }


@app.post("/reindex", response_model=ReindexResponse)
async def reindex(background_tasks: BackgroundTasks) -> ReindexResponse:
    """Trigger a full re-index of all previously uploaded documents.

    Rejected if ingestion is already running.
    """
    current = read_status(STORAGE_DIR)
    if current.get("status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="Ingestion already in progress. Please wait for it to complete.",
        )

    background_tasks.add_task(_run_reindex_background)

    return ReindexResponse(
        started=True,
        message="Re-indexing started. Poll /status for progress.",
    )
