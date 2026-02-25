"""Ingestion API - FastAPI service for document uploads and status.

This API provides endpoints for:
- Uploading PDF files to be ingested
- Checking ingestion status

The API writes files to the shared volume where the ingestion watcher
(pipeline) picks them up for processing.

Routes:
    - POST /upload: Upload a PDF file
    - GET /status: Get current ingestion status
    - GET /health: Health check endpoint
"""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration - paths relative to /app (Docker container root)
APP_DIR = Path("/app")
DATA_DIR = APP_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
STORAGE_DIR = APP_DIR / "storage"
CONFIG_PATH = APP_DIR / "config.toml"

# Local development paths (when running outside Docker)
if not APP_DIR.exists():
    APP_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = APP_DIR / "data"
    PDF_DIR = DATA_DIR / "pdfs"
    STORAGE_DIR = APP_DIR / "storage"
    CONFIG_PATH = APP_DIR / "config.toml"


class StatusResponse(BaseModel):
    """Response model for status endpoint."""

    status: str
    started_at: str | None = None
    completed_at: str | None = None
    files_processed: int | None = None
    error_message: str | None = None


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""

    success: bool
    filename: str
    message: str


app = FastAPI(
    title="RecRAG Ingestion API",
    description="API for uploading documents and checking ingestion status",
    version="1.0.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_status_file() -> Path:
    """Get the path to the ingestion status file."""
    return STORAGE_DIR / "ingestion_status.json"


def read_status() -> dict[str, Any]:
    """Read the current ingestion status from file."""
    status_file = get_status_file()
    if not status_file.exists():
        return {"status": "idle"}
    try:
        with open(status_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"status": "idle"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile) -> UploadResponse:
    """Upload a PDF file for ingestion.

    The file is saved to the shared volume where the ingestion
    watcher will pick it up and process it.

    Args:
        file: The PDF file to upload.

    Returns:
        UploadResponse with success status and filename.

    Raises:
        HTTPException: If the file is not a PDF or upload fails.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
        )

    # Ensure directory exists
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize filename and save
    safe_filename = Path(file.filename).name
    file_path = PDF_DIR / safe_filename

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}",
        )

    return UploadResponse(
        success=True,
        filename=safe_filename,
        message="File uploaded successfully. It will be processed by the ingestion service.",
    )


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get the current ingestion status.

    Returns the status of the ingestion process including:
    - idle: No ingestion in progress
    - processing: Currently processing files
    - complete: Successfully processed
    - error: Failed with an error

    Returns:
        StatusResponse with current status details.
    """
    status = read_status()
    return StatusResponse(
        status=status.get("status", "idle"),
        started_at=status.get("started_at"),
        completed_at=status.get("completed_at"),
        files_processed=status.get("files_processed"),
        error_message=status.get("error_message"),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Dictionary with health status.
    """
    return {"status": "healthy", "service": "ingestion-api"}
