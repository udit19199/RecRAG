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
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    files_processed: int | None = None
    error_message: str | None = None


class UploadResponse(BaseModel):
    success: bool
    filename: str
    message: str


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


from src.utils.status import read_status


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile) -> UploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(file.filename).name
    file_path = PDF_DIR / safe_filename

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    return UploadResponse(
        success=True,
        filename=safe_filename,
        message="File uploaded successfully. It will be processed by the ingestion service.",
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
