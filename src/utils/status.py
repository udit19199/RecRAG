import json
from pathlib import Path
from typing import Any

from ._io import atomic_write_json, now


def get_status_file(storage_dir: Path) -> Path:
    """Return the ingestion status file path inside ``storage_dir``."""
    return storage_dir / "ingestion_status.json"


def read_status(storage_dir: Path) -> dict[str, Any]:
    """Read ingestion status JSON. Returns ``{"status": "idle"}`` when missing or malformed."""
    status_file = get_status_file(storage_dir)
    if not status_file.exists():
        return {"status": "idle"}
    try:
        with status_file.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"status": "idle"}


def write_status(
    storage_dir: Path,
    status: str,
    started_at: str | None = None,
    completed_at: str | None = None,
    files_processed: int = 0,
    error_message: str | None = None,
    extraction_mode: str | None = None,
) -> None:
    """Write ingestion status JSON to ``storage_dir``."""
    data: dict[str, Any] = {
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "files_processed": files_processed,
        "error_message": error_message,
        "extraction_mode": extraction_mode,
    }
    atomic_write_json(get_status_file(storage_dir), data)


# Export now for backwards compatibility
__all__ = ["get_status_file", "now", "read_status", "write_status"]
