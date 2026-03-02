import json
from datetime import datetime
from pathlib import Path
from typing import Any


def get_status_file(storage_dir: Path) -> Path:
    """Return the ingestion status file path inside ``storage_dir``."""
    return storage_dir / "ingestion_status.json"


def read_status(storage_dir: Path) -> dict[str, Any]:
    """Read ingestion status JSON. Returns ``{"status": "idle"}`` when missing or malformed."""
    status_file = get_status_file(storage_dir)
    if not status_file.exists():
        return {"status": "idle"}
    try:
        with open(status_file) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"status": "idle"}


def write_status(
    storage_dir: Path,
    status: str,
    started_at: str | None = None,
    completed_at: str | None = None,
    files_processed: int = 0,
    error_message: str | None = None,
) -> None:
    """Write ingestion status JSON to ``storage_dir``."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "files_processed": files_processed,
        "error_message": error_message,
    }
    with open(get_status_file(storage_dir), "w") as f:
        json.dump(data, f, indent=2)


def now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now().isoformat()
