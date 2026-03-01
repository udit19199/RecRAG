from pathlib import Path
from typing import Any
import json


def get_status_file(storage_dir: Path) -> Path:
    """Return the ingestion status file path inside `storage_dir`."""
    return storage_dir / "ingestion_status.json"


def read_status(storage_dir: Path) -> dict[str, Any]:
    """Read and return the ingestion status JSON from `storage_dir`.

    Returns `{"status": "idle"}` when the file is missing or malformed.
    """
    status_file = get_status_file(storage_dir)
    if not status_file.exists():
        return {"status": "idle"}
    try:
        with open(status_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"status": "idle"}
