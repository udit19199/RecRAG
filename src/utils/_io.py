import json
import os
import tempfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


def now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def atomic_write_json(file_path: Path, data: Any) -> None:
    """Write JSON data atomically to the specified file path."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=file_path.parent,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        json.dump(data, tmp_file, indent=2, sort_keys=True)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())

    try:
        if tmp_path is None:
            raise RuntimeError("Temporary file was not created")
        os.replace(tmp_path, file_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
