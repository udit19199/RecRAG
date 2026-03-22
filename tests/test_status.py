from pathlib import Path
from typing import Any

import pytest

from utils import _io as io_utils
from utils import status as status_utils


def test_write_status_is_atomic(tmp_path: Path) -> None:
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    status_file = storage_dir / "ingestion_status.json"
    status_file.write_text('{"status": "idle"}')

    def fake_dump(data: Any, fp: Any, indent: int = 2, **kwargs: Any) -> None:
        assert indent == 2
        fp.write('{"status": "processing"')
        raise RuntimeError("simulated write failure")

    original_dump = io_utils.json.dump
    io_utils.json.dump = fake_dump  # type: ignore[assignment]

    try:
        with pytest.raises(RuntimeError, match="simulated write failure"):
            status_utils.write_status(storage_dir, "processing")
    finally:
        io_utils.json.dump = original_dump  # type: ignore[assignment]

    assert status_file.read_text() == '{"status": "idle"}'


def test_write_status_persists_valid_json(tmp_path: Path) -> None:
    storage_dir = tmp_path / "storage"

    status_utils.write_status(
        storage_dir,
        "complete",
        started_at="2026-01-01T00:00:00",
        completed_at="2026-01-01T00:01:00",
        files_processed=3,
    )

    status_file = storage_dir / "ingestion_status.json"
    assert status_file.exists()
    assert status_utils.read_status(storage_dir)["status"] == "complete"
