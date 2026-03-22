"""Durable storage for async evaluation jobs."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from utils._io import atomic_write_json, now


_LOCK = threading.Lock()
_FILENAME = "evaluation_jobs.json"


def get_eval_jobs_file(storage_dir: Path) -> Path:
    return storage_dir / _FILENAME


def _read_eval_jobs_unlocked(storage_dir: Path) -> dict[str, dict[str, Any]]:
    """Read jobs file. Caller must hold _LOCK or be in a context where consistency is not required."""
    jobs_file = get_eval_jobs_file(storage_dir)
    if not jobs_file.exists():
        return {}

    try:
        with jobs_file.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    return {
        str(job_id): value for job_id, value in data.items() if isinstance(value, dict)
    }


def read_eval_jobs(storage_dir: Path) -> dict[str, dict[str, Any]]:
    """Read eval jobs with lock protection for thread-safe access."""
    with _LOCK:
        return _read_eval_jobs_unlocked(storage_dir)


def _write_eval_jobs_unlocked(
    storage_dir: Path, jobs: dict[str, dict[str, Any]]
) -> None:
    """Write jobs file. Caller must hold _LOCK."""
    jobs_file = get_eval_jobs_file(storage_dir)
    atomic_write_json(jobs_file, jobs)


def write_eval_jobs(storage_dir: Path, jobs: dict[str, dict[str, Any]]) -> None:
    with _LOCK:
        _write_eval_jobs_unlocked(storage_dir, jobs)


def create_eval_job(storage_dir: Path, job_id: str, query: str) -> dict[str, Any]:
    with _LOCK:
        jobs = _read_eval_jobs_unlocked(storage_dir)
        job = {
            "id": job_id,
            "status": "pending",
            "query": query,
            "created_at": now(),
            "updated_at": now(),
            "scores": None,
            "error": None,
        }
        jobs[job_id] = job
        _write_eval_jobs_unlocked(storage_dir, jobs)
        return job


def update_eval_job(
    storage_dir: Path,
    job_id: str,
    *,
    status: str,
    scores: dict[str, float] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    with _LOCK:
        jobs = _read_eval_jobs_unlocked(storage_dir)
        job = jobs.get(job_id, {"id": job_id, "created_at": now()})
        job.update(
            {
                "status": status,
                "updated_at": now(),
                "scores": scores,
                "error": error,
            }
        )
        jobs[job_id] = job
        _write_eval_jobs_unlocked(storage_dir, jobs)
        return job
