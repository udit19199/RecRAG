"""FiNER-139 graph-construction benchmark router (research-only).

Live-runs the benchmark as a background task with in-memory run tracking and
polling. No DB or disk persistence (results reset on restart) - acceptable for
the small internal research audience. See src/experiments/finer139 for compute
and docs/research/findings/finer139-graph-construction.md for methodology.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import AuthContext, get_auth_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments/finer139", tags=["experiments"])

ALL_METHODS = ["llm", "nlp", "ontology", "hybrid", "dynamic"]
MAX_SAMPLE_SIZE = 500

# In-memory run registry (run_id -> record). Guarded by a lock because the
# background task writes from a worker thread while GET reads from the loop.
_RUNS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


class StartRunRequest(BaseModel):
    sample_size: int = Field(default=100, ge=1, le=MAX_SAMPLE_SIZE)
    seed: int = Field(default=42, ge=0)
    methods: list[str] = Field(default_factory=lambda: list(ALL_METHODS))
    # FiNER-139 defaults to OpenAI; UI may override per run.
    provider: str | None = "openai"
    model: str | None = "gpt-4o-mini"


class StartRunResponse(BaseModel):
    run_id: str
    status: str


class Progress(BaseModel):
    stage: str
    current: int
    total: int
    message: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str  # pending | running | complete | error
    progress: Progress | None = None
    params: dict[str, Any] | None = None
    results: dict[str, Any] | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update(run_id: str, **fields: Any) -> None:
    with _LOCK:
        rec = _RUNS.get(run_id)
        if rec is None:
            return
        rec.update(fields)
        rec["updated_at"] = _now()


def _run_task(
    run_id: str,
    sample_size: int,
    seed: int,
    methods: list[str],
    provider: str | None,
    model: str | None,
) -> None:
    _update(
        run_id,
        status="running",
        progress={
            "stage": "starting",
            "current": 0,
            "total": 0,
            "message": "Starting benchmark...",
        },
    )

    def progress(stage: str, current: int, total: int, message: str) -> None:
        _update(
            run_id,
            progress={
                "stage": stage,
                "current": current,
                "total": total,
                "message": message,
            },
        )

    try:
        # Lazy import so the core service never pulls datasets/spacy at import.
        from experiments.finer139.runner import RunParams, run_benchmark

        params = RunParams(
            sample_size=sample_size,
            seed=seed,
            methods=methods,
            provider=provider,
            model=model,
        )
        results = run_benchmark(params, progress)
        _update(
            run_id,
            status="complete",
            results=results,
            progress={
                "stage": "complete",
                "current": 1,
                "total": 1,
                "message": "Done",
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface to the client
        logger.exception("FiNER-139 run %s failed", run_id)
        _update(run_id, status="error", error=str(exc))


@router.post("/runs", response_model=StartRunResponse)
async def start_run(
    body: StartRunRequest,
    background_tasks: BackgroundTasks,
    _auth: AuthContext = Depends(get_auth_context),
) -> StartRunResponse:
    methods = [m for m in body.methods if m in ALL_METHODS] or list(ALL_METHODS)
    run_id = str(uuid.uuid4())
    with _LOCK:
        _RUNS[run_id] = {
            "run_id": run_id,
            "status": "pending",
            "progress": {
                "stage": "queued",
                "current": 0,
                "total": 0,
                "message": "Queued",
            },
            "params": {
                "sample_size": body.sample_size,
                "seed": body.seed,
                "methods": methods,
                "provider": body.provider,
                "model": body.model,
            },
            "results": None,
            "error": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
    background_tasks.add_task(
        _run_task,
        run_id,
        body.sample_size,
        body.seed,
        methods,
        body.provider,
        body.model,
    )
    return StartRunResponse(run_id=run_id, status="pending")


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(
    run_id: str,
    _auth: AuthContext = Depends(get_auth_context),
) -> RunStatusResponse:
    with _LOCK:
        rec = _RUNS.get(run_id)
        snapshot = dict(rec) if rec is not None else None
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatusResponse(**snapshot)
