"""Retrieval application services."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapters import create_embedder, create_llm
from config import find_config_path, load_config
from pipelines import get_retrieval_pipeline
from pipelines.base import create_vector_store_from_config
from utils.eval_jobs import create_eval_job, update_eval_job

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalQueryResult:
    response: str
    context: list[Any]
    eval_job_id: str


class RetrievalUnavailableError(RuntimeError):
    """Raised when the retrieval pipeline cannot be used."""


def run_query(
    query: str, storage_dir: Path, custom_llm: Any | None = None
) -> RetrievalQueryResult:
    pipeline = require_pipeline()
    result = pipeline.query(query, llm_override=custom_llm)

    job_id = str(uuid.uuid4())
    create_eval_job(storage_dir, job_id, query)
    return RetrievalQueryResult(
        response=result["response"],
        context=result["context"],
        eval_job_id=job_id,
    )


def run_eval_job(
    storage_dir: Path,
    job_id: str,
    query: str,
    contexts: list[str],
    response: str,
    ground_truth: str | None,
) -> None:
    from evaluation.ragas_eval import get_evaluator

    try:
        evaluator = get_evaluator()
        scores = evaluator.evaluate_query(
            query, contexts, response, ground_truth=ground_truth
        )
        update_eval_job(storage_dir, job_id, status="complete", scores=scores)
    except Exception as exc:  # noqa: BLE001
        update_eval_job(storage_dir, job_id, status="error", error=str(exc))


def load_active_config() -> dict[str, Any]:
    return load_config(find_config_path())


def swap_runtime_config(
    config: dict[str, Any],
    config_path: Path,
    *,
    embedding: dict[str, str] | None = None,
    llm: dict[str, str] | None = None,
) -> tuple[str, str, str, str, bool]:
    pipeline = get_pipeline()
    requires_reindex = False

    # Build new components outside the lock
    new_llm = None
    if llm is not None:
        llm_section = config.get("llm", {})
        llm_kwargs: dict[str, Any] = {
            k: v for k, v in llm_section.items() if k not in ("provider", "model")
        }
        new_llm = create_llm(llm["provider"], model=llm["model"], **llm_kwargs)

    new_embedder = None
    new_vs = None
    if embedding is not None:
        embed_section = config.get("embedding", {})
        embed_kwargs: dict[str, Any] = {
            k: v for k, v in embed_section.items() if k not in ("provider", "model")
        }
        new_embedder = create_embedder(
            embedding["provider"], model=embedding["model"], **embed_kwargs
        )
        new_vs = create_vector_store_from_config(config, config_path, new_embedder)
        try:
            requires_reindex = new_vs.count == 0
        except Exception:
            requires_reindex = True

    # Swap components atomically under the lock
    with _PIPELINE_LOCK:
        if new_llm is not None:
            pipeline.llm = new_llm
        if new_embedder is not None:
            pipeline.embedder = new_embedder
        if new_vs is not None:
            pipeline.vector_store = new_vs

    return (
        getattr(pipeline.embedder, "provider", "unknown"),
        getattr(pipeline.embedder, "model", "unknown"),
        getattr(pipeline.llm, "provider", "unknown"),
        getattr(pipeline.llm, "model", "unknown"),
        requires_reindex,
    )


_PIPELINE: Any | None = None
_PIPELINE_LOCK = threading.Lock()
_PIPELINE_WARMER_LOCK = threading.Lock()
_PIPELINE_WARMER_STARTED = False
_PIPELINE_READY = threading.Event()
_PIPELINE_STATUS = "starting"
_PIPELINE_ERROR: str | None = None


def get_pipeline() -> Any:
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = get_retrieval_pipeline(find_config_path())
                _mark_pipeline_ready()
    return _PIPELINE


def require_pipeline() -> Any:
    if _PIPELINE is None:
        status, error = get_pipeline_status()
        message = error or "Retrieval pipeline is still starting"
        raise RetrievalUnavailableError(f"{status}: {message}")
    return _PIPELINE


def is_pipeline_loaded() -> bool:
    return _PIPELINE is not None


def get_pipeline_status() -> tuple[str, str | None]:
    return _PIPELINE_STATUS, _PIPELINE_ERROR


def _mark_pipeline_ready() -> None:
    global _PIPELINE_STATUS, _PIPELINE_ERROR
    _PIPELINE_STATUS = "healthy"
    _PIPELINE_ERROR = None
    _PIPELINE_READY.set()


def _mark_pipeline_degraded(error: str) -> None:
    global _PIPELINE_STATUS, _PIPELINE_ERROR
    _PIPELINE_STATUS = "degraded"
    _PIPELINE_ERROR = error


def warm_pipeline_once() -> bool:
    try:
        get_pipeline()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval pipeline warmup failed")
        _mark_pipeline_degraded(str(exc))
        return False


def start_pipeline_warmer(retry_interval: float = 5.0) -> None:
    global _PIPELINE_WARMER_STARTED

    with _PIPELINE_WARMER_LOCK:
        if _PIPELINE_WARMER_STARTED:
            return
        _PIPELINE_WARMER_STARTED = True

    def _loop() -> None:
        while not _PIPELINE_READY.is_set():
            if warm_pipeline_once():
                return
            _PIPELINE_READY.wait(retry_interval)

    thread = threading.Thread(
        target=_loop,
        name="retrieval-pipeline-warmer",
        daemon=True,
    )
    thread.start()
