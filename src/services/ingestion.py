"""Ingestion application services."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from adapters import create_embedder
from config import find_config_path, load_config
from pipelines import IngestionPipeline
from pipelines.base import create_vector_store_from_config
from utils._io import now
from utils.status import write_status

logger = logging.getLogger(__name__)

_PIPELINE: IngestionPipeline | None = None
_PIPELINE_LOCK = threading.Lock()
_INGESTION_LOCK = threading.Lock()


def get_pipeline(config_path: Path | None = None) -> IngestionPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                resolved = find_config_path(config_path)
                config = load_config(resolved)
                _PIPELINE = IngestionPipeline.from_config(config, resolved)
                logger.info("Ingestion pipeline initialised")
    return _PIPELINE


def run_ingestion_background(storage_dir: Path) -> None:
    with _INGESTION_LOCK:
        started_at = now()
        write_status(storage_dir, "processing", started_at=started_at)
        logger.info("Ingestion started")
        try:
            pipeline = get_pipeline()
            results = pipeline.process_documents_streaming(force=True)
            completed_at = now()
            write_status(
                storage_dir,
                "complete",
                started_at=started_at,
                completed_at=completed_at,
                files_processed=results.get("documents", 0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingestion failed")
            write_status(
                storage_dir,
                "error",
                started_at=started_at,
                completed_at=now(),
                error_message=str(exc),
            )


def run_reindex_background(storage_dir: Path) -> None:
    run_ingestion_background(storage_dir)


def swap_runtime_config(
    config: dict[str, Any],
    config_path: Path,
    *,
    embedding: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Swap the embedding model used by the ingestion pipeline.

    Returns (embed_provider, embed_model) after applying the change.
    """
    pipeline = get_pipeline()

    if embedding is None:
        return (
            getattr(pipeline.embedder, "provider", "unknown"),
            getattr(pipeline.embedder, "model", "unknown"),
        )

    # Build new components outside the lock
    embed_section = config.get("embedding", {})
    embed_kwargs: dict[str, Any] = {
        k: v for k, v in embed_section.items() if k not in ("provider", "model")
    }
    new_embedder = create_embedder(
        embedding["provider"], model=embedding["model"], **embed_kwargs
    )
    new_vs = create_vector_store_from_config(config, config_path, new_embedder)

    # Swap components atomically under the lock
    with _PIPELINE_LOCK:
        pipeline.embedder = new_embedder
        pipeline.vector_store = new_vs

    return (
        getattr(pipeline.embedder, "provider", "unknown"),
        getattr(pipeline.embedder, "model", "unknown"),
    )
