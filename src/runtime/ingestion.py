"""Runtime manager for ingestion pipeline lifecycle."""

from __future__ import annotations

import asyncio
import copy
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from collections.abc import AsyncIterator

from config import find_config_path, load_config
from models.api import ExtractionMode
from pipelines.ingestion import IngestionPipeline
from utils._io import now
from utils.status import write_status

from .base import RuntimeState, RuntimeUnavailableError, close_pipeline_resources

logger = logging.getLogger(__name__)


class IngestionRuntime:
    """Own ingestion pipeline lifecycle, defaults, and ingestion locking."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = find_config_path(config_path)
        self._pipeline: IngestionPipeline | None = None
        self._state = RuntimeState.STARTING
        self._error: str | None = None
        self._lock = asyncio.Lock()
        self._ingestion_lock = asyncio.Lock()
        self._in_flight: dict[int, int] = {}
        self._drain_events: dict[int, asyncio.Event] = {}
        self._embedding_override: dict[str, str] | None = None

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    async def warm(self) -> None:
        try:
            pipeline = await asyncio.to_thread(
                self._build_pipeline,
                None,
                None,
                ExtractionMode.TEXT_ONLY,
                None,
                None,
                None,
                None,
            )
        except Exception as exc:
            self._state = RuntimeState.DEGRADED
            self._error = str(exc)
            raise

        async with self._lock:
            self._pipeline = pipeline
            self._state = RuntimeState.HEALTHY
            self._error = None

    async def update_embedding_default(
        self,
        *,
        config: dict[str, Any],
        config_path: Path,
        embedding: dict[str, str],
    ) -> tuple[str, str]:
        """Apply new default embedding for future ingestion jobs."""

        self._state = RuntimeState.RELOADING
        self._error = None

        try:
            pipeline = await asyncio.to_thread(
                self._build_pipeline,
                config,
                config_path,
                ExtractionMode.TEXT_ONLY,
                None,
                None,
                embedding["provider"],
                embedding["model"],
            )
        except Exception as exc:
            self._state = RuntimeState.DEGRADED
            self._error = str(exc)
            raise

        old_pipeline: IngestionPipeline | None
        old_event: asyncio.Event | None
        async with self._lock:
            old_pipeline = self._pipeline
            old_id = id(old_pipeline) if old_pipeline is not None else None
            old_event = self._drain_events.get(old_id) if old_id is not None else None
            self._pipeline = pipeline
            self._embedding_override = {
                "provider": embedding["provider"],
                "model": embedding["model"],
            }
            self._state = RuntimeState.HEALTHY
            self._error = None

        if old_pipeline is not None:
            await self._drain_and_close(old_pipeline, old_event, timeout_s=30.0)

        return (
            getattr(pipeline.embedder, "provider", embedding["provider"]),
            getattr(pipeline.embedder, "model", embedding["model"]),
        )

    @asynccontextmanager
    async def acquire(self, timeout_s: float = 2.0) -> AsyncIterator[IngestionPipeline]:
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout_s)
        except TimeoutError as exc:
            raise RuntimeUnavailableError(
                "Timed out waiting for ingestion runtime"
            ) from exc

        pipeline = self._pipeline
        if pipeline is None:
            self._lock.release()
            message = self._error or "Ingestion pipeline is not ready"
            raise RuntimeUnavailableError(f"{self._state.value}: {message}")

        pipeline_id = id(pipeline)
        self._in_flight[pipeline_id] = self._in_flight.get(pipeline_id, 0) + 1
        event = self._drain_events.get(pipeline_id)
        if event is None:
            event = asyncio.Event()
            self._drain_events[pipeline_id] = event
        event.clear()
        self._lock.release()

        try:
            yield pipeline
        finally:
            async with self._lock:
                current = self._in_flight.get(pipeline_id, 0)
                if current <= 1:
                    self._in_flight.pop(pipeline_id, None)
                    done_event = self._drain_events.pop(pipeline_id, None)
                    if done_event is not None:
                        done_event.set()
                else:
                    self._in_flight[pipeline_id] = current - 1

    async def run_ingestion(
        self,
        storage_dir: Path,
        *,
        extraction_mode: ExtractionMode = ExtractionMode.TEXT_ONLY,
        vision_provider: str | None = None,
        vision_model: str | None = None,
    ) -> None:
        """Run ingestion with runtime defaults and provided mode overrides."""

        await self._run_ingestion_internal(
            storage_dir,
            extraction_mode=extraction_mode,
            vision_provider=vision_provider,
            vision_model=vision_model,
            embedding_provider=None,
            embedding_model=None,
            force=True,
            log_name="Ingestion",
        )

    async def run_targeted_ingestion(
        self,
        storage_dir: Path,
        *,
        extraction_mode: ExtractionMode = ExtractionMode.TEXT_ONLY,
        vision_provider: str | None = None,
        vision_model: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Run targeted ingest for a specific embedding/vision permutation."""

        await self._run_ingestion_internal(
            storage_dir,
            extraction_mode=extraction_mode,
            vision_provider=vision_provider,
            vision_model=vision_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            force=True,
            log_name="Targeted ingestion",
        )

    async def shutdown(self, timeout_s: float = 30.0) -> None:
        async with self._lock:
            pipeline = self._pipeline
            pipeline_id = id(pipeline) if pipeline is not None else None
            event = (
                self._drain_events.get(pipeline_id) if pipeline_id is not None else None
            )
            self._pipeline = None
            self._state = RuntimeState.STARTING
            self._error = None

        if pipeline is not None:
            await self._drain_and_close(pipeline, event, timeout_s)

        # Clean up any dev-mode temp directories
        from stores import cleanup_dev_temp_dirs

        await asyncio.to_thread(cleanup_dev_temp_dirs)

    async def _run_ingestion_internal(
        self,
        storage_dir: Path,
        *,
        extraction_mode: ExtractionMode,
        vision_provider: str | None,
        vision_model: str | None,
        embedding_provider: str | None,
        embedding_model: str | None,
        force: bool,
        log_name: str,
    ) -> None:
        async with self._ingestion_lock:
            started_at = now()
            write_status(
                storage_dir,
                "processing",
                started_at=started_at,
                extraction_mode=extraction_mode.value,
            )
            logger.info("%s started (mode=%s)", log_name, extraction_mode.value)

            try:
                results = await asyncio.to_thread(
                    self._execute_ingestion,
                    extraction_mode,
                    vision_provider,
                    vision_model,
                    embedding_provider,
                    embedding_model,
                    force,
                )
                completed_at = now()
                write_status(
                    storage_dir,
                    "complete",
                    started_at=started_at,
                    completed_at=completed_at,
                    files_processed=results.get("documents", 0),
                    extraction_mode=extraction_mode.value,
                )
            except Exception as exc:
                logger.exception("%s failed", log_name)
                write_status(
                    storage_dir,
                    "error",
                    started_at=started_at,
                    completed_at=now(),
                    error_message=str(exc),
                    extraction_mode=extraction_mode.value,
                )

    def _execute_ingestion(
        self,
        extraction_mode: ExtractionMode,
        vision_provider: str | None,
        vision_model: str | None,
        embedding_provider: str | None,
        embedding_model: str | None,
        force: bool,
    ) -> dict[str, Any]:
        config_path = self._config_path
        config = load_config(config_path)

        if embedding_provider and embedding_model:
            config.setdefault("embedding", {})["provider"] = embedding_provider
            config.setdefault("embedding", {})["model"] = embedding_model
        elif self._embedding_override is not None:
            config.setdefault("embedding", {})["provider"] = self._embedding_override[
                "provider"
            ]
            config.setdefault("embedding", {})["model"] = self._embedding_override[
                "model"
            ]

        pipeline = IngestionPipeline.from_config(
            config,
            config_path,
            extraction_mode=extraction_mode,
            vision_provider=vision_provider,
            vision_model=vision_model,
        )

        return pipeline.process_documents_streaming(force=force)

    def _build_pipeline(
        self,
        config: dict[str, Any] | None,
        config_path: Path | None,
        extraction_mode: ExtractionMode,
        vision_provider: str | None,
        vision_model: str | None,
        embedding_provider: str | None,
        embedding_model: str | None,
    ) -> IngestionPipeline:
        resolved_path = find_config_path(config_path or self._config_path)
        source = config if config is not None else load_config(resolved_path)
        merged = copy.deepcopy(source)

        if embedding_provider and embedding_model:
            merged.setdefault("embedding", {})["provider"] = embedding_provider
            merged.setdefault("embedding", {})["model"] = embedding_model

        return IngestionPipeline.from_config(
            merged,
            resolved_path,
            extraction_mode=extraction_mode,
            vision_provider=vision_provider,
            vision_model=vision_model,
        )

    async def _drain_and_close(
        self,
        pipeline: IngestionPipeline,
        event: asyncio.Event | None,
        timeout_s: float,
    ) -> None:
        if event is not None:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_s)
            except TimeoutError:
                logger.warning("Timed out draining ingestion pipeline; forcing close")

        await asyncio.to_thread(close_pipeline_resources, pipeline)
