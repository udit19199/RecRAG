"""Runtime manager for retrieval pipeline lifecycle."""

from __future__ import annotations

import asyncio
import copy
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from collections.abc import AsyncIterator

from config import find_config_path, load_config
from pipelines.retrieval import RetrievalPipeline

from .base import (
    RuntimeState,
    RuntimeUnavailableError,
    close_pipeline_resources,
)

logger = logging.getLogger(__name__)


class RetrievalRuntime:
    """Own retrieval pipeline lifecycle, state, and concurrency."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = find_config_path(config_path)
        self._pipeline: RetrievalPipeline | None = None
        self._state = RuntimeState.STARTING
        self._error: str | None = None
        self._lock = asyncio.Lock()
        self._in_flight: dict[int, int] = {}
        self._drain_events: dict[int, asyncio.Event] = {}

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def get_active_config(self) -> tuple[dict[str, str], dict[str, str]]:
        """Return active embedder and llm provider/model pairs."""

        pipeline = self._pipeline
        if pipeline is None:
            config = load_config(self._config_path)
            embed_cfg = config.get("embedding", {})
            llm_cfg = config.get("llm", {})
            return (
                {
                    "provider": str(embed_cfg.get("provider", "gemini")),
                    "model": str(embed_cfg.get("model", "gemini-embedding-001")),
                },
                {
                    "provider": str(llm_cfg.get("provider", "gemini")),
                    "model": str(llm_cfg.get("model", "gemini-2.0-flash")),
                },
            )

        return (
            {
                "provider": str(getattr(pipeline.embedder, "provider", "gemini")),
                "model": str(
                    getattr(pipeline.embedder, "model", "gemini-embedding-001")
                ),
            },
            {
                "provider": str(getattr(pipeline.llm, "provider", "gemini")),
                "model": str(getattr(pipeline.llm, "model", "gemini-2.0-flash")),
            },
        )

    async def warm(self) -> None:
        """Load the default pipeline from config."""

        try:
            pipeline = await asyncio.to_thread(
                self._build_pipeline, None, None, None, None
            )
        except Exception as exc:
            self._state = RuntimeState.DEGRADED
            self._error = str(exc)
            print("DEBUG: Exception:", repr(exc))
            raise

        async with self._lock:
            self._pipeline = pipeline
            self._state = RuntimeState.HEALTHY
            self._error = None

    async def warm_with_retry(self, retry_interval: float = 5.0) -> None:
        """Keep warming until runtime becomes healthy."""

        while not self.is_loaded():
            try:
                await self.warm()
                return
            except Exception:
                logger.exception("Retrieval runtime warmup failed")
                await asyncio.sleep(retry_interval)

    async def reload(
        self,
        *,
        config: dict[str, Any],
        config_path: Path,
        embedding: dict[str, str] | None = None,
        llm: dict[str, str] | None = None,
    ) -> tuple[str, str, str, str, bool]:
        """Rebuild the active pipeline and atomically swap it in."""

        self._state = RuntimeState.RELOADING
        self._error = None

        try:
            new_pipeline = await asyncio.to_thread(
                self._build_pipeline, config, config_path, embedding, llm
            )
        except Exception as exc:
            self._state = RuntimeState.DEGRADED
            self._error = str(exc)
            print("DEBUG: Exception:", repr(exc))
            raise

        requires_reindex = False
        if embedding is not None:
            try:
                requires_reindex = new_pipeline.vector_store.count == 0
            except Exception:
                requires_reindex = True

        old_pipeline: RetrievalPipeline | None
        old_event: asyncio.Event | None
        async with self._lock:
            old_pipeline = self._pipeline
            old_id = id(old_pipeline) if old_pipeline is not None else None
            old_event = self._drain_events.get(old_id) if old_id is not None else None

            self._pipeline = new_pipeline
            self._state = RuntimeState.HEALTHY
            self._error = None

        if old_pipeline is not None:
            await self._drain_and_close(old_pipeline, old_event, timeout_s=30.0)

        return (
            getattr(new_pipeline.embedder, "provider", "unknown"),
            getattr(new_pipeline.embedder, "model", "unknown"),
            getattr(new_pipeline.llm, "provider", "unknown"),
            getattr(new_pipeline.llm, "model", "unknown"),
            requires_reindex,
        )

    @asynccontextmanager
    async def acquire(self, timeout_s: float = 2.0) -> AsyncIterator[RetrievalPipeline]:
        """Yield active pipeline while tracking in-flight readers."""

        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout_s)
        except TimeoutError as exc:
            raise RuntimeUnavailableError(
                "Timed out waiting for retrieval runtime"
            ) from exc

        pipeline = self._pipeline
        if pipeline is None:
            self._lock.release()
            message = self._error or "Retrieval pipeline is not ready"
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

    async def shutdown(self, timeout_s: float = 30.0) -> None:
        """Wait for in-flight requests and close resources."""

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

    def _build_pipeline(
        self,
        config: dict[str, Any] | None,
        config_path: Path | None,
        embedding: dict[str, str] | None,
        llm: dict[str, str] | None,
    ) -> RetrievalPipeline:
        resolved_path = find_config_path(config_path or self._config_path)
        source = config if config is not None else load_config(resolved_path)
        merged = copy.deepcopy(source)

        if embedding is not None:
            merged.setdefault("embedding", {})["provider"] = embedding["provider"]
            merged.setdefault("embedding", {})["model"] = embedding["model"]

        if llm is not None:
            merged.setdefault("llm", {})["provider"] = llm["provider"]
            merged.setdefault("llm", {})["model"] = llm["model"]

        return RetrievalPipeline.from_config(merged, resolved_path)

    async def _drain_and_close(
        self,
        pipeline: RetrievalPipeline,
        event: asyncio.Event | None,
        timeout_s: float,
    ) -> None:
        if event is not None:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_s)
            except TimeoutError:
                logger.warning("Timed out draining retrieval pipeline; forcing close")

        await asyncio.to_thread(close_pipeline_resources, pipeline)
