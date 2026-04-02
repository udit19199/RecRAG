"""Shared runtime abstractions for pipeline lifecycle management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncIterator, Protocol, TypeVar

PipelineT = TypeVar("PipelineT")


class RuntimeState(str, Enum):
    """Lifecycle state for runtime-managed pipelines."""

    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RELOADING = "reloading"


class RuntimeUnavailableError(RuntimeError):
    """Raised when a runtime cannot currently provide a pipeline."""


class PipelineRuntime(Protocol[PipelineT]):
    """Protocol for runtime-managed pipeline access."""

    @property
    def state(self) -> RuntimeState: ...

    @property
    def error(self) -> str | None: ...

    def is_loaded(self) -> bool: ...

    @asynccontextmanager
    async def acquire(self, timeout_s: float = 2.0) -> AsyncIterator[PipelineT]: ...

    async def warm(self) -> None: ...

    async def shutdown(self, timeout_s: float = 30.0) -> None: ...


def close_pipeline_resources(pipeline: Any) -> None:
    """Best-effort close for known pipeline resources."""

    vector_store = getattr(pipeline, "vector_store", None)
    close = getattr(vector_store, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
