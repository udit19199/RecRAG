"""Shared runtime abstractions for pipeline lifecycle management."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, TypeVar
from collections.abc import AsyncIterator

PipelineT = TypeVar("PipelineT", covariant=True)


class RuntimeState(StrEnum):
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
