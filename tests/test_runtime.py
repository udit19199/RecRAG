import asyncio

import pytest

from runtime import RuntimeState, RuntimeUnavailableError
from runtime.retrieval import RetrievalRuntime


class _DummyVectorStore:
    def __init__(self) -> None:
        self.count = 1
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _DummyPipeline:
    def __init__(self) -> None:
        self.embedder = type("Embedder", (), {"provider": "openai", "model": "e1"})()
        self.llm = type("LLM", (), {"provider": "openai", "model": "gpt"})()
        self.vector_store = _DummyVectorStore()


def test_acquire_times_out_under_lock() -> None:
    runtime = RetrievalRuntime()
    runtime._pipeline = _DummyPipeline()  # type: ignore[attr-defined]
    runtime._state = RuntimeState.HEALTHY  # type: ignore[attr-defined]

    async def _exercise() -> None:
        await runtime._lock.acquire()  # type: ignore[attr-defined]
        try:
            with pytest.raises(RuntimeUnavailableError):
                async with runtime.acquire(timeout_s=0.01):
                    pass
        finally:
            runtime._lock.release()  # type: ignore[attr-defined]

    asyncio.run(_exercise())


def test_reload_waits_for_inflight_and_closes_old_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RetrievalRuntime()
    old_pipeline = _DummyPipeline()
    runtime._pipeline = old_pipeline  # type: ignore[attr-defined]
    runtime._state = RuntimeState.HEALTHY  # type: ignore[attr-defined]

    build_calls: list[int] = []

    def _build(*args: object, **kwargs: object) -> _DummyPipeline:
        build_calls.append(1)
        return _DummyPipeline()

    monkeypatch.setattr(runtime, "_build_pipeline", _build)

    async def _exercise() -> None:
        gate = asyncio.Event()

        async def hold_pipeline() -> None:
            async with runtime.acquire():
                await gate.wait()

        holder = asyncio.create_task(hold_pipeline())
        await asyncio.sleep(0)

        reload_task = asyncio.create_task(
            runtime.reload(
                config={},
                config_path=runtime._config_path,  # type: ignore[attr-defined]
                embedding=None,
                llm=None,
            )
        )
        await asyncio.sleep(0.05)
        assert not old_pipeline.vector_store.closed

        gate.set()
        await holder
        await reload_task

    asyncio.run(_exercise())
    assert old_pipeline.vector_store.closed
    assert build_calls


def test_warm_failure_marks_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = RetrievalRuntime()

    def _fail(*args: object, **kwargs: object) -> _DummyPipeline:
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime, "_build_pipeline", _fail)

    async def _exercise() -> None:
        with pytest.raises(RuntimeError):
            await runtime.warm()

    asyncio.run(_exercise())
    assert runtime.state == RuntimeState.DEGRADED
    assert runtime.error == "boom"
