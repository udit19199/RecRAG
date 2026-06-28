"""Stateless /query used by benchmark must work without a warmed default pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from models.chunk import RetrievalResult
from runtime import RuntimeState


@pytest.fixture
def retrieval_client_unloaded() -> TestClient:
    with patch("app.retrieval.main.RetrievalRuntime") as MockRuntime:
        mock_runtime = MagicMock()
        mock_runtime.warm = AsyncMock()
        mock_runtime.warm_with_retry = AsyncMock()
        mock_runtime.shutdown = AsyncMock()
        mock_runtime.is_loaded = MagicMock(return_value=False)
        mock_runtime.state = RuntimeState.STARTING
        mock_runtime.error = "embedding provider unreachable"
        MockRuntime.return_value = mock_runtime

        from app.retrieval.main import app

        with TestClient(app) as tc:
            yield tc


def test_stateless_query_works_when_pipeline_not_loaded(
    retrieval_client_unloaded: TestClient,
) -> None:
    mock_result = {
        "response": "answer",
        "context": [
            RetrievalResult(text="chunk", source="a.pdf", distance=0.1, metadata={}),
        ],
    }
    with patch.dict("os.environ", {}, clear=True):
        with (
            patch("app.retrieval.main.create_embedder"),
            patch("app.retrieval.main.create_llm_from_config"),
            patch("app.retrieval.main.create_vector_store_from_config"),
            patch("app.retrieval.main.RetrievalPipeline") as MockPipeline,
        ):
            pipeline = MagicMock()
            pipeline.query.return_value = mock_result
            MockPipeline.return_value = pipeline

            response = retrieval_client_unloaded.post(
                "/query",
                json={
                    "query": "What is in the documents?",
                    "embedding": {"provider": "gemini", "model": "text-embedding-004"},
                    "llm": {"provider": "gemini", "model": "gemini-2.0-flash"},
                    "collection_name": "recrag_run_test_naive",
                },
            )

    assert response.status_code == 200
    assert response.json()["response"] == "answer"


def test_default_query_still_requires_loaded_pipeline(
    retrieval_client_unloaded: TestClient,
) -> None:
    with patch.dict("os.environ", {}, clear=True):
        response = retrieval_client_unloaded.post(
            "/query",
            json={"query": "hello"},
        )
    assert response.status_code == 503
