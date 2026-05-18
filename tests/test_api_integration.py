"""API integration tests for ingestion and retrieval endpoints.

Uses FastAPI TestClient to exercise routes with mocked dependencies.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from runtime import RuntimeState


# ── Ingestion API Tests ───────────────────────────────────────────────────────


class TestIngestionAPI:
    """Tests for the ingestion FastAPI application."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Patch the ingestion runtime and return TestClient."""
        with patch("app.ingestion.main.IngestionRuntime") as MockRuntime:
            mock_runtime = MagicMock()
            mock_runtime.warm = AsyncMock()
            mock_runtime.shutdown = AsyncMock()
            MockRuntime.return_value = mock_runtime

            from app.ingestion.main import app

            with TestClient(app) as tc:
                yield tc

    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ingestion-api"

    def test_metrics(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_status_returns_idle_initially(self, client: TestClient) -> None:
        with patch("app.ingestion.main.read_status", return_value={"status": "idle"}):
            response = client.get("/status")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "idle"

    def test_upload_requires_files(self, client: TestClient) -> None:
        response = client.post("/upload")
        assert response.status_code == 422  # validation error

    def test_upload_without_auth_when_key_configured(
        self, client: TestClient
    ) -> None:
        with patch.dict("os.environ", {"REC_RAG_API_KEY": "test-key"}):
            response = client.post(
                "/upload",
                files={"files": ("test.pdf", b"%PDF-1.4 test %%EOF", "application/pdf")},
            )
            # Without API key header, should be 401
            assert response.status_code == 401


class TestIngestionAPIAuth:
    """Tests for ingestion API with authentication enabled."""

    @pytest.fixture
    def client(self) -> TestClient:
        with patch("app.ingestion.main.IngestionRuntime") as MockRuntime:
            mock_runtime = MagicMock()
            mock_runtime.warm = AsyncMock()
            mock_runtime.shutdown = AsyncMock()
            MockRuntime.return_value = mock_runtime

            from app.ingestion.main import app

            with TestClient(app) as tc:
                yield tc

    def test_upload_with_valid_key(self, client: TestClient) -> None:
        with patch.dict("os.environ", {"REC_RAG_API_KEY": "valid-key"}):
            response = client.post(
                "/upload",
                files={
                    "files": (
                        "test.pdf",
                        b"%PDF-1.4 test content %%EOF",
                        "application/pdf",
                    )
                },
                headers={"RecRAG-API-Key": "valid-key"},
            )
            # Should get past auth, fail on something else (depends on runtime)
            assert response.status_code != 401

    def test_delete_requires_auth(self, client: TestClient) -> None:
        """DELETE /documents requires authentication."""
        with patch.dict("os.environ", {"REC_RAG_API_KEY": "test-key"}):
            response = client.delete("/documents/test.pdf")
            assert response.status_code == 401

    def test_delete_with_valid_key(self, client: TestClient) -> None:
        with patch.dict("os.environ", {"REC_RAG_API_KEY": "valid-key"}):
            with patch("app.ingestion.main.PDF_DIR", Path("/tmp/nonexistent")):
                response = client.delete(
                    "/documents/test.pdf",
                    headers={"RecRAG-API-Key": "valid-key"},
                )
                # File doesn't exist, should be 404
                assert response.status_code == 404

    def test_reindex_requires_auth(self, client: TestClient) -> None:
        with patch.dict("os.environ", {"REC_RAG_API_KEY": "test-key"}):
            response = client.post("/reindex")
            assert response.status_code == 401


# ── Retrieval API Tests ───────────────────────────────────────────────────────


class TestRetrievalAPI:
    """Tests for the retrieval FastAPI application."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Patch the retrieval runtime and return TestClient."""
        with patch("app.retrieval.main.RetrievalRuntime") as MockRuntime:
            mock_runtime = MagicMock()
            mock_runtime.warm = AsyncMock()
            mock_runtime.warm_with_retry = AsyncMock()
            mock_runtime.shutdown = AsyncMock()
            mock_runtime.is_loaded = MagicMock(return_value=True)
            mock_runtime.state = RuntimeState.HEALTHY
            mock_runtime.error = None
            MockRuntime.return_value = mock_runtime

            from app.retrieval.main import app

            with TestClient(app) as tc:
                yield tc

    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_pipeline_status(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "pipeline_loaded" in data
        assert "has_documents" in data

    def test_metrics(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_query_requires_auth_when_configured(self, client: TestClient) -> None:
        with patch.dict("os.environ", {"REC_RAG_API_KEY": "test-key"}):
            response = client.post(
                "/query",
                json={"query": "test query"},
            )
            assert response.status_code == 401

    def test_query_without_auth_when_configured(self, client: TestClient) -> None:
        with patch.dict("os.environ", {"REC_RAG_API_KEY": "test-key"}):
            response = client.post(
                "/query",
                json={"query": "test"},
            )
            assert response.status_code == 401

    def test_config_requires_auth(self, client: TestClient) -> None:
        with patch.dict("os.environ", {"REC_RAG_API_KEY": "test-key"}):
            response = client.get("/config")
            assert response.status_code == 401

    def test_providers_returns_structure(self, client: TestClient) -> None:
        """Providers endpoint should return structured data."""
        with patch.dict("os.environ", {}, clear=True):
            response = client.get("/providers")
            assert response.status_code == 200
            data = response.json()
            assert "embedders" in data
            assert "llms" in data
            assert "vision" in data


# ── CORS Tests ────────────────────────────────────────────────────────────────


class TestCORS:
    """CORS headers must be set correctly on both APIs."""

    def test_ingestion_cors_headers(self) -> None:
        with patch("app.ingestion.main.IngestionRuntime") as MockRuntime:
            mock_runtime = MagicMock()
            mock_runtime.warm = AsyncMock()
            mock_runtime.shutdown = AsyncMock()
            MockRuntime.return_value = mock_runtime

            from app.ingestion.main import app

            with TestClient(app) as tc:
                response = tc.options(
                    "/health",
                    headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "GET",
                    },
                )
                assert response.status_code == 200
                assert response.headers.get("access-control-allow-origin") in (
                    "http://localhost:3000",
                    "*",
                )

    def test_retrieval_cors_headers(self) -> None:
        with patch("app.retrieval.main.RetrievalRuntime") as MockRuntime:
            mock_runtime = MagicMock()
            mock_runtime.warm = AsyncMock()
            mock_runtime.warm_with_retry = AsyncMock()
            mock_runtime.shutdown = AsyncMock()
            mock_runtime.is_loaded = MagicMock(return_value=True)
            MockRuntime.return_value = mock_runtime

            from app.retrieval.main import app

            with TestClient(app) as tc:
                response = tc.options(
                    "/health",
                    headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "GET",
                    },
                )
                assert response.status_code == 200

    def test_delete_method_allowed_on_ingestion(self) -> None:
        """DELETE must be in the allowed methods for ingestion API."""
        with patch("app.ingestion.main.IngestionRuntime") as MockRuntime:
            mock_runtime = MagicMock()
            mock_runtime.warm = AsyncMock()
            mock_runtime.shutdown = AsyncMock()
            MockRuntime.return_value = mock_runtime

            from app.ingestion.main import app

            # Check the CORS middleware allows DELETE
            cors_middleware = None
            for middleware in app.user_middleware:
                if middleware.cls.__name__ == "CORSMiddleware":
                    cors_middleware = middleware
                    break

            assert cors_middleware is not None
            kwargs = cors_middleware.kwargs
            assert "DELETE" in kwargs.get("allow_methods", [])
