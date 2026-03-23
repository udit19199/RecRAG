from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.retrieval.main import app


def test_retrieval_health_reports_degraded_when_warmup_fails(monkeypatch) -> None:
    monkeypatch.setattr("api.retrieval.main.start_pipeline_warmer", MagicMock())
    monkeypatch.setattr("api.retrieval.main.is_pipeline_loaded", lambda: False)
    monkeypatch.setattr(
        "api.retrieval.main.get_pipeline_status",
        lambda: ("degraded", "Milvus unavailable"),
    )

    with TestClient(app) as client:
        res = client.get("/health")

    assert res.status_code == 200
    assert res.json()["status"] == "degraded"
    assert res.json()["error_message"] == "Milvus unavailable"


def test_retrieval_query_returns_503_when_not_ready(monkeypatch) -> None:
    monkeypatch.setattr("api.retrieval.main.start_pipeline_warmer", MagicMock())
    monkeypatch.setattr("api.retrieval.main.is_pipeline_loaded", lambda: False)
    monkeypatch.setattr(
        "api.retrieval.main.get_pipeline_status",
        lambda: ("degraded", "Milvus unavailable"),
    )

    with TestClient(app) as client:
        res = client.post("/query", json={"query": "hello"})

    assert res.status_code == 503
    assert "Milvus unavailable" in res.json()["detail"]
