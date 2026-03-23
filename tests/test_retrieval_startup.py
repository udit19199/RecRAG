from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.retrieval.main import app


def test_retrieval_pipeline_is_warmed_on_startup(monkeypatch) -> None:
    mock_start_pipeline_warmer = MagicMock()
    monkeypatch.setattr(
        "api.retrieval.main.start_pipeline_warmer", mock_start_pipeline_warmer
    )

    with TestClient(app):
        pass

    mock_start_pipeline_warmer.assert_called_once()
