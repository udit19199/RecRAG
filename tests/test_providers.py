from unittest.mock import MagicMock, patch

from providers import _fetch_nim_models, _fetch_openai_models, _fetch_ollama_models


def test_fetch_nim_models_deduplicates_ids() -> None:
    payload = {
        "data": [
            {
                "id": "openai/gpt-oss-120b",
                "object": "model",
                "owned_by": "openai",
                "created": 1,
            },
            {
                "id": "openai/gpt-oss-120b",
                "object": "model",
                "owned_by": "openai",
                "created": 1,
            },
            {
                "id": "nvidia/llama3-chatqa-1.5-8b",
                "object": "model",
                "owned_by": "nvidia",
                "created": 1,
            },
        ]
    }

    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status = MagicMock()

    with patch("requests.Session.get", return_value=mock_response):
        embed_models, llm_models = _fetch_nim_models("test-key")

    assert llm_models == ["nvidia/llama3-chatqa-1.5-8b", "openai/gpt-oss-120b"]
    assert embed_models == []


def test_fetch_openai_models_deduplicates_ids() -> None:
    payload = {
        "data": [
            {"id": "gpt-4o-mini"},
            {"id": "gpt-4o-mini"},
            {"id": "text-embedding-3-small"},
            {"id": "text-embedding-3-small"},
        ]
    }

    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status = MagicMock()

    with patch("requests.Session.get", return_value=mock_response):
        embed_models, llm_models = _fetch_openai_models("test-key")

    assert embed_models == ["text-embedding-3-small"]
    assert llm_models == ["gpt-4o-mini"]


def test_fetch_ollama_models_deduplicates_ids() -> None:
    payload = {
        "models": [
            {"name": "llama3"},
            {"name": "llama3"},
            {"name": "nomic-embed-text"},
        ]
    }

    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status = MagicMock()

    with patch("requests.Session.get", return_value=mock_response):
        embed_models, llm_models = _fetch_ollama_models("http://localhost:11434")

    # nomic-embed-text matches embedding keyword; llama3 is LLM-only
    assert embed_models == ["nomic-embed-text"]
    assert llm_models == ["llama3"]
