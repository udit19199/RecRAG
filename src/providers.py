"""Provider model-listing helpers for the retrieval API.

Uses a shared ``requests.Session`` per function call (short-lived) to avoid
the ``httpx`` dependency.
"""

import requests
from typing import Any

# ── Static fallback model lists ───────────────────────────────────────────────

OPENAI_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

OPENAI_LLM_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o3-mini",
]


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def _fetch_json(
    url: str, headers: dict[str, str] | None = None, timeout: float = 10.0
) -> dict[str, Any] | None:
    try:
        with requests.Session() as session:
            resp = session.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


def _fetch_ollama_models(base_url: str) -> tuple[list[str], list[str]]:
    """Fetch models from Ollama /api/tags.

    Ollama does not distinguish embed vs LLM models in the tags endpoint,
    so the same list is returned for both roles.

    Returns:
        (embed_models, llm_models)
    """
    url = base_url.rstrip("/") + "/api/tags"
    data = _fetch_json(url, timeout=5.0)
    if data is None:
        return [], []

    models = _sorted_unique([m["name"] for m in data.get("models", [])])
    return models, models


def _fetch_openai_models(
    api_key: str, base_url: str | None = None
) -> tuple[list[str], list[str]]:
    """Fetch models from OpenAI /v1/models.

    Returns:
        (embed_models, llm_models)
    """
    url = (base_url or "https://api.openai.com").rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = _fetch_json(url, headers=headers)
    if data is None:
        return OPENAI_EMBEDDING_MODELS, OPENAI_LLM_MODELS

    all_ids: list[str] = [m["id"] for m in data.get("data", [])]

    embed_models = (
        _sorted_unique(
            [m for m in all_ids if "embed" in m.lower() or "embedding" in m.lower()]
        )
        or OPENAI_EMBEDDING_MODELS
    )
    llm_models = (
        _sorted_unique(
            [
                m
                for m in all_ids
                if any(
                    m.startswith(prefix)
                    for prefix in ("gpt-", "o1", "o3", "o4", "chatgpt")
                )
            ]
        )
        or OPENAI_LLM_MODELS
    )
    return embed_models, llm_models


def _fetch_nim_models(
    api_key: str, base_url: str | None = None
) -> tuple[list[str], list[str]]:
    """Fetch models from NVIDIA NIM /v1/models.

    Returns:
        (embed_models, llm_models)
    """
    url = (base_url or "https://integrate.api.nvidia.com/v1").rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = _fetch_json(url, headers=headers)
    if data is None:
        return [], []

    all_models = data.get("data", [])
    embed_models = _sorted_unique(
        [
            m["id"]
            for m in all_models
            if "embed" in m.get("id", "").lower()
            or m.get("model_type", "") == "embedding"
        ]
    )
    llm_models = _sorted_unique(
        [
            m["id"]
            for m in all_models
            if "embed" not in m.get("id", "").lower()
            and m.get("model_type", "") != "embedding"
        ]
    )
    return embed_models, llm_models
