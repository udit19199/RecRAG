"""Provider model-listing helpers for the retrieval API.

Uses a shared ``requests.Session`` per function call (short-lived) to avoid
the ``httpx`` dependency.
"""


import requests

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


def _fetch_ollama_models(base_url: str) -> tuple[list[str], list[str]]:
    """Fetch models from Ollama /api/tags.

    Ollama does not distinguish embed vs LLM models in the tags endpoint,
    so the same list is returned for both roles.

    Returns:
        (embed_models, llm_models)
    """
    try:
        url = base_url.rstrip("/") + "/api/tags"
        with requests.Session() as session:
            resp = session.get(url, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return models, models
    except Exception:
        return [], []


def _fetch_openai_models(
    api_key: str, base_url: str | None = None
) -> tuple[list[str], list[str]]:
    """Fetch models from OpenAI /v1/models.

    Returns:
        (embed_models, llm_models)
    """
    try:
        url = (base_url or "https://api.openai.com").rstrip("/") + "/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        with requests.Session() as session:
            resp = session.get(url, headers=headers, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            all_ids: list[str] = [m["id"] for m in data.get("data", [])]

        embed_models = (
            sorted(
                [m for m in all_ids if "embed" in m.lower() or "embedding" in m.lower()]
            )
            or OPENAI_EMBEDDING_MODELS
        )
        llm_models = (
            sorted(
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
    except Exception:
        return OPENAI_EMBEDDING_MODELS, OPENAI_LLM_MODELS


def _fetch_nim_models(
    api_key: str, base_url: str | None = None
) -> tuple[list[str], list[str]]:
    """Fetch models from NVIDIA NIM /v1/models.

    Returns:
        (embed_models, llm_models)
    """
    try:
        url = (base_url or "https://integrate.api.nvidia.com/v1").rstrip(
            "/"
        ) + "/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        with requests.Session() as session:
            resp = session.get(url, headers=headers, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            all_models = data.get("data", [])

        embed_models = sorted(
            [
                m["id"]
                for m in all_models
                if "embed" in m.get("id", "").lower()
                or m.get("model_type", "") == "embedding"
            ]
        )
        llm_models = sorted(
            [
                m["id"]
                for m in all_models
                if "embed" not in m.get("id", "").lower()
                and m.get("model_type", "") != "embedding"
            ]
        )
        return embed_models, llm_models
    except Exception:
        return [], []
