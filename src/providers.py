"""Provider model-listing helpers for the retrieval API.

Uses a shared ``requests.Session`` per function call (short-lived) to avoid
the ``httpx`` dependency.
"""

import requests
from typing import Any

# ── Static fallback model lists ───────────────────────────────────────────────

# Known embedding-capable models on Ollama
OLLAMA_EMBEDDING_MODELS = [
    "nomic-embed-text",
    "all-minilm",
    "mxbai-embed-large",
    "snowflake-arctic-embed",
    "bge-m3",
    "bge-large",
    "bge-small",
]

# Known LLM models on Ollama (most popular ones)
OLLAMA_LLM_MODELS = [
    "llama3.2",
    "llama3.2:1b",
    "llama3.1",
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.1:405b",
    "llama3",
    "llama2",
    "mistral",
    "mixtral",
    "codellama",
    "phi3",
    "phi3:mini",
    "phi3:medium",
    "gemma2",
    "gemma2:2b",
    "qwen2.5",
    "qwen2.5:7b",
    "qwen2.5:72b",
    "deepseek-r1",
    "deepseek-coder",
    "neural-chat",
    "orca-mini",
    "tinyllama",
]

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

# Vision-capable models (multimodal)
OPENAI_VISION_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
]

OLLAMA_VISION_MODELS = [
    "llama3.2-vision",
    "llama3.2-vision:90b",
    "llava",
    "llava:latest",
    "llava-llama3",
    "llava:13b",
    "llava:34b",
    "bakllava",
    "minicpm-v",
    "moondream",
]

NIM_VISION_MODELS = [
    "microsoft/phi-4-multimodal-instruct",
]

# Known embedding model name patterns (contains these substrings)
EMBEDDING_KEYWORDS = ["embed", "minilm", "bge-", "snowflake", "mxbai"]


def _is_embedding_model(model_name: str) -> bool:
    """Heuristic to guess if an Ollama model supports embeddings."""
    lower = model_name.lower()
    return any(kw in lower for kw in EMBEDDING_KEYWORDS)


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

    Uses heuristics to distinguish embedding vs LLM models.

    Returns:
        (embed_models, llm_models)
    """
    url = base_url.rstrip("/") + "/api/tags"
    data = _fetch_json(url, timeout=5.0)
    if data is None:
        return OLLAMA_EMBEDDING_MODELS, OLLAMA_LLM_MODELS

    all_models = sorted(
        {m["name"] for m in data.get("models", [])},
    )

    if not all_models:
        return OLLAMA_EMBEDDING_MODELS, OLLAMA_LLM_MODELS

    embed_models = _sorted_unique(
        [m for m in all_models if _is_embedding_model(m)]
    )
    llm_models = _sorted_unique(
        [m for m in all_models if not _is_embedding_model(m)]
    )

    # If embedding heuristic found nothing, return all models for both (conservative)
    if not embed_models:
        return all_models, all_models

    return embed_models, llm_models


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


def _fetch_ollama_vision_models(base_url: str) -> list[str]:
    """Fetch vision-capable models from Ollama.

    Returns:
        List of available vision model names.
    """
    url = base_url.rstrip("/") + "/api/tags"
    data = _fetch_json(url, timeout=5.0)
    if data is None:
        return OLLAMA_VISION_MODELS

    available = {m["name"] for m in data.get("models", [])}
    if not available:
        return OLLAMA_VISION_MODELS

    # Return intersection of known vision models and available models
    vision_models = [m for m in OLLAMA_VISION_MODELS if m in available]

    # Also include any model with vision-related keywords
    for name in available:
        lower = name.lower()
        if any(kw in lower for kw in ("vision", "llava", "minicpm", "moondream", "bakllava")):
            if name not in vision_models:
                vision_models.append(name)

    return _sorted_unique(vision_models) if vision_models else OLLAMA_VISION_MODELS


def _fetch_openai_vision_models(api_key: str, base_url: str | None = None) -> list[str]:
    """Fetch vision-capable models from OpenAI.

    Returns:
        List of available vision model names.
    """
    url = (base_url or "https://api.openai.com").rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = _fetch_json(url, headers=headers)
    if data is None:
        return OPENAI_VISION_MODELS

    all_ids: set[str] = {m["id"] for m in data.get("data", [])}
    # Return intersection of known vision models and available models
    return (
        _sorted_unique([m for m in OPENAI_VISION_MODELS if m in all_ids])
        or OPENAI_VISION_MODELS
    )


def _fetch_nim_vision_models(api_key: str, base_url: str | None = None) -> list[str]:
    """Fetch vision-capable models from NVIDIA NIM.

    Returns:
        List of available vision model names.
    """
    url = (base_url or "https://integrate.api.nvidia.com/v1").rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = _fetch_json(url, headers=headers)
    if data is None:
        return NIM_VISION_MODELS

    all_ids: set[str] = {m["id"] for m in data.get("data", [])}
    # Return intersection of known vision models and available models
    # Also include any model with 'multimodal' or 'vision' in the name
    vision_models = [m for m in NIM_VISION_MODELS if m in all_ids]
    for model_id in all_ids:
        if (
            "multimodal" in model_id.lower() or "vision" in model_id.lower()
        ) and model_id not in vision_models:
            vision_models.append(model_id)
    return _sorted_unique(vision_models) if vision_models else NIM_VISION_MODELS
