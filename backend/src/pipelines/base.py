from pathlib import Path
from typing import Any, Callable

from adapters import BaseEmbedder, BaseLLM, create_embedder, create_llm
from config import get_storage_dir, resolve_path

DEFAULT_CONTEXT_TEMPLATE = """Context information:
{context}

Question: {question}

Answer:"""

DEFAULT_BATCH_SIZE = 100
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 4


def _create_adapter_from_config(
    config: dict[str, Any],
    section: str,
    create_fn: Callable[..., Any],
    defaults: dict[str, str],
) -> Any:
    """Create an adapter (embedder or LLM) from configuration."""
    section_config = config.get(section, {})
    provider = section_config.get("provider", defaults["provider"])
    model = section_config.get("model", defaults["model"])

    extra_kwargs = {
        k: v for k, v in section_config.items() if k not in ("provider", "model")
    }

    return create_fn(provider, model=model, **extra_kwargs)


def create_embedder_from_config(config: dict[str, Any]) -> BaseEmbedder:
    """Create an embedder instance from configuration."""
    defaults = {"provider": "openai", "model": "text-embedding-3-small"}
    return _create_adapter_from_config(config, "embedding", create_embedder, defaults)


def create_llm_from_config(config: dict[str, Any]) -> BaseLLM:
    """Create an LLM instance from configuration."""
    defaults = {"provider": "openai", "model": "gpt-4o-mini"}
    return _create_adapter_from_config(config, "llm", create_llm, defaults)


def get_collection_name(config: dict[str, Any], embedder_model: str) -> str:
    """Return a Milvus collection name derived from the embedder model.

    The name is prefixed by ``storage.collection_prefix`` (default ``recrag_``)
    and the model slug (slashes and hyphens replaced with underscores).
    """
    prefix = config.get("storage", {}).get("collection_prefix", "recrag_")
    embedding_id = embedder_model.replace("/", "_").replace("-", "_")
    return f"{prefix}{embedding_id}"


def get_milvus_uri(config: dict[str, Any], config_path: Path) -> str:
    """Return the Milvus connection URI from configuration.

    - ``deployment = "lite"`` → local file path (Milvus Lite).
    - ``deployment = "server"`` → ``http://<host>:<port>`` (Standalone / Distributed).
    """
    storage = config.get("storage", {})
    deployment = storage.get("deployment", "lite")

    if deployment == "lite":
        db_path = storage.get("lite", {}).get("db_path", "storage/milvus_lite.db")
        resolved = resolve_path(db_path, config_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return str(resolved)

    server = storage.get("server", {})
    host = server.get("host", "localhost")
    port = server.get("port", 19530)
    return f"http://{host}:{port}"
