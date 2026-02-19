from pathlib import Path
from typing import Any, Callable

from adapters import BaseEmbedder, BaseLLM, create_embedder, create_llm
from config import resolve_path

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

def get_vector_store_paths(
    config: dict[str, Any], config_path: Path, embedder_model: str
) -> tuple[Path, Path]:
    """Get index and metadata paths for the vector store."""
    storage_dir = resolve_path(
        config.get("storage", {}).get("directory", "storage"), config_path
    )
    # Rename embedder_id -> embedding_id
    embedding_id = embedder_model.replace("/", "_").replace("-", "_")
    return (
        storage_dir / f"faiss_{embedding_id}.index",
        storage_dir / f"faiss_{embedding_id}.json",
    )
