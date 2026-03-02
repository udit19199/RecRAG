from pathlib import Path
import urllib.parse
from typing import Any, Callable

from adapters import BaseEmbedder, BaseLLM, create_embedder, create_llm

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
    section_config = config.get(section, {})
    provider = section_config.get("provider", defaults["provider"])
    model = section_config.get("model", defaults["model"])

    extra_kwargs = {
        k: v for k, v in section_config.items() if k not in ("provider", "model")
    }

    return create_fn(provider, model=model, **extra_kwargs)


def create_embedder_from_config(config: dict[str, Any]) -> BaseEmbedder:
    defaults = {"provider": "openai", "model": "text-embedding-3-small"}
    return _create_adapter_from_config(config, "embedding", create_embedder, defaults)


def create_llm_from_config(config: dict[str, Any]) -> BaseLLM:
    defaults = {"provider": "openai", "model": "gpt-4o-mini"}
    return _create_adapter_from_config(config, "llm", create_llm, defaults)


def get_collection_name(config: dict[str, Any], embedder_model: str) -> str:
    """Return a Milvus collection name derived from the embedder model.

    The name is prefixed by ``storage.collection_prefix`` (default ``recrag_``)
    and the model slug (slashes and hyphens replaced with underscores).
    """
    prefix = config.get("storage", {}).get("collection_prefix", "recrag_")
    embedding_id = embedder_model.replace("/", "_").replace("-", "_").replace(":", "_")
    return f"{prefix}{embedding_id}"


def get_milvus_uri(config: dict[str, Any], config_path: Path) -> str:
    """Return the Milvus connection URI.

    Milvus Lite (embedded) support has been removed — always use a
    standalone Milvus server. The server address is hardcoded to the
    local default for simplicity and to match deployment assumptions.
    """
    server_cfg = config.get("storage", {}).get("server", {}) or {}
    host = server_cfg.get("host", "localhost")
    port = server_cfg.get("port", 19530)

    username = server_cfg.get("username") or server_cfg.get("user")
    password = server_cfg.get("password")

    # If host already includes a scheme, parse and inject credentials if provided.
    if isinstance(host, str) and (
        host.startswith("http://") or host.startswith("https://") or "://" in host
    ):
        parsed = urllib.parse.urlparse(host)
        netloc = parsed.netloc
        if username or password:
            if not username or not password:
                raise ValueError(
                    "Milvus username and password must be set for cloud endpoints. "
                    "Set MILVUS_USERNAME and MILVUS_PASSWORD in your .env or config.toml."
                )
            user_enc = urllib.parse.quote(str(username), safe="")
            pass_enc = urllib.parse.quote(str(password), safe="")
            # Remove any existing userinfo from netloc
            host_port = netloc.split("@")[-1]
            netloc = f"{user_enc}:{pass_enc}@{host_port}"
        rebuilt = parsed._replace(netloc=netloc)
        return urllib.parse.urlunparse(rebuilt)

    # If host already contains a port, prepend scheme and inject credentials if present.
    host_str = str(host)
    scheme = "http"
    if ":" in host_str:
        authority = host_str
    else:
        authority = f"{host_str}:{port}"

    if username or password:
        if not username or not password:
            raise ValueError(
                "Milvus username and password must be set for cloud endpoints. "
                "Set MILVUS_USERNAME and MILVUS_PASSWORD in your .env or config.toml."
            )
        user_enc = urllib.parse.quote(str(username), safe="")
        pass_enc = urllib.parse.quote(str(password), safe="")
        authority = f"{user_enc}:{pass_enc}@{authority}"

    return f"{scheme}://{authority}"
