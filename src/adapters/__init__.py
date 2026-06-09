from typing import Any

from adapters.base import BaseEmbedder, BaseLLM

_EMBEDDER_REGISTRY: dict[str, type[BaseEmbedder]] = {}
_LLM_REGISTRY: dict[str, type[BaseLLM]] = {}


def register_embedder(provider: str, cls: type[BaseEmbedder]) -> None:
    _EMBEDDER_REGISTRY[provider] = cls


def register_llm(provider: str, cls: type[BaseLLM]) -> None:
    _LLM_REGISTRY[provider] = cls


def create_embedder(provider: str, **kwargs: Any) -> BaseEmbedder:
    """Raises ValueError for unknown providers."""
    if provider not in _EMBEDDER_REGISTRY:
        available = list(_EMBEDDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown embedder provider: {provider}. Available: {available}"
        )
    return _EMBEDDER_REGISTRY[provider](**kwargs)


def create_llm(provider: str, **kwargs: Any) -> BaseLLM:
    """Raises ValueError for unknown providers."""
    if provider not in _LLM_REGISTRY:
        available = list(_LLM_REGISTRY.keys())
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {available}")
    return _LLM_REGISTRY[provider](**kwargs)


from adapters.embedding import OpenAIEmbedder, OllamaEmbedder  # noqa: E402
from adapters.llm import OpenAILLM, OllamaLLM  # noqa: E402
from adapters.nim import NIMEmbedder, NIMLLM  # noqa: E402

try:
    from adapters.gemini import GeminiEmbedder, GeminiLLM

    register_embedder("gemini", GeminiEmbedder)
    register_llm("gemini", GeminiLLM)
except ImportError:
    pass

register_embedder("openai", OpenAIEmbedder)
register_embedder("ollama", OllamaEmbedder)
register_embedder("nim", NIMEmbedder)
register_llm("openai", OpenAILLM)
register_llm("ollama", OllamaLLM)
register_llm("nim", NIMLLM)


def create_llm_from_config(
    config: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> BaseLLM:
    """Build a BaseLLM from a loaded config dict.

    Priority: explicit args > config.toml [llm] section > built-in defaults.
    Extra keys from the [llm] config section (timeout, base_url, etc.) are
    forwarded to the adapter constructor; ``provider`` and ``model`` are
    consumed here. Caller-supplied ``overrides`` take precedence over config.

    Args:
        config: Parsed config dict (output of ``load_config``).
        provider: Override the LLM provider name (e.g. ``"ollama"``).
        model: Override the model name.
        **overrides: Additional kwargs forwarded to the adapter (highest priority).

    Returns:
        A configured ``BaseLLM`` instance.
    """
    llm_cfg = config.get("llm", {})
    actual_provider = provider or llm_cfg.get("provider", "gemini")
    actual_model = model or llm_cfg.get("model", "gemini-2.0-flash")
    kwargs = {
        k: v
        for k, v in llm_cfg.items()
        if k not in ("provider", "model") and not k.startswith("_")
    }
    kwargs.update(overrides)
    return create_llm(actual_provider, model=actual_model, **kwargs)
