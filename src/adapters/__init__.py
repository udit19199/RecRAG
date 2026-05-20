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


def list_embedder_providers() -> list[str]:
    return list(_EMBEDDER_REGISTRY.keys())


def list_llm_providers() -> list[str]:
    return list(_LLM_REGISTRY.keys())


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
