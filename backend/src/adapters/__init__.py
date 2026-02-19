from typing import Any, Type

from adapters.base import BaseEmbedder, BaseLLM

_EMBEDDER_REGISTRY: dict[str, Type[BaseEmbedder]] = {}
_LLM_REGISTRY: dict[str, Type[BaseLLM]] = {}


def register_embedder(provider: str, cls: Type[BaseEmbedder]) -> None:
    """Register an embedder provider.

    Args:
        provider: Provider name (e.g., "openai", "ollama")
        cls: Embedder class to register
    """
    _EMBEDDER_REGISTRY[provider] = cls


def register_llm(provider: str, cls: Type[BaseLLM]) -> None:
    """Register an LLM provider.

    Args:
        provider: Provider name (e.g., "openai", "ollama")
        cls: LLM class to register
    """
    _LLM_REGISTRY[provider] = cls


def create_embedder(provider: str, **kwargs: Any) -> BaseEmbedder:
    """Create an embedder instance based on provider.

    Args:
        provider: Provider name
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseEmbedder instance

    Raises:
        ValueError: If provider is not registered
    """
    if provider not in _EMBEDDER_REGISTRY:
        available = list(_EMBEDDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown embedder provider: {provider}. Available: {available}"
        )
    return _EMBEDDER_REGISTRY[provider](**kwargs)


def create_llm(provider: str, **kwargs: Any) -> BaseLLM:
    """Create an LLM instance based on provider.

    Args:
        provider: Provider name
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseLLM instance

    Raises:
        ValueError: If provider is not registered
    """
    if provider not in _LLM_REGISTRY:
        available = list(_LLM_REGISTRY.keys())
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {available}")
    return _LLM_REGISTRY[provider](**kwargs)


def list_embedder_providers() -> list[str]:
    """List all registered embedder providers."""
    return list(_EMBEDDER_REGISTRY.keys())


def list_llm_providers() -> list[str]:
    """List all registered LLM providers."""
    return list(_LLM_REGISTRY.keys())


from adapters.embedding import OpenAIEmbedder, OllamaEmbedder
from adapters.llm import OpenAILLM, OllamaLLM
from adapters.nim import NIMEmbedder, NIMLLM

register_embedder("openai", OpenAIEmbedder)
register_embedder("ollama", OllamaEmbedder)
register_embedder("nim", NIMEmbedder)
register_llm("openai", OpenAILLM)
register_llm("ollama", OllamaLLM)
register_llm("nim", NIMLLM)
