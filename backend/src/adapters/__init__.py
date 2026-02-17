from typing import Any

from adapters.base import BaseEmbedder, BaseLLM
from adapters.embedding import OpenAIEmbedder, OllamaEmbedder
from adapters.llm import OpenAILLM, OllamaLLM


def create_embedder(provider: str, **kwargs: Any) -> BaseEmbedder:
    """Create an embedder instance based on provider.

    Args:
        provider: Provider name ("openai" or "ollama")
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseEmbedder instance
    """
    if provider == "openai":
        return OpenAIEmbedder(**kwargs)
    elif provider == "ollama":
        return OllamaEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")


def create_llm(provider: str, **kwargs: Any) -> BaseLLM:
    """Create an LLM instance based on provider.

    Args:
        provider: Provider name ("openai" or "ollama")
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseLLM instance
    """
    if provider == "openai":
        return OpenAILLM(**kwargs)
    elif provider == "ollama":
        return OllamaLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
