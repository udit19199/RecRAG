from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        pass
