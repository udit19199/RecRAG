"""NVIDIA NIM adapters using native LlamaIndex integrations."""

import os
from typing import Any, Optional

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA as NVIDIALLM
from llama_index.core.llms import ChatMessage

from adapters.base import BaseEmbedder, BaseLLM


class NIMEmbedder(BaseEmbedder):
    """NVIDIA NIM embedding provider.

    Uses llama-index-embeddings-nvidia for cloud-hosted NIM models.
    Auto-detects dimension by making a test API call on initialization.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        truncate: str = "NONE",
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self._api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable required for NIM provider"
            )

        self._client = NVIDIAEmbedding(
            model=model,
            base_url=base_url,
            api_key=self._api_key,
            truncate=truncate,
        )

        # Auto-detect dimension with test call
        self._dimension = self._detect_dimension()

    def _detect_dimension(self) -> int:
        """Detect embedding dimension by making a test API call."""
        test_embedding = self._client.get_query_embedding("test")
        return len(test_embedding)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return self._client.get_query_embedding(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._client.get_text_embedding_batch(texts)


class NIMLLM(BaseLLM):
    """NVIDIA NIM LLM provider.

    Uses llama-index-llms-nvidia for cloud-hosted NIM models.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self._api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable required for NIM provider"
            )

        self._temperature = temperature
        self._max_tokens = max_tokens

        self._client = NVIDIALLM(
            model=model,
            base_url=base_url,
            api_key=self._api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @property
    def supports_streaming(self) -> bool:
        return False  # Not implemented per requirements

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        response = self._client.complete(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.text

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages
        ]

        response = self._client.chat(
            chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.message.content or ""
