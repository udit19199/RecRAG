import os
from typing import Any, Optional

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA as NVIDIALLM
from llama_index.core.llms import ChatMessage

from adapters.base import BaseEmbedder, BaseLLM

# Module-level cache: model name -> dimension, avoids a live API call on every init
_NIM_DIMENSION_CACHE: dict[str, int] = {}


class NIMEmbedder(BaseEmbedder):
    """NVIDIA NIM embedding provider.

    Detects embedding dimension via a test call on first use per model;
    subsequent instantiations with the same model name reuse the cached value.
    """

    provider = "nim"

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

        if model not in _NIM_DIMENSION_CACHE:
            from utils.rate_limit import check_rate_limit, record_success

            check_rate_limit("embedding")
            _NIM_DIMENSION_CACHE[model] = len(self._client.get_query_embedding("test"))
            record_success("embedding")
        self._dimension = _NIM_DIMENSION_CACHE[model]

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        from utils.rate_limit import check_rate_limit, record_success

        check_rate_limit("embedding")
        res = self._client.get_query_embedding(text)
        record_success("embedding")
        return res

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        from utils.rate_limit import check_rate_limit, record_success

        check_rate_limit("embedding")
        res = self._client.get_text_embedding_batch(texts)
        record_success("embedding")
        return res


class NIMLLM(BaseLLM):
    """NVIDIA NIM LLM provider via llama-index-llms-nvidia."""

    provider = "nim"

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
        return False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        from utils.rate_limit import check_rate_limit, record_success

        check_rate_limit("llm")
        response = self._client.complete(
            prompt,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        record_success("llm")
        return response.text

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        from utils.rate_limit import check_rate_limit, record_success

        check_rate_limit("llm")
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages
        ]
        response = self._client.chat(
            chat_messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        record_success("llm")
        return response.message.content or ""
