import os
from typing import Any, Optional

import requests
from openai import OpenAI

from adapters.base import BaseLLM
from adapters.utils import create_session_with_pooling

DEFAULT_TEMPERATURE = 0.7


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        api_key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
        base_url = kwargs.pop("base_url", None)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def supports_streaming(self) -> bool:
        return True

    def _get_completion_params(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        """Build parameters for chat completion."""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

    def generate(self, prompt: str, **kwargs: Any) -> str:
        params = self._get_completion_params(
            [{"role": "user", "content": prompt}], **kwargs
        )
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        params = self._get_completion_params(messages, **kwargs)
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content or ""


class OllamaLLM(BaseLLM):
    """Ollama local LLM provider with connection pooling."""

    def __init__(
        self,
        model: str = "llama3",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = create_session_with_pooling()

    @property
    def supports_streaming(self) -> bool:
        return False

    def _build_payload(self, **kwargs: Any) -> dict[str, Any]:
        """Build request payload for Ollama API."""
        payload = {
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False,
        }
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}
        return payload

    def generate(self, prompt: str, **kwargs: Any) -> str:
        payload = self._build_payload(**kwargs)
        payload["prompt"] = prompt

        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        payload = self._build_payload(**kwargs)
        payload["messages"] = messages

        response = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
