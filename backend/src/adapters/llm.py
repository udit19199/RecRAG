import os
from typing import Any, Optional

import requests
from openai import OpenAI

from adapters.base import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
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

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class OllamaLLM(BaseLLM):
    """Ollama local LLM provider."""

    def __init__(
        self,
        model: str = "llama3",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def supports_streaming(self) -> bool:
        return False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
