"""Google Gemini provider — LLM, Embedding, and Vision adapters.

Requires ``google-generativeai`` and a ``GEMINI_API_KEY`` env var.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from adapters.base import BaseEmbedder, BaseLLM
from adapters.vision import BaseVisionExtractor

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Models known to work — users can pass any valid Gemini model name.
EMBEDDING_MODELS = {
    "text-embedding-004",
    "models/embedding-001",
    "gemini-embedding-001",
    "gemini-embedding-2",
}
# Defaults
DEFAULT_LLM_MODEL = "gemini-2.0-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_VISION_MODEL = "gemini-2.0-flash"


# ── Shared helpers ────────────────────────────────────────────────────────────


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is required for Gemini provider"
        )
    return key


def _build_url(model: str, action: str = "generateContent") -> str:
    base = os.environ.get("GEMINI_API_BASE", DEFAULT_GEMINI_API_BASE)
    model_id = model.removeprefix("models/")
    return f"{base}/models/{model_id}:{action}"


def _call_gemini(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: int = 120,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    resp = requests.post(
        url,
        headers=headers,
        params={"key": api_key},
        json=payload,
        timeout=timeout,
    )
    if not resp.ok:
        detail = resp.text
        try:
            detail = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            pass
        raise RuntimeError(f"Gemini API error ({resp.status_code}): {detail}")
    return resp.json()


# ── LLM ───────────────────────────────────────────────────────────────────────


class GeminiLLM(BaseLLM):
    """Google Gemini LLM provider."""

    provider = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ):
        kwargs.pop("api_key", None)
        self._timeout = kwargs.pop("timeout", 120)
        super().__init__(model, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = _get_api_key()

    @property
    def supports_streaming(self) -> bool:
        return False

    def _build_payload(
        self,
        contents: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.max_tokens or kwargs.get("max_tokens"):
            generation_config["maxOutputTokens"] = kwargs.get(
                "max_tokens", self.max_tokens
            )
        return {
            "contents": contents,
            "generationConfig": generation_config,
        }

    def _extract_text(self, data: dict[str, Any]) -> str:
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            # Check for blocked response
            try:
                reason = data["candidates"][0]["finishReason"]
                msg = (
                    data["candidates"][0]
                    .get("safetyRatings", [{}])[0]
                    .get("category", "unknown")
                )
                raise RuntimeError(
                    f"Gemini response blocked: finishReason={reason}, category={msg}"
                )
            except (KeyError, IndexError):
                pass
            raise RuntimeError(f"Unexpected Gemini response: {data}")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        url = _build_url(self.model, "generateContent")
        payload = self._build_payload(
            [{"role": "user", "parts": [{"text": prompt}]}], **kwargs
        )
        data = _call_gemini(url, payload, self._api_key, self._timeout)
        return self._extract_text(data)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        url = _build_url(self.model, "generateContent")
        # Convert OpenAI-format messages to Gemini format
        gemini_contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Gemini uses "model" instead of "assistant"
            gemini_role = "model" if role == "assistant" else "user"
            gemini_contents.append(
                {"role": gemini_role, "parts": [{"text": msg.get("content", "")}]}
            )
        payload = self._build_payload(gemini_contents, **kwargs)
        data = _call_gemini(url, payload, self._api_key, self._timeout)
        return self._extract_text(data)


# ── Embedding ─────────────────────────────────────────────────────────────────


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini embedding provider."""

    provider = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        **kwargs: Any,
    ):
        kwargs.pop("api_key", None)
        self._timeout = kwargs.pop("timeout", 120)
        super().__init__(model, **kwargs)
        self._api_key = _get_api_key()
        # gemini-embedding-001 defaults to 3072; text-embedding-* defaults to 768
        # and supports output_dimensionality to reduce dims
        self._dimension = kwargs.pop("dimensions", 3072)

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _supports_output_dimensionality(model: str) -> bool:
        """Check if the model supports the output_dimensionality parameter.

        Only newer text-embedding-* models accept this parameter.
        gemini-embedding-* models always use their default dimension.
        """
        model_name = model.removeprefix("models/")
        return model_name.startswith("text-embedding-")

    def embed(self, text: str) -> list[float]:
        url = _build_url(self.model, "embedContent")
        payload: dict[str, Any] = {"content": {"parts": [{"text": text}]}}
        if self._supports_output_dimensionality(self.model) and self._dimension != 3072:
            payload["output_dimensionality"] = self._dimension
        data = _call_gemini(url, payload, self._api_key, self._timeout)
        return data["embedding"]["values"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = _build_url(self.model, "batchEmbedContents")
        payload: dict[str, Any] = {
            "requests": [
                {
                    "model": f"models/{self.model.removeprefix('models/')}",
                    "content": {"parts": [{"text": t}]},
                }
                for t in texts
            ]
        }
        if self._supports_output_dimensionality(self.model) and self._dimension != 3072:
            payload["output_dimensionality"] = self._dimension
        data = _call_gemini(url, payload, self._api_key, self._timeout)
        return [e["values"] for e in data.get("embeddings", [])]


# ── Vision ────────────────────────────────────────────────────────────────────


class GeminiVisionExtractor(BaseVisionExtractor):
    """Vision extractor using Gemini's multimodal capabilities."""

    provider = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_VISION_MODEL,
        **kwargs: Any,
    ):
        kwargs.pop("api_key", None)
        self._timeout = kwargs.pop("timeout", 120)
        super().__init__(model, **kwargs)
        self._api_key = _get_api_key()
        self._prompt = kwargs.pop(
            "prompt",
            "Extract all text from this document page image accurately.",
        )

    def extract_text(self, image_bytes: bytes, prompt: str | None = None) -> str:
        import base64

        prompt = prompt or self._prompt
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        url = _build_url(self.model, "generateContent")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/png", "data": b64}},
                    ]
                }
            ]
        }
        data = _call_gemini(url, payload, self._api_key, self._timeout)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Gemini vision failed: {data}")

    def extract_text_batch(
        self, images: list[bytes], prompt: str | None = None
    ) -> list[str]:
        return [self.extract_text(img, prompt) for img in images]
