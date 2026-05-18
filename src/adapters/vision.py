"""Vision extractors for PDF page image analysis."""

from abc import ABC, abstractmethod
from typing import Any, Type
import base64
import logging
import os

import requests

logger = logging.getLogger(__name__)

# ── Base interface ────────────────────────────────────────────────────────────


class BaseVisionExtractor(ABC):
    """Base class for vision-based text extraction from images."""

    provider: str = "unknown"

    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def extract_text(self, image_bytes: bytes, prompt: str | None = None) -> str:
        """Extract text from an image.

        Args:
            image_bytes: PNG/JPEG image data.
            prompt: Optional prompt to guide extraction.

        Returns:
            Extracted text from the image.
        """
        pass

    @abstractmethod
    def extract_text_batch(
        self, images: list[bytes], prompt: str | None = None
    ) -> list[str]:
        """Extract text from multiple images.

        Args:
            images: List of PNG/JPEG image data.
            prompt: Optional prompt to guide extraction.

        Returns:
            List of extracted text, one per image.
        """
        pass


# ── Registry ──────────────────────────────────────────────────────────────────

_VISION_REGISTRY: dict[str, Type[BaseVisionExtractor]] = {}


def register_vision_extractor(provider: str, cls: Type[BaseVisionExtractor]) -> None:
    _VISION_REGISTRY[provider] = cls


def create_vision_extractor(provider: str, **kwargs: Any) -> BaseVisionExtractor:
    """Create a vision extractor by provider name.

    Raises ValueError for unknown providers.
    """
    if provider not in _VISION_REGISTRY:
        available = list(_VISION_REGISTRY.keys())
        raise ValueError(
            f"Unknown vision extractor provider: {provider}. Available: {available}"
        )
    return _VISION_REGISTRY[provider](**kwargs)


def list_vision_providers() -> list[str]:
    return list(_VISION_REGISTRY.keys())


# ── Default prompt ────────────────────────────────────────────────────────────

DEFAULT_EXTRACTION_PROMPT = """Extract all text from this document page image.
Include:
- All visible text, headings, paragraphs, and captions
- Text from charts, graphs, diagrams, and infographics
- Table contents in a readable format
- Any labels, legends, or annotations

Return only the extracted text, preserving logical reading order.
Do not include commentary or descriptions of visual elements unless they contain text."""


# ── OpenAI Vision Extractor ───────────────────────────────────────────────────


class OpenAIVisionExtractor(BaseVisionExtractor):
    """Vision extractor using OpenAI's GPT-4 Vision API."""

    provider = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)
        self._timeout = kwargs.pop("timeout", 120)
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
            )

    def _encode_image(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def _call_api(self, messages: list[dict]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def extract_text(self, image_bytes: bytes, prompt: str | None = None) -> str:
        prompt = prompt or DEFAULT_EXTRACTION_PROMPT
        base64_image = self._encode_image(image_bytes)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        try:
            return self._call_api(messages)
        except requests.RequestException as e:
            logger.error("OpenAI vision API call failed: %s", e)
            raise

    def extract_text_batch(
        self, images: list[bytes], prompt: str | None = None
    ) -> list[str]:
        # OpenAI doesn't have native batch vision; process sequentially
        return [self.extract_text(img, prompt) for img in images]


# ── Ollama Vision Extractor ───────────────────────────────────────────────────


class OllamaVisionExtractor(BaseVisionExtractor):
    """Vision extractor using Ollama's local vision models (e.g., llava, bakllava)."""

    provider = "ollama"

    def __init__(
        self,
        model: str = "llava:latest",
        base_url: str | None = None,
        **kwargs: Any,
    ):
        kwargs.pop("base_url", None)
        self._timeout = kwargs.pop("timeout", 180)
        super().__init__(model, **kwargs)
        resolved_url = base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        self.base_url = resolved_url.rstrip("/")

    def _encode_image(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def extract_text(self, image_bytes: bytes, prompt: str | None = None) -> str:
        prompt = prompt or DEFAULT_EXTRACTION_PROMPT
        base64_image = self._encode_image(image_bytes)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.RequestException as e:
            logger.error("Ollama vision API call failed: %s", e)
            raise

    def extract_text_batch(
        self, images: list[bytes], prompt: str | None = None
    ) -> list[str]:
        # Ollama processes one at a time
        return [self.extract_text(img, prompt) for img in images]


# ── NIM Vision Extractor ──────────────────────────────────────────────────────


class NIMVisionExtractor(BaseVisionExtractor):
    """Vision extractor using NVIDIA NIM API."""

    provider = "nim"

    def __init__(
        self,
        model: str = "microsoft/phi-4-multimodal-instruct",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)
        self._timeout = kwargs.pop("timeout", 120)
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = (base_url or "https://integrate.api.nvidia.com/v1").rstrip("/")

        if not self.api_key:
            raise ValueError(
                "NVIDIA API key required. Set NVIDIA_API_KEY or pass api_key."
            )

    def _encode_image(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def extract_text(self, image_bytes: bytes, prompt: str | None = None) -> str:
        from utils.rate_limit import check_rate_limit, record_success

        check_rate_limit("vlm")
        prompt = prompt or DEFAULT_EXTRACTION_PROMPT
        base64_image = self._encode_image(image_bytes)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            res = response.json()["choices"][0]["message"]["content"]
            record_success("vlm")
            return res
        except requests.RequestException as e:
            logger.error("NIM vision API call failed: %s", e)
            raise

    def extract_text_batch(
        self, images: list[bytes], prompt: str | None = None
    ) -> list[str]:
        return [self.extract_text(img, prompt) for img in images]


# ── Register providers ────────────────────────────────────────────────────────

register_vision_extractor("openai", OpenAIVisionExtractor)
register_vision_extractor("ollama", OllamaVisionExtractor)
register_vision_extractor("nim", NIMVisionExtractor)

__all__ = [
    "BaseVisionExtractor",
    "OpenAIVisionExtractor",
    "OllamaVisionExtractor",
    "NIMVisionExtractor",
    "create_vision_extractor",
    "register_vision_extractor",
    "list_vision_providers",
    "DEFAULT_EXTRACTION_PROMPT",
]
