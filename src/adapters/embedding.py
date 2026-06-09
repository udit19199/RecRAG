import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from openai import OpenAI

from adapters.base import BaseEmbedder
from adapters.utils import create_session_with_pooling

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embedding-nomic-embed-text-v1.5": 768,
    "nomic-embed-text": 768,
}

DEFAULT_BATCH_SIZE = 500
DEFAULT_OLLAMA_DIMENSION = 768
DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider."""

    provider = "openai"

    def __init__(self, model: str = "text-embedding-3-small", **kwargs: Any):
        base_url = kwargs.pop("base_url", None)
        api_key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
        if not api_key and base_url:
            # LM Studio and other local OpenAI-compatible servers accept any key.
            api_key = "lm-studio"
        dimensions = kwargs.pop("dimensions", None)
        self._timeout = kwargs.pop("timeout", None)
        super().__init__(model, **kwargs)

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if self._timeout:
            client_kwargs["timeout"] = float(self._timeout)
        self.client = OpenAI(**client_kwargs)
        self._dimension: int | None = dimensions

    @property
    def dimension(self) -> int:
        return self._dimension or EMBEDDING_DIMENSIONS.get(self.model, 1536)

    def _create_embedding_params(self, input_data: str | list[str]) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "input": input_data}
        if self._dimension is not None:
            params["dimensions"] = self._dimension
        return params

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(**self._create_embedding_params(text))
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(**self._create_embedding_params(texts))
        return [item.embedding for item in response.data]


class LMStudioEmbedder(OpenAIEmbedder):
    """LM Studio local embedding server (OpenAI-compatible /v1 API)."""

    provider = "lmstudio"

    def __init__(
        self,
        model: str = "text-embedding-nomic-embed-text-v1.5",
        **kwargs: Any,
    ):
        base_url = (
            kwargs.pop("base_url", None)
            or os.environ.get("LM_STUDIO_BASE_URL")
            or DEFAULT_LM_STUDIO_BASE_URL
        )
        super().__init__(model=model, base_url=base_url, **kwargs)


class OllamaEmbedder(BaseEmbedder):
    """Ollama local embedding provider with batch processing and connection pooling."""

    provider = "ollama"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str | None = None,
        max_workers: int = 8,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ):
        dimension = kwargs.pop("dimension", DEFAULT_OLLAMA_DIMENSION)
        kwargs.pop(
            "api_key", None
        )  # Not used by Ollama; pop to avoid leaking into self.kwargs
        kwargs.pop("base_url", None)
        self._timeout = kwargs.pop("timeout", 120)
        super().__init__(model, **kwargs)
        # Resolve base_url: explicit > OLLAMA_HOST env > default
        resolved_url = (
            base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        )
        self.base_url = resolved_url.rstrip("/")
        self._dimension = dimension
        self._max_workers = max_workers
        self._batch_size = batch_size
        self.session = create_session_with_pooling()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        response = self.session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if len(texts) <= self._batch_size:
            return self._embed_batch_single(texts)

        results = []
        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i : i + self._batch_size]
            results.extend(self._embed_batch_single(chunk))
        return results

    def _embed_batch_single(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.session.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json().get("embeddings", [])
        except requests.exceptions.RequestException:
            return self._embed_batch_parallel(texts)

    def _embed_batch_parallel(self, texts: list[str]) -> list[list[float]]:
        """Fallback when the batch /api/embed endpoint fails."""
        results: list[list[float] | None] = [None] * len(texts)
        errors: list[tuple[int, Exception]] = []

        def embed_with_index(args: tuple[int, str]) -> tuple[int, list[float]]:
            idx, text = args
            return idx, self.embed(text)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(embed_with_index, (i, text)): i
                for i, text in enumerate(texts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    _, embedding = future.result()
                    results[idx] = embedding
                except Exception as e:
                    errors.append((idx, e))

        if errors:
            failed_indices = [idx for idx, _ in errors]
            first_error = errors[0][1]
            raise RuntimeError(
                f"Embedding failed for {len(errors)}/{len(texts)} texts "
                f"at indices {failed_indices}. First error: {first_error}"
            )

        return results  # type: ignore
