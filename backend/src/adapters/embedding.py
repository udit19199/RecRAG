import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests
from openai import OpenAI

from adapters.base import BaseEmbedder
from adapters.utils import create_session_with_pooling

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

DEFAULT_BATCH_SIZE = 500
DEFAULT_OLLAMA_DIMENSION = 768


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider."""

    def __init__(self, model: str = "text-embedding-3-small", **kwargs: Any):
        super().__init__(model, **kwargs)
        api_key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
        base_url = kwargs.pop("base_url", None)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._dimension: Optional[int] = kwargs.get("dimensions")

    @property
    def dimension(self) -> int:
        return self._dimension or EMBEDDING_DIMENSIONS.get(self.model, 1536)

    def _create_embedding_params(self, input_data: str | list[str]) -> dict[str, Any]:
        """Build parameters for embedding API call."""
        params = {"model": self.model, "input": input_data}
        if self._dimension is not None:
            params["dimensions"] = self._dimension
        return params

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(**self._create_embedding_params(text))
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(**self._create_embedding_params(texts))
        return [item.embedding for item in response.data]


class OllamaEmbedder(BaseEmbedder):
    """Ollama local embedding provider with batch processing and connection pooling."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        max_workers: int = 8,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._dimension = kwargs.get("dimension", DEFAULT_OLLAMA_DIMENSION)
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
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding using /api/embed endpoint with chunking for large batches."""
        if not texts:
            return []

        if len(texts) <= self._batch_size:
            return self._embed_batch_single(texts)

        # Chunk large batches to avoid payload size issues
        results = []
        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i : i + self._batch_size]
            results.extend(self._embed_batch_single(chunk))

        return results

    def _embed_batch_single(self, texts: list[str]) -> list[list[float]]:
        """Send a single batch request to Ollama's /api/embed endpoint."""
        if not texts:
            return []

        try:
            # Use the batch endpoint /api/embed (plural)
            response = self.session.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embeddings", [])
        except requests.exceptions.RequestException:
            # Fall back to parallel individual requests if batch fails
            return self._embed_batch_parallel(texts)

    def _embed_batch_parallel(self, texts: list[str]) -> list[list[float]]:
        """Fallback: parallel embedding using ThreadPoolExecutor."""
        results: list[Optional[list[float]]] = [None] * len(texts)
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
