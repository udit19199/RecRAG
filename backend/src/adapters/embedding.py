import os
from typing import Any, Optional

import requests
from openai import OpenAI

from adapters.base import BaseEmbedder


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider."""

    def __init__(self, model: str = "text-embedding-3-small", **kwargs: Any):
        super().__init__(model, **kwargs)
        api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        base_url = kwargs.get("base_url")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._dimensions: Optional[int] = kwargs.get("dimensions")

    @property
    def dimension(self) -> int:
        if self._dimensions:
            return self._dimensions
        return EMBEDDING_DIMENSIONS.get(self.model, 1536)

    def embed(self, text: str) -> list[float]:
        if self._dimensions is not None:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self._dimensions,
            )
        else:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._dimensions is not None:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self._dimensions,
            )
        else:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
        return [item.embedding for item in response.data]


class OllamaEmbedder(BaseEmbedder):
    """Ollama local embedding provider."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._dimension = kwargs.get("dimension", 768)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
        return embeddings
