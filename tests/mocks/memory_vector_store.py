"""Lightweight in-memory vector store for fast unit tests.

This module provides ``InMemoryVectorStore``, a ``VectorStore`` implementation
backed by plain Python lists.  It performs brute-force L2 similarity search so
tests stay fast and dependency-free (no Milvus instance required).
"""

from typing import Any

from models.chunk import RetrievalResult


def _l2_distance(a: list[float], b: list[float]) -> float:
    """Return the squared L2 distance between two equal-length vectors."""
    return sum((x - y) ** 2 for x, y in zip(a, b))


class InMemoryVectorStore:
    """In-memory vector store used for unit testing.

    Mirrors the public contract of ``VectorStore`` without requiring a
    running Milvus instance.  Source-based deduplication (overwrite-on-re-add)
    is preserved so pipeline logic can be exercised end-to-end.
    """

    def __init__(self, dimension: int, **kwargs: Any) -> None:
        self.dimension = dimension
        self._vectors: list[list[float]] = []
        self._docs: list[dict[str, Any]] = []  # {text, source, **extra_meta}

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> None:
        if metadata_list is None:
            metadata_list = [{} for _ in documents]

        # Remove stale entries for sources being re-added.
        sources: set[str] = {str(m["source"]) for m in metadata_list if m.get("source")}
        if sources:
            kept = [
                (v, d)
                for v, d in zip(self._vectors, self._docs)
                if d.get("source") not in sources
            ]
            self._vectors = [v for v, _ in kept]
            self._docs = [d for _, d in kept]

        for embedding, doc, meta in zip(embeddings, documents, metadata_list):
            self._vectors.append(embedding)
            self._docs.append({"text": doc, **meta})

    def search(
        self,
        query_embedding: list[float],
        k: int = 4,
    ) -> tuple[list[float], list[RetrievalResult]]:
        if not self._vectors:
            return [], []

        scored = sorted(
            range(len(self._vectors)),
            key=lambda i: _l2_distance(self._vectors[i], query_embedding),
        )
        top = scored[:k]

        distances = [_l2_distance(self._vectors[i], query_embedding) for i in top]
        results = [
            RetrievalResult(
                text=self._docs[i].get("text", ""),
                source=self._docs[i].get("source", "unknown"),
                distance=distances[j],
                metadata={
                    kk: vv
                    for kk, vv in self._docs[i].items()
                    if kk not in ("text", "source")
                },
            )
            for j, i in enumerate(top)
        ]
        return distances, results

    def delete_all(self) -> None:
        self._vectors = []
        self._docs = []

    @property
    def count(self) -> int:
        return len(self._vectors)

    def save(self) -> None:  # no-op
        pass
