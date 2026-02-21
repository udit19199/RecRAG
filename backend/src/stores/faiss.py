import fcntl
import json
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from models.chunk import RetrievalResult
from .base import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store with persistence."""

    def __init__(
        self,
        dimension: int,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        super().__init__(dimension)
        self._index_path = index_path
        self._metadata_path = metadata_path

        self._index: faiss.Index = self._load_index()
        self._metadata: list[dict[str, Any]] = self._load_metadata()

    def _load_index(self) -> faiss.Index:
        if self._index_path and self._index_path.exists():
            return faiss.read_index(str(self._index_path))
        return faiss.IndexFlatL2(self.dimension)

    def _load_metadata(self) -> list[dict[str, Any]]:
        if self._metadata_path and self._metadata_path.exists():
            with open(self._metadata_path, "r") as f:
                return json.load(f)
        return []

    def _acquire_lock(self) -> None:
        if self._metadata_path:
            self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock_file = open(self._metadata_path.with_suffix(".lock"), "w")
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)

    def _release_lock(self) -> None:
        if hasattr(self, "_lock_file"):
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            del self._lock_file

    def save(self) -> None:
        if self._index_path:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self._index_path))

        if self._metadata_path:
            self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._metadata_path, "w") as f:
                json.dump(self._metadata, f, indent=2)

    def _remove_by_sources(self, sources: set[str]) -> int:
        if not sources:
            return 0

        indices_to_remove = set(
            i for i, m in enumerate(self._metadata) if m.get("source") in sources
        )

        if not indices_to_remove:
            return 0

        kept_indices = [
            i for i in range(len(self._metadata)) if i not in indices_to_remove
        ]

        if kept_indices:
            kept_vectors = np.array(
                [self._index.reconstruct(i) for i in kept_indices], dtype=np.float32
            )

            self._index = faiss.IndexFlatL2(self.dimension)
            self._index.add(kept_vectors)

            self._metadata = [self._metadata[i] for i in kept_indices]
            for new_idx, meta in enumerate(self._metadata):
                meta["index"] = new_idx
        else:
            self._index = faiss.IndexFlatL2(self.dimension)
            self._metadata = []

        return len(indices_to_remove)

    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadata_list: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._acquire_lock()
        try:
            if metadata_list is None:
                metadata_list = [{} for _ in documents]

            sources_to_update: set[str] = {
                str(m.get("source")) for m in metadata_list if m.get("source")
            }
            if sources_to_update:
                self._remove_by_sources(sources_to_update)

            start_index = self._index.ntotal
            vectors = np.array(embeddings, dtype=np.float32)
            self._index.add(vectors)

            for i, (doc, meta) in enumerate(zip(documents, metadata_list)):
                self._metadata.append(
                    {
                        "text": doc,
                        "index": start_index + i,
                        **meta,
                    }
                )

            self.save()
        finally:
            self._release_lock()

    def search(
        self,
        query_embedding: list[float],
        k: int = 4,
    ) -> tuple[list[float], list[RetrievalResult]]:
        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self._index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._metadata):
                meta = self._metadata[idx]
                results.append(
                    RetrievalResult(
                        text=meta.get("text", ""),
                        source=meta.get("source", "unknown"),
                        distance=float(dist),
                        metadata={k: v for k, v in meta.items() if k not in ["text", "source"]},
                    )
                )

        return distances[0].tolist(), results

    def delete_all(self) -> None:
        self._index = faiss.IndexFlatL2(self.dimension)
        self._metadata = []
        self.save()

    @property
    def count(self) -> int:
        return self._index.ntotal
