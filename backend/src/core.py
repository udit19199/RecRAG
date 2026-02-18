import fcntl
import json
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument


class DocumentLoader:
    """Document loader for various file formats."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def load(self) -> list[LlamaDocument]:
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        reader = SimpleDirectoryReader(str(self.directory))
        return reader.load_data()

    def load_file(self, file_path: Path | str) -> list[LlamaDocument]:
        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        return reader.load_data()


class TextSplitter:
    """Text splitter for chunking documents."""

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: list) -> list[str]:
        chunks = []
        for doc in documents:
            text = doc.text if hasattr(doc, "text") else str(doc)
            doc_chunks = self.splitter.split_text(text)
            chunks.extend(doc_chunks)
        return chunks

    def split_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)


class VectorStore:
    """FAISS-based vector store with persistence."""

    def __init__(
        self,
        dimension: int,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index: faiss.Index = self._load_index()
        self.metadata: list[dict[str, Any]] = self._load_metadata()

    def _load_index(self) -> faiss.Index:
        if self.index_path and self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatL2(self.dimension)

    def _load_metadata(self) -> list[dict[str, Any]]:
        if self.metadata_path and self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return []

    def _acquire_lock(self) -> None:
        """Acquire file lock for thread/process-safe operations."""
        if self.metadata_path:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock_file = open(self.metadata_path.with_suffix(".lock"), "w")
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)

    def _release_lock(self) -> None:
        """Release file lock."""
        if hasattr(self, "_lock_file"):
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            del self._lock_file

    def save(self) -> None:
        if self.index_path:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))

        if self.metadata_path:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

    def _remove_by_sources(self, sources: set[str]) -> int:
        """Remove metadata entries for given sources. Returns count removed."""
        if not sources:
            return 0

        indices_to_remove = [
            i for i, m in enumerate(self.metadata) if m.get("source") in sources
        ]

        if not indices_to_remove:
            return 0

        for idx in reversed(indices_to_remove):
            del self.metadata[idx]

        self.index = faiss.IndexFlatL2(self.dimension)
        remaining_embeddings = []
        for m in self.metadata:
            if "embedding" in m:
                remaining_embeddings.append(m["embedding"])

        if remaining_embeddings:
            vectors = np.array(remaining_embeddings, dtype=np.float32)
            self.index.add(vectors)  # type: ignore[call-arg]

        return len(indices_to_remove)

    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._acquire_lock()
        try:
            if metadatas is None:
                metadatas = [{} for _ in documents]

            sources_to_update: set[str] = {
                str(m.get("source")) for m in metadatas if m.get("source")
            }
            if sources_to_update:
                self._remove_by_sources(sources_to_update)

            start_index = self.index.ntotal
            vectors = np.array(embeddings, dtype=np.float32)
            self.index.add(vectors)  # type: ignore[call-arg]

            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                self.metadata.append(
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
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query, k)  # type: ignore[call-arg]

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                result = dict(self.metadata[idx])
                result["distance"] = float(dist)
                results.append(result)

        return distances.tolist(), results

    def delete_all(self) -> None:
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.save()

    @property
    def count(self) -> int:
        return self.index.ntotal
