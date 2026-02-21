from abc import ABC, abstractmethod
from typing import Any, Optional

from models.chunk import RetrievalResult


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, dimension: int, **kwargs: Any):
        self.dimension = dimension

    @abstractmethod
    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadata_list: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Add embeddings and documents to the store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 4,
    ) -> tuple[list[float], list[RetrievalResult]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete_all(self) -> None:
        """Delete all documents from the store."""
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the store."""
        pass
