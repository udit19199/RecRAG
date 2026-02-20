from .base import BaseVectorStore
from .faiss import FAISSVectorStore

VectorStore = FAISSVectorStore

__all__ = ["BaseVectorStore", "FAISSVectorStore", "VectorStore"]
