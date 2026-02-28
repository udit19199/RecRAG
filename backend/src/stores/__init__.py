from .base import BaseVectorStore
from .milvus import MilvusVectorStore

VectorStore = MilvusVectorStore

__all__ = ["BaseVectorStore", "MilvusVectorStore", "VectorStore"]
