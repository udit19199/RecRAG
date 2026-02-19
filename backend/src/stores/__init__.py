from typing import Any

from core import BaseVectorStore


def create_vector_store(
    provider: str,
    dimension: int,
    **kwargs: Any,
) -> BaseVectorStore:
    """Create a vector store instance based on provider.

    Args:
        provider: Provider name (currently only "faiss" supported)
        dimension: Embedding dimension
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseVectorStore instance
    """
    if provider == "faiss":
        from core import VectorStore

        return VectorStore(dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")
