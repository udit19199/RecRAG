"""Data models for RecRAG."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a text chunk with preserved source metadata.

    Attributes:
        text: The chunk text content.
        source: The source file name (e.g., "document.pdf").
        metadata: Full metadata from the source document.
    """

    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Represents a retrieved document with its score/distance.

    Attributes:
        text: The retrieved text content.
        source: The source file name.
        distance: The similarity score/distance from the query.
        metadata: Any additional metadata from the original chunk.
    """

    text: str
    source: str
    distance: float
    metadata: dict[str, Any] = Field(default_factory=dict)
