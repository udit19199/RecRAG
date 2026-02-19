"""Data models for RecRAG."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    """Represents a text chunk with preserved source metadata.

    Attributes:
        text: The chunk text content.
        source: The source file name (e.g., "document.pdf").
        metadata: Full metadata from the source document.
    """

    text: str
    source: str
    metadata: dict[str, Any]
