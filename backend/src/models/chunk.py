"""Data models for RecRAG."""

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    text: str
    source: str
    distance: float
    metadata: dict[str, Any] = Field(default_factory=dict)
