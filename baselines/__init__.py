"""Shared benchmark types for Normal RAG and HippoRAG."""

from pydantic import BaseModel


class RetrievedContext(BaseModel):
    passages: list[str]
    scores: list[float]
    construction_seconds: float
    retrieval_seconds: float
