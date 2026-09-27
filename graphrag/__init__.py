"""Shared inputs for RecRAG research modules."""

from dataclasses import dataclass


@dataclass(slots=True)
class SourcePage:
    title: str
    passages: list[str]
