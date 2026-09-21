from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..construction.construction import SourcePage


class DatasetRecord(Protocol):
    id: str
    question: str
    answer: str
    pages: Sequence[SourcePage]

    def answer_aliases(self) -> Sequence[str]: ...

    def supporting_sentences(self) -> Sequence[str]: ...


class DatasetAdapter(Protocol):
    name: str
    display_name: str

    def load_records(self, limit: int) -> list[DatasetRecord]: ...
