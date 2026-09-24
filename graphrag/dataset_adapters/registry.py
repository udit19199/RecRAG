from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..construction import SourcePage
from . import hotpotqa, multihop_rag, natural_questions


class DatasetRecord(Protocol):
    id: str
    question: str
    answer: str
    pages: Sequence[SourcePage]

    def answer_aliases(self) -> Sequence[str]: ...

    def supporting_sentences(self) -> Sequence[str]: ...


@dataclass(slots=True, frozen=True)
class DatasetSource:
    name: str
    display_name: str
    load_record: Callable[[int], DatasetRecord]


DATASET_SOURCES: list[DatasetSource] = [
    DatasetSource(
        name="hotpotqa",
        display_name="HotpotQA",
        load_record=hotpotqa.load_record,
    ),
    DatasetSource(
        name="multihop_rag",
        display_name="MultiHop-RAG",
        load_record=multihop_rag.load_record,
    ),
    DatasetSource(
        name="natural_questions",
        display_name="Natural Questions",
        load_record=natural_questions.load_record,
    ),
]


def get_source(name: str) -> DatasetSource:
    for source in DATASET_SOURCES:
        if source.name == name:
            return source
    raise ValueError(f"Unknown dataset: {name}")
