from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..construction import SourcePage
from . import hotpotqa, natural_questions, two_wiki_multihopqa


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
        name="2wikimultihopqa",
        display_name="2WikiMultiHopQA",
        load_record=two_wiki_multihopqa.load_record,
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
