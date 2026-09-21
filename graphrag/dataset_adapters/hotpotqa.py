from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
from pyarrow import Table

from ..construction import SourcePage


@dataclass(slots=True, frozen=True)
class SupportingFact:
    title: str
    sentence_index: int


@dataclass(slots=True, frozen=True)
class HotpotRecord:
    id: str
    question: str
    pages: list[SourcePage]
    question_type: str
    level: str
    answer: str
    supporting_facts: list[SupportingFact]

    def answer_aliases(self) -> list[str]:
        return [self.answer]

    def supporting_sentences(self) -> list[str]:
        sentences = []
        for fact in self.supporting_facts:
            for page in self.pages:
                if page.title == fact.title:
                    sentences.append(page.passages[fact.sentence_index])
                    break
        return sentences


def load_record(index: int) -> HotpotRecord:
    row: dict[str, Any] = _table().slice(index, 1).to_pylist()[0]
    return _record(row)


@cache
def _table() -> Table:
    path = Path("datasets/hotpotqa/distractor-validation.parquet")
    return parquet.read_table(
        path,
        columns=[
            "id",
            "question",
            "answer",
            "type",
            "level",
            "context",
            "supporting_facts",
        ],
    )


def _record(row: dict[str, Any]) -> HotpotRecord:
    context: dict[str, Any] = row["context"]
    facts: dict[str, Any] = row["supporting_facts"]
    return HotpotRecord(
        id=row["id"],
        question=row["question"],
        pages=[
            SourcePage(title=title, passages=sentences)
            for title, sentences in zip(context["title"], context["sentences"])
        ],
        question_type=row["type"],
        level=row["level"],
        answer=row["answer"],
        supporting_facts=[
            SupportingFact(title=title, sentence_index=index)
            for title, index in zip(facts["title"], facts["sent_id"])
        ],
    )
