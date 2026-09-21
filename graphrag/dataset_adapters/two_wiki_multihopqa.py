from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ..construction import SourcePage


@dataclass(slots=True, frozen=True)
class TwoWikiRecord:
    id: str
    question: str
    pages: list[SourcePage]
    question_type: str
    answer: str
    answer_id: str | None
    supporting_facts: tuple[tuple[str, int], ...]
    evidence_triples: tuple[tuple[str, str, str], ...]
    evidence_ids: tuple[tuple[str, str, str], ...]

    def answer_aliases(self) -> tuple[str, ...]:
        if self.answer_id is None:
            return (self.answer,)
        return (self.answer, *_aliases_by_id()[self.answer_id])

    def supporting_sentences(self) -> list[str]:
        pages = {page.title: page for page in self.pages}
        return [pages[title].passages[index] for title, index in self.supporting_facts]


@cache
def _aliases_by_id() -> dict[str, tuple[str, ...]]:
    aliases_path = Path("datasets/2wikimultihopqa/id_aliases.json")
    aliases: dict[str, tuple[str, ...]] = {}
    with aliases_path.open() as lines:
        for line in lines:
            entry: dict[str, Any] = json.loads(line)
            aliases[entry["Q_id"]] = tuple(entry["aliases"] + entry["demonyms"])
    return aliases


def load_record(index: int) -> TwoWikiRecord:
    return _record(_rows()[index])


@cache
def _rows() -> list[dict[str, Any]]:
    path = Path("datasets/2wikimultihopqa/dev.json")
    return json.loads(path.read_text())


def _record(row: dict[str, Any]) -> TwoWikiRecord:
    return TwoWikiRecord(
        id=row["_id"],
        question=row["question"],
        pages=[
            SourcePage(title=title, passages=sentences)
            for title, sentences in row["context"]
        ],
        question_type=row["type"],
        answer=row["answer"],
        answer_id=row["answer_id"],
        supporting_facts=tuple(
            (title, int(index)) for title, index in row["supporting_facts"]
        ),
        evidence_triples=tuple(tuple(triple) for triple in row["evidences"]),
        evidence_ids=tuple(tuple(triple) for triple in row["evidences_id"]),
    )
