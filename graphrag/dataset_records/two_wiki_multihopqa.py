from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from ..construction.construction import SourcePage


@dataclass(slots=True)
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
        return (self.answer, *_aliases_by_id().get(self.answer_id, ()))

    def supporting_sentences(self) -> list[str]:
        pages = {page.title: page for page in self.pages}
        return [
            pages[title].passages[index]
            for title, index in self.supporting_facts
            if title in pages and index < len(pages[title].passages)
        ]


@cache
def _aliases_by_id() -> dict[str, tuple[str, ...]]:
    aliases_path = Path("datasets/2wikimultihopqa/id_aliases.json")
    aliases: dict[str, tuple[str, ...]] = {}
    with aliases_path.open() as lines:
        for line in lines:
            entry = json.loads(line)
            aliases[entry["Q_id"]] = tuple(entry["aliases"] + entry["demonyms"])
    return aliases


def answer_aliases(record: TwoWikiRecord) -> tuple[str, ...]:
    return record.answer_aliases()


def load_records(limit: int) -> list[TwoWikiRecord]:
    records = json.loads(Path("datasets/2wikimultihopqa/dev.json").read_text())
    return [
        TwoWikiRecord(
            id=record["_id"],
            question=record["question"],
            pages=[
                SourcePage(title=page[0], passages=page[1])
                for page in record["context"]
            ],
            question_type=record["type"],
            answer=record["answer"],
            answer_id=record["answer_id"],
            supporting_facts=tuple(
                (title, int(index)) for title, index in record["supporting_facts"]
            ),
            evidence_triples=tuple(tuple(triple) for triple in record["evidences"]),
            evidence_ids=tuple(tuple(triple) for triple in record["evidences_id"]),
        )
        for record in records[:limit]
    ]


class TwoWikiMultiHopQAAdapter:
    name = "2wikimultihopqa"
    display_name = "2WikiMultiHopQA"

    def load_records(self, limit: int) -> list[TwoWikiRecord]:
        return load_records(limit)


adapter = TwoWikiMultiHopQAAdapter()
