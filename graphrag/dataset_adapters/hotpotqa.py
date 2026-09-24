from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from ..construction import SourcePage


@dataclass(slots=True, frozen=True)
class HotpotRecord:
    id: str
    question: str
    pages: list[SourcePage]
    question_type: str
    level: str
    answer: str
    evidence: list[str]

    def answer_aliases(self) -> list[str]:
        return [self.answer]

    def supporting_sentences(self) -> list[str]:
        return self.evidence


def load_record(index: int) -> HotpotRecord:
    row = _table()[index]
    return _record(row)


@cache
def _table():
    from datasets import load_dataset

    return load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")


def _record(row) -> HotpotRecord:
    context = row["context"]
    facts = row["supporting_facts"]
    pages = [
        SourcePage(title=title, passages=sentences)
        for title, sentences in zip(context["title"], context["sentences"])
    ]
    return HotpotRecord(
        id=row["id"],
        question=row["question"],
        pages=pages,
        question_type=row["type"],
        level=row["level"],
        answer=row["answer"],
        evidence=[
            page.passages[index]
            for title, index in zip(facts["title"], facts["sent_id"])
            for page in pages
            if page.title == title
        ],
    )
