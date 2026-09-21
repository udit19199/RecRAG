from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as parquet

from ..construction.construction import SourcePage


@dataclass(slots=True, frozen=True)
class SupportingFact:
    title: str
    sentence_index: int


@dataclass(slots=True)
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
                    if fact.sentence_index < len(page.passages):
                        sentences.append(page.passages[fact.sentence_index])
                    break
        return sentences


def answer_aliases(record: HotpotRecord) -> list[str]:
    return record.answer_aliases()


def load_records(
    limit: int,
    path: Path = Path("datasets/hotpotqa/distractor-validation.parquet"),
) -> list[HotpotRecord]:
    table = parquet.read_table(
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
    context = table["context"].combine_chunks()
    supporting_facts = table["supporting_facts"].combine_chunks()
    records = []
    for row in range(min(limit, table.num_rows)):
        titles = context.field("title")[row].as_py()
        passages = context.field("sentences")[row].as_py()
        fact_titles = supporting_facts.field("title")[row].as_py()
        fact_indices = supporting_facts.field("sent_id")[row].as_py()
        records.append(
            HotpotRecord(
                id=table["id"][row].as_py(),
                question=table["question"][row].as_py(),
                pages=[
                    SourcePage(title=titles[index], passages=passages[index])
                    for index in range(len(titles))
                ],
                question_type=table["type"][row].as_py(),
                level=table["level"][row].as_py(),
                answer=table["answer"][row].as_py(),
                supporting_facts=[
                    SupportingFact(
                        title=fact_titles[index], sentence_index=fact_indices[index]
                    )
                    for index in range(len(fact_titles))
                ],
            )
        )
    return records


class HotpotQAAdapter:
    name = "hotpotqa"
    display_name = "HotpotQA"

    def load_records(self, limit: int) -> list[HotpotRecord]:
        return load_records(limit)


adapter = HotpotQAAdapter()
