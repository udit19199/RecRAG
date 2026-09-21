from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as parquet

from ..construction.construction import SourcePage


@dataclass(slots=True)
class TriviaRecord:
    id: str
    question: str
    question_source: str
    pages: list[SourcePage]
    answer: str
    aliases: list[str]

    def answer_aliases(self) -> list[str]:
        return self.aliases

    def supporting_sentences(self) -> list[str]:
        return [passage for page in self.pages for passage in page.passages]


def answer_aliases(record: TriviaRecord) -> list[str]:
    return record.answer_aliases()


def load_records(
    limit: int,
    path: Path = Path("datasets/triviaqa/rc.wikipedia-validation.parquet"),
) -> list[TriviaRecord]:
    table = parquet.read_table(
        path,
        columns=[
            "question",
            "question_id",
            "question_source",
            "entity_pages",
            "answer",
        ],
    )
    entity_pages = table["entity_pages"].combine_chunks()
    answers = table["answer"].combine_chunks()
    records = []
    for row in range(min(limit, table.num_rows)):
        titles = entity_pages.field("title")[row].as_py()
        passages = entity_pages.field("wiki_context")[row].as_py()
        answer = answers.field("value")[row].as_py()
        aliases = list(answers.field("aliases")[row].as_py() or [])
        if answer not in aliases:
            aliases.insert(0, answer)
        records.append(
            TriviaRecord(
                id=table["question_id"][row].as_py(),
                question=table["question"][row].as_py(),
                question_source=table["question_source"][row].as_py(),
                pages=[
                    SourcePage(title=titles[index], passages=[passages[index]])
                    for index in range(len(titles))
                ],
                answer=answer,
                aliases=aliases,
            )
        )
    return records


class TriviaQAAdapter:
    name = "triviaqa"
    display_name = "TriviaQA"

    def load_records(self, limit: int) -> list[TriviaRecord]:
        return load_records(limit)


adapter = TriviaQAAdapter()
