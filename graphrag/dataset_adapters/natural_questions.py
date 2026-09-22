from __future__ import annotations

import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..construction import SourcePage

NQ_PATH = Path("datasets/natural_questions/dev.jsonl")


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _tokens(record: dict[str, Any]) -> list[str]:
    tokens = record["document_tokens"]
    return [
        str(item["token"])
        for item in tokens
        if not item["html_token"] and item["token"]
    ]


def _document_text(record: dict[str, Any], tokens: list[str]) -> str:
    if tokens:
        return " ".join(tokens)
    parser = _TextParser()
    parser.feed(str(record["document_html"]))
    return " ".join(" ".join(parser.parts).split())


def _span(record: dict[str, Any], answer: dict[str, Any]) -> str:
    start = int(answer["start_token"])
    end = int(answer["end_token"])
    return " ".join(
        str(item["token"])
        for item in record["document_tokens"][start:end]
        if not item["html_token"]
    ).strip()


def _answers(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for annotation in record["annotations"]:
        yes_no = str(annotation["yes_no_answer"]).casefold()
        if yes_no in {"yes", "no"}:
            values.append(yes_no)
        for answer in annotation["short_answers"]:
            value = _span(record, answer)
            if value and value not in values:
                values.append(value)
        long_answer = annotation["long_answer"]
        value = _span(record, long_answer)
        if value and value not in values:
            values.append(value)
    return values


@dataclass(slots=True, frozen=True)
class NaturalQuestionsRecord:
    id: str
    question: str
    pages: list[SourcePage]
    answer: str
    aliases: list[str]
    supporting_passages: list[str]

    def answer_aliases(self) -> list[str]:
        return self.aliases

    def supporting_sentences(self) -> list[str]:
        return self.supporting_passages


def load_record(index: int) -> NaturalQuestionsRecord:
    with open(NQ_PATH, "rt", encoding="utf-8") as lines:
        for position, line in enumerate(lines):
            if position == index:
                raw: dict[str, Any] = json.loads(line)
                return _record(raw)
    raise IndexError(f"Record index out of range: {index}")


def _record(raw: dict[str, Any]) -> NaturalQuestionsRecord:
    tokens = _tokens(raw)
    aliases = _answers(raw)
    supporting_passages = []
    for annotation in raw["annotations"]:
        passage = _span(raw, annotation["long_answer"])
        if passage and passage not in supporting_passages:
            supporting_passages.append(passage)
    text = _document_text(raw, tokens)
    return NaturalQuestionsRecord(
        id=str(raw["example_id"]),
        question=str(raw["question_text"]),
        pages=[SourcePage(title=str(raw["document_title"]), passages=[text])],
        answer=aliases[0],
        aliases=aliases,
        supporting_passages=supporting_passages,
    )
