from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..construction.construction import SourcePage

DATASET_NAME = "natural_questions"
DEFAULT_PATHS = [
    Path("datasets/natural_questions/dev.jsonl"),
    Path("datasets/natural_questions/v1.0-dev.jsonl"),
    Path("datasets/natural_questions/v1.0-simplified-nq-dev.jsonl"),
    Path("datasets/natural_questions/v1.0-dev.jsonl.gz"),
    Path("datasets/natural_questions/v1.0-simplified-nq-dev.jsonl.gz"),
]


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _tokens(record: Any) -> list[str]:
    tokens = record.get("document_tokens", [])
    return [
        str(item["token"])
        for item in tokens
        if not item.get("html_token") and item.get("token")
    ]


def _document_text(record: Any, tokens: list[str]) -> str:
    if tokens:
        return " ".join(tokens)
    parser = _TextParser()
    parser.feed(str(record.get("document_html", "")))
    return " ".join(" ".join(parser.parts).split())


def _span(tokens: list[str], answer: Any) -> str:
    start = int(answer.get("start_token", 0))
    end = int(answer.get("end_token", start))
    return " ".join(tokens[start:end]).strip()


def _answers(record: Any, tokens: list[str]) -> list[str]:
    values: list[str] = []
    for annotation in record.get("annotations", []):
        yes_no = str(annotation.get("yes_no_answer", "NONE")).casefold()
        if yes_no in {"yes", "no"}:
            values.append(yes_no)
        for answer in annotation.get("short_answers", []):
            value = _span(tokens, answer)
            if value and value not in values:
                values.append(value)
        long_answer = annotation.get("long_answer", {})
        value = _span(tokens, long_answer)
        if value and value not in values:
            values.append(value)
    return values


@dataclass(slots=True)
class NaturalQuestionsRecord:
    id: str
    question: str
    pages: list[SourcePage]
    answer: str
    aliases: list[str]

    def answer_aliases(self) -> list[str]:
        return self.aliases

    def supporting_sentences(self) -> list[str]:
        return [self.answer] if self.answer else []


def _path() -> Path:
    for path in DEFAULT_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Natural Questions JSONL is missing. Put the dev file under "
        "datasets/natural_questions/."
    )


def load_records(limit: int) -> list[NaturalQuestionsRecord]:
    path = _path()
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[NaturalQuestionsRecord] = []
    with opener(path, "rt", encoding="utf-8") as lines:
        for line in lines:
            if len(records) >= limit:
                break
            raw = json.loads(line)
            tokens = _tokens(raw)
            text = _document_text(raw, tokens)
            aliases = _answers(raw, tokens)
            answer = aliases[0] if aliases else ""
            title = str(raw.get("document_title") or raw.get("document_url", ""))
            records.append(
                NaturalQuestionsRecord(
                    id=str(raw["example_id"]),
                    question=str(raw["question_text"]),
                    pages=[SourcePage(title=title, passages=[text])],
                    answer=answer,
                    aliases=aliases,
                )
            )
    return records


class NaturalQuestionsAdapter:
    name = DATASET_NAME
    display_name = "Natural Questions"

    def load_records(self, limit: int) -> list[NaturalQuestionsRecord]:
        return load_records(limit)


adapter = NaturalQuestionsAdapter()
