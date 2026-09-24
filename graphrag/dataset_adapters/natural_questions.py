from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from ..construction import SourcePage


@dataclass(slots=True, frozen=True)
class NaturalQuestionsRecord:
    id: str
    question: str
    pages: list[SourcePage]
    answer: str
    aliases: list[str]
    evidence: list[str]

    def answer_aliases(self) -> list[str]:
        return self.aliases

    def supporting_sentences(self) -> list[str]:
        return self.evidence


def load_record(index: int) -> NaturalQuestionsRecord:
    return _record(_table()[index])


@cache
def _table():
    from datasets import load_dataset

    return load_dataset(
        "google-research-datasets/natural_questions", "default", split="validation"
    )


def _record(row) -> NaturalQuestionsRecord:
    document = row["document"]
    tokens = document["tokens"]
    text = " ".join(
        token
        for token, is_html in zip(tokens["token"], tokens["is_html"])
        if not is_html and token
    )
    annotation_columns = row["annotations"]
    aliases = []
    evidence = []
    for index, yes_no in enumerate(annotation_columns["yes_no_answer"]):
        if yes_no in (0, 1):
            answer = "no" if yes_no == 0 else "yes"
            if answer not in aliases:
                aliases.append(answer)
        short_answers = annotation_columns["short_answers"]
        for answer_text in short_answers["text"][index]:
            if answer_text and answer_text not in aliases:
                aliases.append(answer_text)
        long_answer = annotation_columns["long_answer"]
        start = long_answer["start_token"][index]
        end = long_answer["end_token"][index]
        if start >= 0 and end > start:
            passage = " ".join(
                token
                for token, is_html in zip(
                    tokens["token"][start:end], tokens["is_html"][start:end]
                )
                if not is_html and token
            )
            if passage and passage not in evidence:
                evidence.append(passage)
            if passage and not aliases:
                aliases.append(passage)
    if not aliases:
        aliases.append("No answer")
    return NaturalQuestionsRecord(
        id=str(row["id"]),
        question=row["question"]["text"],
        pages=[SourcePage(title=document["title"], passages=[text])],
        answer=aliases[0],
        aliases=aliases,
        evidence=evidence,
    )
