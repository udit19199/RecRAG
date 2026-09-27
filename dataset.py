"""Dataset record types, loaders, and the shared source registry.

Keeping these small adapters together makes it easier to compare their record
shapes without scattering the active dataset interface across modules.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class SourcePage:
    title: str
    passages: list[str]


class DatasetRecord(Protocol):
    id: str
    question: str
    answer: str
    pages: Sequence[SourcePage]

    def answer_aliases(self) -> Sequence[str]: ...

    def supporting_sentences(self) -> Sequence[str]: ...


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


def load_hotpotqa_record(index: int) -> HotpotRecord:
    return _hotpot_record(_hotpot_table()[index])


@cache
def _hotpot_table():
    from datasets import load_dataset

    return load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")


def _hotpot_record(row) -> HotpotRecord:
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


@dataclass(slots=True, frozen=True)
class MultiHopRAGRecord:
    id: str
    question: str
    answer: str
    question_type: str
    pages: list[SourcePage]
    evidence: list[str]

    def answer_aliases(self) -> list[str]:
        return [self.answer]

    def supporting_sentences(self) -> list[str]:
        return self.evidence


def load_multihop_rag_record(index: int) -> MultiHopRAGRecord:
    row = _multihop_queries()[index]
    evidence = row["evidence_list"]
    return MultiHopRAGRecord(
        id=f"multihop-rag-{index}",
        question=row["query"],
        answer=row["answer"],
        question_type=row["question_type"],
        pages=_multihop_corpus(),
        evidence=[item["fact"] for item in evidence],
    )


@cache
def _multihop_queries():
    from datasets import load_dataset

    return load_dataset("yixuantt/MultiHopRAG", split="train")


@cache
def _multihop_corpus() -> list[SourcePage]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="yixuantt/MultiHopRAG", repo_type="dataset", filename="corpus.json"
    )
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        SourcePage(
            title=f"{row['title']} ({row['source']}, {row['published_at']})",
            passages=[row["body"]],
        )
        for row in rows
    ]


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


def load_natural_questions_record(index: int) -> NaturalQuestionsRecord:
    return _natural_questions_record(_natural_questions_table()[index])


@cache
def _natural_questions_table():
    from datasets import load_dataset

    return load_dataset(
        "google-research-datasets/natural_questions", "default", split="validation"
    )


def _natural_questions_record(row) -> NaturalQuestionsRecord:
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


@dataclass(slots=True, frozen=True)
class DatasetSource:
    name: str
    display_name: str
    load_record: Callable[[int], DatasetRecord]


DATASET_SOURCES: list[DatasetSource] = [
    DatasetSource(
        name="hotpotqa",
        display_name="HotpotQA",
        load_record=load_hotpotqa_record,
    ),
    DatasetSource(
        name="multihop_rag",
        display_name="MultiHop-RAG",
        load_record=load_multihop_rag_record,
    ),
    DatasetSource(
        name="natural_questions",
        display_name="Natural Questions",
        load_record=load_natural_questions_record,
    ),
]


def get_source(name: str) -> DatasetSource:
    for source in DATASET_SOURCES:
        if source.name == name:
            return source
    raise ValueError(f"Unknown dataset: {name}")
