from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from ..construction import SourcePage


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


def load_record(index: int) -> MultiHopRAGRecord:
    row = _queries()[index]
    evidence = row["evidence_list"]
    return MultiHopRAGRecord(
        id=f"multihop-rag-{index}",
        question=row["query"],
        answer=row["answer"],
        question_type=row["question_type"],
        pages=_corpus(),
        evidence=[item["fact"] for item in evidence],
    )


@cache
def _queries():
    from datasets import load_dataset

    return load_dataset("yixuantt/MultiHopRAG", split="train")


@cache
def _corpus() -> list[SourcePage]:
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
