from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..construction import SourcePage

DATA_PATH = Path("datasets/browsecomp_plus/test.jsonl")


@dataclass(slots=True, frozen=True)
class BrowseCompPlusRecord:
    id: str
    question: str
    pages: list[SourcePage]
    answer: str
    evidence: list[str]

    def answer_aliases(self) -> list[str]:
        return [self.answer]

    def supporting_sentences(self) -> list[str]:
        return self.evidence


def load_record(index: int) -> BrowseCompPlusRecord:
    with DATA_PATH.open("rt", encoding="utf-8") as lines:
        for position, line in enumerate(lines):
            if position == index:
                return _record(json.loads(line))
    raise IndexError(f"Record index out of range: {index}")


def _record(row: Any) -> BrowseCompPlusRecord:
    pages = []
    evidence = []
    seen_docids = set()
    for field in ("evidence_docs", "gold_docs", "negative_docs"):
        for document in row[field]:
            if field == "evidence_docs":
                evidence.append(str(document["text"]))
            docid = str(document["docid"])
            if docid not in seen_docids:
                seen_docids.add(docid)
                pages.append(SourcePage(title=docid, passages=[str(document["text"])]))
    return BrowseCompPlusRecord(
        id=str(row["query_id"]),
        question=str(row["query"]),
        pages=pages,
        answer=str(row["answer"]),
        evidence=evidence,
    )
