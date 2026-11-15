"""Hand the judgments exact matching cannot make to a Claude Haiku subagent.

Two cases need a judge, and only these are exported:

* answers: exact match is 0 but the model gave an answer (not an abstention).
  Inference answers are free text, so "Sam Bankman-Fried" vs "Bankman-Fried"
  should not count as wrong.
* evidence: retrieved items that have no position in the source text, so the
  offset-based recall in ``evals.retrieval_quality`` scores them as misses:
  Agentic Hybrid RAG's graph-query rows and RAPTOR's generated summaries. Only
  gold sentences the located evidence did not already cover are asked about.

Items are deduplicated by content, so repeats of a run cost no extra judging.
The subagent reads ``items.json`` and writes ``verdicts.json``; no OpenAI calls
are made here. Export with
``uv run python -m evals.subagent_judge export OUT_DIR RUN_DIR ...``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from evals.qa_metrics import normalize_answer
from evals.retrieval_quality import (
    Corpus,
    Gold,
    any_overlap,
    graphrag_items,
    load_corpus,
    load_run,
    locate_gold,
    ranked_items,
)

ABSTENTION = normalize_answer("Insufficient Information")

JUDGE_GUIDE = """You are grading a question-answering benchmark. Judge each item on its own.

For an "answer" item: is the model answer the same answer as any gold answer?
Count it correct when it names the same entity, value or yes/no decision, even
with different wording, extra title words, or a short explanation. Count it
wrong when it is a different entity, hedges between options, says it cannot
answer, or adds a claim that contradicts the gold answer.

For an "evidence" item: list the indexes of the gold sentences whose fact is
stated by the evidence text. A fact is stated when a reader of the evidence
alone would learn it (same entity, same claim, same date or number when the
sentence has one). Be strict: topic overlap or a generic description of the
same subject is not enough. Return an empty list when none.

Write verdicts.json as a JSON array with one object per item, in any order:
{"id": "...", "correct": true}             for answer items
{"id": "...", "supported": [0, 2]}          for evidence items
Use only ids from items.json and do not skip any item."""


class AnswerItem(BaseModel):
    id: str
    kind: str = "answer"
    question: str
    gold_answers: list[str]
    model_answer: str


class EvidenceItem(BaseModel):
    id: str
    kind: str = "evidence"
    question: str
    gold_sentences: list[str]
    evidence: str


class JudgeItems(BaseModel):
    guide: str = JUDGE_GUIDE
    answers: list[AnswerItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class Verdict(BaseModel):
    id: str
    correct: bool | None = None
    supported: list[int] = Field(default_factory=list)


class _Result(BaseModel):
    record_id: str
    question: str
    answer: str
    expected_answers: list[str]
    exact_match: float | None


def _key(*parts: str) -> str:
    return sha256("\0".join(parts).encode()).hexdigest()[:16]


def answer_key(question: str, answer: str) -> str:
    return _key("answer", question, normalize_answer(answer))


def needs_answer_judge(result: _Result) -> bool:
    normalized = normalize_answer(result.answer)
    return (
        result.exact_match == 0
        and bool(normalized)
        and normalized != ABSTENTION
        and bool(result.expected_answers)
    )


def answer_items(run_dirs: Sequence[Path]) -> list[AnswerItem]:
    items: list[AnswerItem] = []
    seen: set[str] = set()
    for run_dir in run_dirs:
        for line in (run_dir / "results.jsonl").read_text().splitlines():
            result = _Result.model_validate_json(line)
            key = answer_key(result.question, result.answer)
            if needs_answer_judge(result) and key not in seen:
                seen.add(key)
                items.append(
                    AnswerItem(
                        id=key,
                        question=result.question,
                        gold_answers=result.expected_answers,
                        model_answer=result.answer,
                    )
                )
    return items


class UnlocatedEvidence(BaseModel):
    """One unlocated evidence text and the gold sentences it could still add."""

    id: str
    evidence: str
    open_gold: list[int]


def unlocated_evidence(
    saved, corpus: Corpus, golds: Sequence[Gold]
) -> list[UnlocatedEvidence]:
    if saved.method == "graphrag":
        from graphrag_retrieval import parse_vector_evidence

        items = graphrag_items(saved, corpus)
        # Graph-query rows: whatever columns the generated query returned.
        # From 9 Nov 2026 a vector_search passage is saved as plain text, so
        # a row is JSON that is not vector_search evidence.
        texts = [
            passage
            for passage in saved.retrieval.passages
            if passage.lstrip().startswith(("{", "["))
            and parse_vector_evidence(passage) is None
        ]
    else:
        items = ranked_items(saved, corpus)
        texts = [item.text for item in items if not item.spans]
    spans = [span for item in items for span in item.spans]
    open_gold = [
        index for index, gold in enumerate(golds) if not any_overlap(gold, spans)
    ]
    if not open_gold:
        return []
    return [
        UnlocatedEvidence(
            id=_key(
                "evidence",
                text,
                *[golds[index].sentence for index in open_gold],
            ),
            evidence=text,
            open_gold=open_gold,
        )
        for text in texts
        if text.strip()
    ]


def evidence_items(run_dirs: Sequence[Path], dataset: str) -> list[EvidenceItem]:
    corpus = load_corpus(dataset)
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for run_dir in run_dirs:
        for saved in load_run(run_dir):
            golds = locate_gold(saved.supporting_sentences, corpus)
            for found in unlocated_evidence(saved, corpus, golds):
                if found.id in seen:
                    continue
                seen.add(found.id)
                items.append(
                    EvidenceItem(
                        id=found.id,
                        question=saved.question,
                        gold_sentences=[
                            golds[index].sentence for index in found.open_gold
                        ],
                        evidence=found.evidence,
                    )
                )
    return items


def load_verdicts(path: Path) -> list[Verdict]:
    return [Verdict.model_validate(row) for row in json.loads(path.read_text())]


def export(out_dir: Path, run_dirs: Sequence[Path], dataset: str, batch: int) -> None:
    """Write items in batches so several Haiku subagents can judge in parallel."""
    answers = answer_items(run_dirs)
    evidence = evidence_items(run_dirs, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [*answers, *evidence]
    for number, start in enumerate(range(0, len(pending), batch)):
        chunk = pending[start : start + batch]
        (out_dir / f"items-{number:03d}.json").write_text(
            JudgeItems(
                answers=[item for item in chunk if isinstance(item, AnswerItem)],
                evidence=[item for item in chunk if isinstance(item, EvidenceItem)],
            ).model_dump_json(indent=1)
        )
    print(f"{len(answers)} answer items, {len(evidence)} evidence items -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("out_dir", type=Path)
    exporter.add_argument("run_dirs", type=Path, nargs="+")
    exporter.add_argument("--dataset", default="multihop_rag")
    exporter.add_argument("--batch", type=int, default=60)
    args = parser.parse_args()
    export(args.out_dir, args.run_dirs, args.dataset, args.batch)


if __name__ == "__main__":
    main()
