"""Answer saved questions with no retrieved text, to measure model knowledge.

Retrieval runs answer from context, but the answer model may also know the
answer. A closed-book run with the same model, reasoning effort and scorers
puts a number on that, so a correct answer with weak retrieval can be traced
to model knowledge instead of explained away.

Run with ``uv run python -m evals.closed_book runs/<method>/<run> ...``; any
method's run works, since only its questions and gold answers are read.
"""

from __future__ import annotations

import argparse
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from benchmark import ROOT, Message, SavedRetrieval, Settings
from cost import MeteredClient, QuestionCost, measure
from evals.qa_metrics import hotpotqa_f1, score_qa_answer
from evals.retrieval_quality import _relative, load_run
from rag_retrieval import ResponsesLLM

# The retrieval runs' instructions without the "only from context" rules.
CLOSED_BOOK_INSTRUCTION = (
    "You are an AI assistant that answers questions from your own knowledge. "
    "You must give ONLY the answer as a single word or phrase. Do not give "
    "additional information."
)
NEWS_CLOSED_BOOK_INSTRUCTION = (
    "You are an AI assistant that answers questions from your own knowledge. "
    "Provide clear, concise, and accurate responses. Follow these rules when "
    "answering:"
    " 1. If the question requires inference (e.g., identifying an entity, "
    "extracting information, or summarizing context), provide the answer as "
    "the relevant entity or a short phrase."
    " 2. For binary questions (e.g., those starting with 'Does', 'Was', 'Is', "
    "etc.), respond ONLY with 'Yes' or 'No'."
)


def closed_book_prompt(question: str) -> str:
    # Same wording as benchmark_prompt, minus the context block.
    return (
        f"\n\nQuestion:\n{question}? "
        "Please just give a short answer without explanation.\n\nAnswer:\n"
    )


class ClosedBookResult(BaseModel):
    dataset: str
    record_id: str
    question_type: str | None
    question: str
    expected_answers: list[str]
    answer: str
    exact_match: float
    f1: float
    overlap_accuracy: float
    question_cost: QuestionCost


class ClosedBookSummary(BaseModel):
    dataset: str
    source_run: str
    model: str
    count: int
    exact_match: float
    f1: float
    overlap_accuracy: float
    total_cost_usd: float | None


def answer_one(
    llm: ResponsesLLM, settings: Settings, saved: SavedRetrieval
) -> ClosedBookResult:
    multihop = saved.dataset == "multihop_rag"
    with measure(settings.pricing) as cost:
        answer = llm.infer(
            [
                Message(
                    role="system",
                    content=NEWS_CLOSED_BOOK_INSTRUCTION
                    if multihop
                    else CLOSED_BOOK_INSTRUCTION,
                ).model_dump(),
                Message(role="user", content=closed_book_prompt(saved.question)).model_dump(),
            ]
        )[0]
    canonical = saved.canonical_answer or ""
    metrics = score_qa_answer(
        answer, saved.expected_answers, canonical, "", multihop_rag=multihop
    )
    f1 = (
        hotpotqa_f1(answer, saved.expected_answers)[2]
        if saved.dataset == "hotpotqa"
        else metrics.f1
    )
    return ClosedBookResult(
        dataset=saved.dataset,
        record_id=saved.record_id,
        question_type=saved.question_type,
        question=saved.question,
        expected_answers=saved.expected_answers,
        answer=answer,
        exact_match=metrics.exact_match,
        f1=f1,
        overlap_accuracy=metrics.overlap_accuracy,
        question_cost=cost,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    settings = Settings.model_validate(tomllib.loads((ROOT / "config.toml").read_text()))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    with OpenAI(
        timeout=settings.llm.timeout,
        # POST is not idempotent, so failed calls are not retried.
        max_retries=0,
        http_client=MeteredClient(timeout=settings.llm.timeout),
    ) as client:
        llm = ResponsesLLM(client, settings)
        for path in args.runs:
            saved = [
                s for s in load_run(path)[: args.limit] if s.expected_answers or s.canonical_answer
            ]
            dataset = saved[0].dataset
            with ThreadPoolExecutor(args.workers) as pool:
                results = list(pool.map(lambda s: answer_one(llm, settings, s), saved))
            output = ROOT / "runs" / "closed_book" / f"{dataset}-{stamp}"
            output.mkdir(parents=True, exist_ok=True)
            (output / "results.jsonl").write_text(
                "".join(result.model_dump_json() + "\n" for result in results),
                encoding="utf-8",
            )
            costs = [result.question_cost.estimated_usd for result in results]
            summary = ClosedBookSummary(
                dataset=dataset,
                source_run=_relative(path),
                model=settings.llm.model,
                count=len(results),
                exact_match=fmean(result.exact_match for result in results),
                f1=fmean(result.f1 for result in results),
                overlap_accuracy=fmean(result.overlap_accuracy for result in results),
                total_cost_usd=sum(c or 0 for c in costs)
                if all(c is not None for c in costs)
                else None,
            )
            (output / "summary.json").write_text(summary.model_dump_json(indent=2))
            print(
                f"{dataset}: {summary.count} questions, closed-book "
                f"F1 {summary.f1:.3f}, EM {summary.exact_match:.3f}, "
                f"accuracy {summary.overlap_accuracy:.3f} -> {_relative(output)}"
            )


if __name__ == "__main__":
    main()
