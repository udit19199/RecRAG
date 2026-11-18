"""Closed-book baseline of the final 500-question study (evals/closed_book.py).

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/final500/scripts/closed_book.py REP
Answers every question of the 3 datasets (500 each) with no retrieved text,
with the same model, reasoning effort, prompts and scorers as
evals/closed_book.py (answer_one). Questions come straight from the frozen
question sets, so this does not depend on any retrieval run. Each question
gets 2 attempts, as in benchmark.py; a question that fails twice is saved with
an empty answer and its error, and scores 0.
Writes runs/final500/closed_book/DATASET/rREP/{results.jsonl,summary.json}.
"""

import json
import os
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[3]
rep = sys.argv[1]
os.environ["RECRAG_HOTPOTQA_ROWS"] = str(
    ROOT / "runs/final500/hotpotqa/questions_rows.json"
)
os.environ["RECRAG_NQ_RECORDS"] = str(
    ROOT / "runs/final500/natural_questions/records.json"
)

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from benchmark import RetrievedContext, SavedRetrieval, Settings  # noqa: E402
from cost import MeteredClient  # noqa: E402
from dataset import get_source  # noqa: E402
from evals.closed_book import answer_one  # noqa: E402
from rag_retrieval import ResponsesLLM  # noqa: E402

load_dotenv(ROOT / ".env", override=True)
settings = Settings.model_validate(tomllib.loads((ROOT / "config.toml").read_text()))
questions = json.loads((ROOT / "runs/final500/questions.json").read_text())
sets = {
    "multihop_rag": questions["multihop_rag"]["all500"],
    "hotpotqa": questions["hotpotqa"]["record_indices"],
    "natural_questions": questions["natural_questions"]["record_indices"],
}


def saved_for(dataset: str, index: int) -> SavedRetrieval:
    record = get_source(dataset).load_record(index)
    return SavedRetrieval(
        method="rag",
        dataset=dataset,
        record_id=record.id,
        question_type=getattr(record, "question_type", None),
        question=record.question,
        canonical_answer=record.answer,
        expected_answers=list(record.answer_aliases()),
        supporting_sentences=[],
        retrieval=RetrievedContext(
            passages=[], scores=[], construction_seconds=0, retrieval_seconds=0
        ),
        index_dir=Path("."),
        storage_mb=0,
        llm_usage=[],
    )


with OpenAI(
    timeout=settings.llm.timeout,
    max_retries=0,
    http_client=MeteredClient(timeout=settings.llm.timeout),
) as client:
    llm = ResponsesLLM(client, settings)

    def one(saved: SavedRetrieval) -> dict:
        errors = []
        for _ in range(2):
            try:
                return {**json.loads(answer_one(llm, settings, saved).model_dump_json()), "errors": errors}
            except Exception as exc:  # noqa: BLE001 - scored as wrong
                errors.append(repr(exc)[:300])
        return {
            "dataset": saved.dataset,
            "record_id": saved.record_id,
            "question_type": saved.question_type,
            "question": saved.question,
            "expected_answers": saved.expected_answers,
            "answer": "",
            "exact_match": 0.0,
            "f1": 0.0,
            "overlap_accuracy": 0.0,
            "question_cost": None,
            "errors": errors,
        }

    for dataset, indices in sets.items():
        out = ROOT / "runs/final500/closed_book" / dataset / f"r{rep}"
        if (out / "summary.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        saved = [saved_for(dataset, i) for i in indices]
        with ThreadPoolExecutor(8) as pool:
            rows = list(pool.map(one, saved))
        (out / "results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        for row in rows:
            cost = row.get("question_cost")
            row["estimated_question_cost_usd"] = (
                sum(c["estimated_usd"] or 0 for c in cost["calls"]) if cost else 0
            )
        (out / "results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "rep": rep,
                    "model": settings.llm.model,
                    "reasoning_effort": settings.llm.reasoning_effort,
                    "count": len(rows),
                    "failed": sum(1 for r in rows if not r["answer"] and r["errors"]),
                    "exact_match": fmean(r["exact_match"] for r in rows),
                    "total_cost_usd": sum(r["estimated_question_cost_usd"] for r in rows),
                },
                indent=1,
            )
        )
        print(dataset, rep, len(rows), flush=True)
