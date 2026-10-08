from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from statistics import fmean

from pydantic import BaseModel

from cost import QuestionCost


class QABaselineMetrics(BaseModel):
    exact_match: float
    precision: float
    recall: float
    f1: float
    overlap_accuracy: float
    retrieval_accuracy: float


class GraphRAGRecordResult(BaseModel):
    run_id: str
    dataset: str
    record_id: str
    question_type: str | None
    construction_method: str
    retrieval_method: str
    question: str
    expected_answers: list[str]
    answer: str
    retrieved_context: list[str]
    metrics: QABaselineMetrics | None
    construction_seconds: float
    retrieval_and_answer_seconds: float
    question_cost: QuestionCost | None = None
    estimated_question_cost_usd: float | None = None
    complexity: str | None = None


class QuestionTypeSummary(BaseModel):
    question_type: str
    count: int
    metrics: QABaselineMetrics | None


class MethodSummary(BaseModel):
    construction_method: str
    retrieval_method: str
    count: int
    metrics: QABaselineMetrics | None
    question_types: list[QuestionTypeSummary]
    total_question_cost_usd: float | None = None
    mean_retrieval_and_answer_seconds: float = 0


class GraphRAGRunSummary(BaseModel):
    run_id: str
    dataset: str
    count: int
    methods: list[MethodSummary]


def normalize_answer(answer: str) -> str:
    normalized = answer.lower()
    normalized = "".join(
        character for character in normalized if character not in string.punctuation
    )
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(normalized.split())


def hotpotqa_f1(answer: str, expected_answers: Sequence[str]) -> list[float]:
    """Score like the official HotpotQA answer evaluator (best gold alias)."""
    answer = extract_paper_answer(answer)
    prediction = normalize_answer(answer).split()
    best = [0.0, 0.0, 0.0]
    for expected_answer in expected_answers:
        normalized_answer = normalize_answer(expected_answer)
        normalized_prediction = " ".join(prediction)
        if (
            normalized_prediction in {"yes", "no", "noanswer"}
            and normalized_prediction != normalized_answer
        ) or (
            normalized_answer in {"yes", "no", "noanswer"}
            and normalized_prediction != normalized_answer
        ):
            continue
        expected = normalized_answer.split()
        overlap = sum((Counter(prediction) & Counter(expected)).values())
        precision = overlap / len(prediction) if prediction else 0.0
        recall = overlap / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
        if f1 > best[0]:
            best = [f1, precision, recall]
    return [best[1], best[2], best[0]]


def benchmark_prompt(context: str, question: str) -> str:
    return (
        f"Context:\n{context}\n\nQuestion:\n{question}? "
        "Please just give a short answer without explanation.\n\nAnswer:"
    )


def extract_paper_answer(answer: str) -> str:
    """Match the answer wrapper handled by the authors' QA evaluators."""
    answer = answer.strip()
    match = re.search(r'The answer to the question is "(.*?)"', answer)
    return match.group(1) if match else answer


def score_qa_answer(
    answer: str,
    expected_answers: Sequence[str],
    canonical_answer: str,
    prompt: str,
    *,
    multihop_rag: bool = False,
) -> QABaselineMetrics:
    answer = extract_paper_answer(answer)
    aliases = expected_answers or [canonical_answer]
    prediction = normalize_answer(answer).split()
    precision = recall = f1 = 0.0
    for alias in aliases:
        expected = normalize_answer(alias).split()
        overlap = sum((Counter(prediction) & Counter(expected)).values())
        alias_precision = overlap / len(prediction) if prediction else 0.0
        alias_recall = overlap / len(expected) if expected else 0.0
        alias_f1 = (
            2 * alias_precision * alias_recall / (alias_precision + alias_recall)
            if overlap
            else 0.0
        )
        if alias_f1 > f1:
            precision, recall, f1 = alias_precision, alias_recall, alias_f1
    best_exact_match = float(
        normalize_answer(answer).split()
        in [normalize_answer(alias).split() for alias in aliases]
    )

    paper_tokens = set(answer.lower().replace(".", "").split())
    # The published MultiHop-RAG scorer strips periods only from predictions
    # and accepts any shared word without a special yes/no rule.
    gold_tokens = set(
        (canonical_answer if multihop_rag else canonical_answer.replace(".", ""))
        .lower()
        .split()
    )
    if (
        not multihop_rag
        and (
            answer.lower() in {"yes", "no", "noanswer"}
            or canonical_answer.lower() in {"yes", "no", "noanswer"}
        )
        and answer.lower() != canonical_answer.lower()
    ):
        paper_tokens = set()
    overlap_accuracy = float(bool(paper_tokens & gold_tokens))
    if multihop_rag:
        # The authors count each answer as one binary prediction, so all
        # three aggregate metrics equal their overlap accuracy.
        precision = recall = f1 = overlap_accuracy
    return QABaselineMetrics(
        exact_match=best_exact_match,
        precision=precision,
        recall=recall,
        f1=f1,
        overlap_accuracy=overlap_accuracy,
        retrieval_accuracy=float(canonical_answer in prompt),
    )


def score_novelqa_answer(answer: str, gold: str, prompt: str) -> QABaselineMetrics:
    """The paper evaluator compares the first letter after trimming quotes."""
    prediction = answer.strip().strip("'").lower()[:1]
    expected = gold.strip().lower()
    correct = float(prediction == expected)
    return QABaselineMetrics(
        exact_match=correct,
        precision=correct,
        recall=correct,
        f1=correct,
        overlap_accuracy=correct,
        retrieval_accuracy=float(expected in prompt),
    )


def summarize_qa_results(
    run_id: str,
    dataset: str,
    results: Sequence[GraphRAGRecordResult],
) -> GraphRAGRunSummary:
    method_summaries = []
    construction_methods = sorted({result.construction_method for result in results})
    for construction_method in construction_methods:
        retrieval_methods = sorted(
            {
                result.retrieval_method
                for result in results
                if result.construction_method == construction_method
            }
        )
        for retrieval_method in retrieval_methods:
            selected = [
                result
                for result in results
                if result.construction_method == construction_method
                and result.retrieval_method == retrieval_method
            ]
            scored = [
                result.metrics for result in selected if result.metrics is not None
            ]
            metrics = _mean_metrics(scored) if scored else None
            question_types = []
            names = sorted(
                {
                    result.question_type
                    for result in selected
                    if result.question_type is not None
                }
            )
            for question_type in names:
                category = [
                    result
                    for result in selected
                    if result.question_type == question_type
                ]
                category_metrics = [
                    result.metrics for result in category if result.metrics is not None
                ]
                question_types.append(
                    QuestionTypeSummary(
                        question_type=question_type,
                        count=len(category),
                        metrics=(
                            _mean_metrics(category_metrics)
                            if category_metrics
                            else None
                        ),
                    )
                )
            method_summaries.append(
                MethodSummary(
                    construction_method=construction_method,
                    retrieval_method=retrieval_method,
                    count=len(selected),
                    metrics=metrics,
                    question_types=question_types,
                    total_question_cost_usd=(
                        sum(
                            result.estimated_question_cost_usd or 0
                            for result in selected
                        )
                        if all(
                            result.estimated_question_cost_usd is not None
                            for result in selected
                        )
                        else None
                    ),
                    mean_retrieval_and_answer_seconds=fmean(
                        result.retrieval_and_answer_seconds for result in selected
                    ),
                )
            )
    return GraphRAGRunSummary(
        run_id=run_id,
        dataset=dataset,
        count=len(results),
        methods=method_summaries,
    )


def _mean_metrics(metrics: Sequence[QABaselineMetrics]) -> QABaselineMetrics:
    return QABaselineMetrics(
        exact_match=fmean(metric.exact_match for metric in metrics),
        precision=fmean(metric.precision for metric in metrics),
        recall=fmean(metric.recall for metric in metrics),
        f1=fmean(metric.f1 for metric in metrics),
        overlap_accuracy=fmean(metric.overlap_accuracy for metric in metrics),
        retrieval_accuracy=fmean(metric.retrieval_accuracy for metric in metrics),
    )
