from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from statistics import fmean

from pydantic import BaseModel


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
    metrics: QABaselineMetrics
    construction_seconds: float
    retrieval_and_answer_seconds: float


class QuestionTypeSummary(BaseModel):
    question_type: str
    count: int
    metrics: QABaselineMetrics


class MethodSummary(BaseModel):
    construction_method: str
    retrieval_method: str
    count: int
    metrics: QABaselineMetrics
    question_types: list[QuestionTypeSummary]


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


def benchmark_prompt(context: str, question: str) -> str:
    return (
        f"Context:\n{context}\n\nQuestion:\n{question}? "
        "Please just give a short answer without explanation.\n\nAnswer:"
    )


def score_qa_answer(
    answer: str,
    expected_answers: Sequence[str],
    canonical_answer: str,
    prompt: str,
) -> QABaselineMetrics:
    prediction = normalize_answer(answer).split()
    aliases = expected_answers or [canonical_answer]
    best_exact_match = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0
    for alias in aliases:
        expected = normalize_answer(alias).split()
        overlap = sum((Counter(prediction) & Counter(expected)).values())
        precision = overlap / len(prediction) if prediction else 0.0
        recall = overlap / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
        if f1 > best_f1:
            best_exact_match = float(prediction == expected)
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    paper_tokens = set(answer.lower().replace(".", "").split())
    gold_tokens = set(canonical_answer.lower().replace(".", "").split())
    return QABaselineMetrics(
        exact_match=best_exact_match,
        precision=best_precision,
        recall=best_recall,
        f1=max(best_f1, 0.0),
        overlap_accuracy=float(bool(paper_tokens & gold_tokens)),
        retrieval_accuracy=float(canonical_answer in prompt),
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
            metrics = _mean_metrics([result.metrics for result in selected])
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
                question_types.append(
                    QuestionTypeSummary(
                        question_type=question_type,
                        count=len(category),
                        metrics=_mean_metrics([result.metrics for result in category]),
                    )
                )
            method_summaries.append(
                MethodSummary(
                    construction_method=construction_method,
                    retrieval_method=retrieval_method,
                    count=len(selected),
                    metrics=metrics,
                    question_types=question_types,
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
