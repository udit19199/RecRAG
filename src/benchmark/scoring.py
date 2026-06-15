"""Composite benchmark scoring with constraint weights (BM-1)."""

from __future__ import annotations

from orchestration.models import BenchmarkSummary, ConfidenceLevel


def composite_score(
    scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    total_w = sum(weights.values()) or 1.0
    value = 0.0
    for metric, weight in weights.items():
        value += scores.get(metric, 0.0) * weight
    return round(value / total_w, 4)


def build_summary(
    scores: dict[str, float],
    weights: dict[str, float],
    candidates_evaluated: int,
    margin: float | None = None,
) -> BenchmarkSummary:
    return BenchmarkSummary(
        confidence=ConfidenceLevel.MEASURED,
        candidates_evaluated=candidates_evaluated,
        faithfulness=scores.get("faithfulness"),
        answer_relevancy=scores.get("answer_relevancy"),
        context_precision=scores.get("context_precision"),
        context_recall=scores.get("context_recall"),
        composite_score=composite_score(scores, weights),
        margin_over_runner_up=margin,
    )
