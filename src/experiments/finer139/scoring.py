"""Scoring for the FiNER-139 detection benchmark.

Type-agnostic span detection with a numeric-only evaluation universe: every
method's predictions are filtered to numeric expressions before matching, so
open-ended methods are not penalized for extracting non-numeric entities that
FiNER never annotates (gold is numeric-only, so recall is unaffected).

Matching modes:
  - strict:  exact token-span match (same start and end)
  - relaxed: any token-index overlap
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.finer139.schema import is_numeric
from experiments.finer139.types import Sentence, Span


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _span_surface(span: Span, sentence: Sentence) -> str:
    if span.text:
        return span.text
    return " ".join(sentence.tokens[span.start : span.end])


def numeric_filter(spans: list[Span], sentence: Sentence) -> list[Span]:
    """Keep only spans whose surface text is a numeric expression."""
    return [s for s in spans if is_numeric(_span_surface(s, sentence))]


def _count(pred: list[Span], gold: list[Span], relaxed: bool) -> tuple[int, int, int]:
    matched: set[int] = set()
    tp = 0
    for p in pred:
        for gi, g in enumerate(gold):
            if gi in matched:
                continue
            if relaxed:
                hit = p.start < g.end and g.start < p.end
            else:
                hit = p.start == g.start and p.end == g.end
            if hit:
                matched.add(gi)
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def _metrics(tp: int, fp: int, fn: int) -> Metrics:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return Metrics(precision, recall, f1, tp, fp, fn)


@dataclass
class ScoreResult:
    strict: Metrics
    relaxed: Metrics
    num_pred: int
    num_gold: int


def score(
    sentences: list[Sentence], predictions: list[list[Span]]
) -> ScoreResult:
    """Micro-average strict and relaxed metrics over all sentences.

    Predictions are numeric-filtered here so all methods share the same fair
    evaluation universe.
    """
    s_tp = s_fp = s_fn = 0
    r_tp = r_fp = r_fn = 0
    total_pred = 0
    total_gold = 0

    for sentence, preds in zip(sentences, predictions):
        gold = sentence.gold_spans
        filtered = numeric_filter(preds, sentence)
        # Dedup by token range within the sentence.
        seen: set[tuple[int, int]] = set()
        deduped: list[Span] = []
        for p in filtered:
            key = (p.start, p.end)
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        total_pred += len(deduped)
        total_gold += len(gold)

        tp, fp, fn = _count(deduped, gold, relaxed=False)
        s_tp, s_fp, s_fn = s_tp + tp, s_fp + fp, s_fn + fn
        tp, fp, fn = _count(deduped, gold, relaxed=True)
        r_tp, r_fp, r_fn = r_tp + tp, r_fp + fp, r_fn + fn

    return ScoreResult(
        strict=_metrics(s_tp, s_fp, s_fn),
        relaxed=_metrics(r_tp, r_fp, r_fn),
        num_pred=total_pred,
        num_gold=total_gold,
    )


def metrics_to_dict(m: Metrics) -> dict[str, float | int]:
    return asdict(m)
