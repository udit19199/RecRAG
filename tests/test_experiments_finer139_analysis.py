"""Tests for extended FiNER-139 evaluation (analysis module)."""

from experiments.finer139.analysis import (
    analyze,
    bootstrap_f1_ci,
    compare_methods_head_to_head,
    span_iou,
)
from experiments.finer139.types import Sentence, Span


def _sentence(tokens: list[str], gold: list[Span]) -> Sentence:
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(tok)
        cursor += len(tok)
        offsets.append((start, cursor))
    return Sentence(
        index=0,
        tokens=tokens,
        text="".join(parts),
        token_offsets=offsets,
        gold_spans=gold,
    )


def test_span_iou_exact_and_partial() -> None:
    a = Span(2, 4)
    b = Span(2, 4)
    assert span_iou(a, b) == 1.0
    c = Span(2, 3)
    assert 0 < span_iou(c, b) < 1.0
    d = Span(5, 6)
    assert span_iou(a, d) == 0.0


def test_analyze_includes_partial_and_macro() -> None:
    s = _sentence(
        ["revenue", "was", "100", "million"],
        [Span(2, 3, "Revenue", "100")],
    )
    preds = [[Span(2, 4, None, "100 million")]]  # boundary error vs gold
    ext = analyze([s], preds, bootstrap_samples=50, seed=1)
    assert ext.strict.f1 == 0.0
    assert ext.partial.f1 > 0.0
    assert ext.macro_strict.f1 == 0.0
    assert ext.errors["boundary_fp"] >= 1
    assert ext.bootstrap_strict_f1[0] <= ext.bootstrap_strict_f1[1]


def test_compare_methods_head_to_head() -> None:
    s = _sentence(["x", "100"], [Span(1, 2, "A", "100")])
    preds = {
        "good": [[Span(1, 2)]],
        "bad": [[Span(0, 1)]],
    }
    cmp = compare_methods_head_to_head([s], preds)
    assert cmp["sentence_wins"]["good"] == 1
    assert cmp["sentences_with_gold"] == 1


def test_bootstrap_ci_bounds() -> None:
    s = _sentence(["100"], [Span(0, 1, "X", "100")])
    preds = [[Span(0, 1)]]
    lo, hi = bootstrap_f1_ci([s], preds, samples=100, seed=0)
    assert lo <= hi
    assert hi == 1.0
