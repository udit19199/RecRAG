"""Extended evaluation for FiNER-139 graph-construction method comparison.

Builds on ``scoring.score`` with metrics suited to technique comparison:
macro vs micro aggregation, partial (IoU) matching, error taxonomy, bootstrap
confidence intervals, concept-stratified recall, and cross-method head-to-head.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from experiments.finer139.scoring import Metrics, _count, _metrics, numeric_filter, score
from experiments.finer139.types import Sentence, Span

IOU_MATCH_THRESHOLD = 0.5
DEFAULT_BOOTSTRAP_SAMPLES = 1000


def span_iou(a: Span, b: Span) -> float:
    """Token-index intersection-over-union for two half-open spans."""
    inter_start = max(a.start, b.start)
    inter_end = min(a.end, b.end)
    if inter_start >= inter_end:
        return 0.0
    inter = inter_end - inter_start
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union else 0.0


def _dedupe_preds(preds: list[Span]) -> list[Span]:
    seen: set[tuple[int, int]] = set()
    out: list[Span] = []
    for p in preds:
        key = (p.start, p.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _partial_count(pred: list[Span], gold: list[Span]) -> tuple[int, int, int]:
    """TP when best IoU with an unmatched gold span >= threshold."""
    matched_gold: set[int] = set()
    tp = 0
    for p in pred:
        best_iou = 0.0
        best_gi: int | None = None
        for gi, g in enumerate(gold):
            if gi in matched_gold:
                continue
            iou = span_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi is not None and best_iou >= IOU_MATCH_THRESHOLD:
            matched_gold.add(best_gi)
            tp += 1
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def _error_breakdown(pred: list[Span], gold: list[Span]) -> dict[str, int]:
    """Classify predictions: strict TP, boundary (overlap, wrong span), spurious."""
    strict_tp, _, _ = _count(pred, gold, relaxed=False)
    matched_gold: set[int] = set()
    boundary = 0
    spurious = 0

    for p in pred:
        if any(p.start == g.start and p.end == g.end for g in gold):
            continue
        best_iou = 0.0
        best_gi: int | None = None
        for gi, g in enumerate(gold):
            iou = span_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou > 0 and best_gi is not None and best_gi not in matched_gold:
            boundary += 1
            matched_gold.add(best_gi)
        elif best_iou == 0:
            spurious += 1

    _, _, fn = _count(pred, gold, relaxed=False)
    return {
        "strict_tp": strict_tp,
        "boundary_fp": boundary,
        "spurious_fp": spurious,
        "missed_fn": fn,
    }


def _sentence_strict_f1(pred: list[Span], gold: list[Span]) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    tp, fp, fn = _count(pred, gold, relaxed=False)
    return _metrics(tp, fp, fn).f1


@dataclass
class ExtendedScore:
    strict: Metrics
    relaxed: Metrics
    partial: Metrics
    macro_strict: Metrics
    macro_relaxed: Metrics
    macro_partial: Metrics
    span_iou_mean: float
    sentence_hit_rate: float
    errors: dict[str, int]
    bootstrap_strict_f1: tuple[float, float]
    concept_recall: list[dict[str, float | int | str]]


def bootstrap_f1_ci(
    sentences: list[Sentence],
    predictions: list[list[Span]],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI for strict micro-F1 via sentence-level bootstrap resampling."""
    if not sentences:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(sentences)
    f1s: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        s_tp = s_fp = s_fn = 0
        for i in indices:
            gold = sentences[i].gold_spans
            pred = _dedupe_preds(numeric_filter(predictions[i], sentences[i]))
            tp, fp, fn = _count(pred, gold, relaxed=False)
            s_tp += tp
            s_fp += fp
            s_fn += fn
        f1s.append(_metrics(s_tp, s_fp, s_fn).f1)
    f1s.sort()
    lo = f1s[int(0.025 * len(f1s))]
    hi = f1s[int(0.975 * len(f1s)) - 1]
    return (lo, hi)


def _concept_recall(
    sentences: list[Sentence], predictions: list[list[Span]], top_k: int | None = 10
) -> list[dict[str, float | int | str]]:
    """Per-XBRL-concept strict recall (gold labels only).

    When ``top_k`` is None, return all concepts with gold counts.
    """
    concept_gold: Counter[str] = Counter()
    concept_tp: Counter[str] = Counter()

    for sentence, preds in zip(sentences, predictions):
        filtered = _dedupe_preds(numeric_filter(preds, sentence))
        gold = sentence.gold_spans
        for g in gold:
            label = g.label or "Unknown"
            concept_gold[label] += 1
        matched: set[int] = set()
        for p in filtered:
            for gi, g in enumerate(gold):
                if gi in matched:
                    continue
                if p.start == g.start and p.end == g.end:
                    matched.add(gi)
                    concept_tp[g.label or "Unknown"] += 1
                    break

    rows: list[dict[str, float | int | str]] = []
    items = (
        concept_gold.most_common(top_k)
        if top_k is not None
        else concept_gold.most_common()
    )
    for concept, count in items:
        tp = concept_tp[concept]
        rows.append(
            {
                "concept": concept,
                "gold_count": count,
                "tp": tp,
                "recall": tp / count if count else 0.0,
            }
        )
    return rows


def analyze(
    sentences: list[Sentence],
    predictions: list[list[Span]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
) -> ExtendedScore:
    """Full evaluation suite for one method's predictions."""
    base = score(sentences, predictions)

    s_tp = s_fp = s_fn = 0
    r_tp = r_fp = r_fn = 0
    p_tp = p_fp = p_fn = 0
    macro_s_f1: list[float] = []
    macro_r_f1: list[float] = []
    macro_p_f1: list[float] = []
    ious: list[float] = []
    sentences_with_hit = 0
    errors: Counter[str] = Counter()

    for sentence, preds in zip(sentences, predictions):
        gold = sentence.gold_spans
        filtered = _dedupe_preds(numeric_filter(preds, sentence))

        tp, fp, fn = _count(filtered, gold, relaxed=False)
        s_tp, s_fp, s_fn = s_tp + tp, s_fp + fp, s_fn + fn
        tp, fp, fn = _count(filtered, gold, relaxed=True)
        r_tp, r_fp, r_fn = r_tp + tp, r_fp + fp, r_fn + fn
        tp, fp, fn = _partial_count(filtered, gold)
        p_tp, p_fp, p_fn = p_tp + tp, p_fp + fp, p_fn + fn

        if gold:
            macro_s_f1.append(_sentence_strict_f1(filtered, gold))
            macro_r_f1.append(_metrics(*_count(filtered, gold, relaxed=True)).f1)
            macro_p_f1.append(_metrics(*_partial_count(filtered, gold)).f1)
            if any(
                p.start == g.start and p.end == g.end for p in filtered for g in gold
            ):
                sentences_with_hit += 1

        breakdown = _error_breakdown(filtered, gold)
        for k, v in breakdown.items():
            errors[k] += v

        matched_gold: set[int] = set()
        for p in filtered:
            best_iou = 0.0
            best_gi: int | None = None
            for gi, g in enumerate(gold):
                if gi in matched_gold:
                    continue
                iou = span_iou(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi is not None and best_iou > 0:
                ious.append(best_iou)
                if best_iou >= IOU_MATCH_THRESHOLD:
                    matched_gold.add(best_gi)

    n_with_gold = sum(1 for s in sentences if s.gold_spans)
    ci = bootstrap_f1_ci(
        sentences, predictions, samples=bootstrap_samples, seed=seed
    )

    def _macro_avg(values: list[float]) -> Metrics:
        if not values:
            return Metrics(0.0, 0.0, 0.0, 0, 0, 0)
        avg_f1 = sum(values) / len(values)
        return Metrics(avg_f1, avg_f1, avg_f1, 0, 0, 0)

    return ExtendedScore(
        strict=base.strict,
        relaxed=base.relaxed,
        partial=_metrics(p_tp, p_fp, p_fn),
        macro_strict=_macro_avg(macro_s_f1),
        macro_relaxed=_macro_avg(macro_r_f1),
        macro_partial=_macro_avg(macro_p_f1),
        span_iou_mean=sum(ious) / len(ious) if ious else 0.0,
        sentence_hit_rate=sentences_with_hit / n_with_gold if n_with_gold else 0.0,
        errors=dict(errors),
        bootstrap_strict_f1=ci,
        concept_recall=_concept_recall(sentences, predictions),
    )


def extended_to_dict(ext: ExtendedScore) -> dict:
    return {
        "strict": asdict(ext.strict),
        "relaxed": asdict(ext.relaxed),
        "partial": asdict(ext.partial),
        "macro_strict": asdict(ext.macro_strict),
        "macro_relaxed": asdict(ext.macro_relaxed),
        "macro_partial": asdict(ext.macro_partial),
        "span_iou_mean": round(ext.span_iou_mean, 4),
        "sentence_hit_rate": round(ext.sentence_hit_rate, 4),
        "errors": ext.errors,
        "bootstrap_strict_f1_ci": {
            "low": round(ext.bootstrap_strict_f1[0], 4),
            "high": round(ext.bootstrap_strict_f1[1], 4),
        },
        "concept_recall": ext.concept_recall,
    }


def _sentence_has_strict_hit(preds: list[Span], gold: list[Span]) -> bool:
    tp, _, _ = _count(preds, gold, relaxed=False)
    return tp > 0


def paired_bootstrap_delta(
    sentences: list[Sentence],
    preds_a: list[list[Span]],
    preds_b: list[list[Span]],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
) -> dict[str, float | bool]:
    """Bootstrap 95% CI for strict micro-F1 delta (B − A) on paired sentences."""
    if not sentences:
        return {"delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "significant": False}

    rng = random.Random(seed)
    n = len(sentences)
    deltas: list[float] = []
    for _ in range(samples):
        tp_a = fp_a = fn_a = 0
        tp_b = fp_b = fn_b = 0
        for _ in range(n):
            i = rng.randrange(n)
            gold = sentences[i].gold_spans
            pa = _dedupe_preds(numeric_filter(preds_a[i], sentences[i]))
            pb = _dedupe_preds(numeric_filter(preds_b[i], sentences[i]))
            t, f, fn = _count(pa, gold, relaxed=False)
            tp_a, fp_a, fn_a = tp_a + t, fp_a + f, fn_a + fn
            t, f, fn = _count(pb, gold, relaxed=False)
            tp_b, fp_b, fn_b = tp_b + t, fp_b + f, fn_b + fn
        f1_a = _metrics(tp_a, fp_a, fn_a).f1
        f1_b = _metrics(tp_b, fp_b, fn_b).f1
        deltas.append(f1_b - f1_a)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1]
    return {
        "delta_mean": round(sum(deltas) / len(deltas), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "significant": lo > 0 or hi < 0,
    }


def mcnemar_sentence_hits(
    sentences: list[Sentence],
    preds_a: list[list[Span]],
    preds_b: list[list[Span]],
) -> dict[str, int]:
    """McNemar-style discordant counts on per-sentence strict-hit (any TP)."""
    a_only = b_only = both = neither = 0
    for i, sentence in enumerate(sentences):
        if not sentence.gold_spans:
            continue
        pa = _dedupe_preds(numeric_filter(preds_a[i], sentence))
        pb = _dedupe_preds(numeric_filter(preds_b[i], sentence))
        ha = _sentence_has_strict_hit(pa, sentence.gold_spans)
        hb = _sentence_has_strict_hit(pb, sentence.gold_spans)
        if ha and hb:
            both += 1
        elif ha:
            a_only += 1
        elif hb:
            b_only += 1
        else:
            neither += 1
    return {
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "discordant": a_only + b_only,
    }


def compare_methods_paired(
    sentences: list[Sentence],
    all_preds: dict[str, list[list[Span]]],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Pairwise bootstrap + McNemar for every scored method pair."""
    from itertools import combinations

    names = sorted(all_preds.keys())
    rows: list[dict[str, Any]] = []
    for a, b in combinations(names, 2):
        delta = paired_bootstrap_delta(
            sentences, all_preds[a], all_preds[b], seed=seed
        )
        mcnemar = mcnemar_sentence_hits(sentences, all_preds[a], all_preds[b])
        rows.append(
            {
                "method_a": a,
                "method_b": b,
                "delta_b_minus_a": delta,
                "mcnemar": mcnemar,
                "interpretation": (
                    f"{b} significantly better than {a}"
                    if delta["ci_low"] > 0
                    else f"{a} significantly better than {b}"
                    if delta["ci_high"] < 0
                    else f"No significant difference between {a} and {b}"
                ),
            }
        )
    return rows


def aggregate_multi_seed(per_seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize strict F1 mean ± std across multiple seeded runs."""
    method_f1: dict[str, list[float]] = defaultdict(list)
    method_partial: dict[str, list[float]] = defaultdict(list)
    seeds: list[int] = []

    for run in per_seed_runs:
        seeds.append(run.get("params", {}).get("seed", -1))
        for m in run.get("methods", []):
            if m.get("error"):
                continue
            strict = m.get("strict")
            if strict:
                method_f1[m["name"]].append(float(strict["f1"]))
            partial = m.get("partial")
            if partial:
                method_partial[m["name"]].append(float(partial["f1"]))

    by_method: dict[str, dict[str, float | int]] = {}
    for name, f1s in method_f1.items():
        mean = sum(f1s) / len(f1s)
        std = (sum((x - mean) ** 2 for x in f1s) / len(f1s)) ** 0.5 if len(f1s) > 1 else 0.0
        by_method[name] = {
            "strict_f1_mean": round(mean, 4),
            "strict_f1_std": round(std, 4),
            "strict_f1_min": round(min(f1s), 4),
            "strict_f1_max": round(max(f1s), 4),
            "n_seeds": len(f1s),
        }
        partials = method_partial.get(name, [])
        if partials:
            pmean = sum(partials) / len(partials)
            by_method[name]["partial_f1_mean"] = round(pmean, 4)

    ranking = sorted(
        by_method.items(),
        key=lambda x: float(x[1]["strict_f1_mean"]),
        reverse=True,
    )
    return {
        "seeds": seeds,
        "n_runs": len(per_seed_runs),
        "by_method": by_method,
        "ranking": [name for name, _ in ranking],
    }


def compare_methods_head_to_head(
    sentences: list[Sentence],
    all_preds: dict[str, list[list[Span]]],
) -> dict[str, Any]:
    """Per-method count of sentences where that method has the best strict F1."""
    methods = list(all_preds.keys())
    wins: Counter[str] = Counter()
    ties = 0
    eligible = 0

    for i, sentence in enumerate(sentences):
        if not sentence.gold_spans:
            continue
        eligible += 1
        best_f1 = -1.0
        best: list[str] = []
        for name in methods:
            preds = _dedupe_preds(
                numeric_filter(all_preds[name][i], sentence)
            )
            f1 = _sentence_strict_f1(preds, sentence.gold_spans)
            if f1 > best_f1:
                best_f1 = f1
                best = [name]
            elif f1 == best_f1:
                best.append(name)
        if len(best) == 1:
            wins[best[0]] += 1
        else:
            ties += 1

    return {
        "sentence_wins": dict(wins),
        "sentences_with_gold": eligible,
        "ties": ties,
    }
