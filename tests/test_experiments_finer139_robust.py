"""Robust evaluation protocol v3 tests (offline, no network)."""

from __future__ import annotations

from pathlib import Path

from experiments.finer139.analysis import (
    aggregate_multi_seed,
    compare_methods_paired,
    mcnemar_sentence_hits,
    paired_bootstrap_delta,
)
from experiments.finer139.dataset import load_fixture_corpus, load_frozen_suite
from experiments.finer139.runner import RunParams, run_benchmark
from experiments.finer139.types import Sentence, Span

FIXTURE = Path(__file__).parent / "fixtures" / "finer139_mini_corpus.json"


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


def test_load_fixture_corpus() -> None:
    sentences = load_fixture_corpus(FIXTURE)
    assert len(sentences) == 4
    assert all(s.gold_spans for s in sentences)


def test_load_frozen_suite_inline() -> None:
    sentences, spec = load_frozen_suite(FIXTURE)
    assert spec["source"] == "inline"
    assert len(sentences) == 4


def test_paired_bootstrap_delta_significant_when_clear_winner() -> None:
    s = _sentence(["x", "100"], [Span(1, 2, "A", "100")])
    preds_good = [[Span(1, 2)]]
    preds_bad = [[Span(0, 1)]]
    delta = paired_bootstrap_delta([s], preds_bad, preds_good, samples=200, seed=1)
    assert delta["delta_mean"] > 0
    assert delta["significant"] is True


def test_mcnemar_discordant_counts() -> None:
    s = _sentence(["a", "100"], [Span(1, 2, "X", "100")])
    a_preds = [[Span(1, 2)]]
    b_preds = [[Span(0, 1)]]
    m = mcnemar_sentence_hits([s], a_preds, b_preds)
    assert m["a_only"] == 1
    assert m["b_only"] == 0
    assert m["discordant"] == 1


def test_compare_methods_paired_returns_rows() -> None:
    s = _sentence(["100"], [Span(0, 1, "C", "100")])
    preds = {
        "good": [[Span(0, 1)]],
        "bad": [[Span(0, 0)]],
    }
    rows = compare_methods_paired([s], preds, seed=0)
    assert len(rows) == 1
    assert rows[0]["method_a"] == "bad"
    assert rows[0]["method_b"] == "good"


def test_aggregate_multi_seed() -> None:
    runs = [
        {
            "params": {"seed": 1},
            "methods": [
                {"name": "ontology", "strict": {"f1": 0.5}, "partial": {"f1": 0.6}},
                {"name": "nlp", "strict": {"f1": 0.2}},
            ],
        },
        {
            "params": {"seed": 2},
            "methods": [
                {"name": "ontology", "strict": {"f1": 0.7}, "partial": {"f1": 0.8}},
                {"name": "nlp", "strict": {"f1": 0.3}},
            ],
        },
    ]
    agg = aggregate_multi_seed(runs)
    assert agg["n_runs"] == 2
    assert agg["by_method"]["ontology"]["strict_f1_mean"] == 0.6
    assert agg["ranking"][0] == "ontology"


def test_run_benchmark_fixture_suite_protocol_v3() -> None:
    params = RunParams(
        suite_path=str(FIXTURE),
        methods=["ontology"],
        sample_size=4,
        seed=1,
    )
    result = run_benchmark(params)
    assert result["evaluation"]["protocol_version"] == 3
    assert result["dataset"]["frozen"] is True
    assert len(result["methods"]) == 1
    assert result["methods"][0]["error"] is None
    assert result["methods"][0]["strict"]["f1"] >= 0.0


def test_run_benchmark_multi_seed_aggregate() -> None:
    params = RunParams(
        suite_path=str(FIXTURE),
        methods=["ontology"],
        seeds=[11, 22],
        sample_size=4,
    )
    result = run_benchmark(params)
    assert result["evaluation"]["multi_seed"] is True
    assert result["multi_seed_aggregate"]["n_runs"] == 2
    assert len(result["per_seed_runs"]) == 2
