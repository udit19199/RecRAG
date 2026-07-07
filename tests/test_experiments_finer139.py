"""Unit tests for FiNER-139 experiment scoring and schema (no network)."""

from experiments.finer139.methods import (
    align_surface_string_to_tokens,
    snap_char_range_to_numeric_spans,
)
from experiments.finer139.schema import (
    build_gazetteer,
    build_gazetteer_regex,
    is_numeric,
)
from experiments.finer139.scoring import score
from experiments.finer139.types import Sentence, Span


def _issued_100_million_sentence() -> Sentence:
    return Sentence(
        index=0,
        tokens=["issued", "$", "100", "million"],
        text="issued $ 100 million",
        token_offsets=[(0, 6), (7, 8), (9, 12), (13, 20)],
        gold_spans=[Span(2, 3, "SomeConcept", "100")],
    )


def test_is_numeric() -> None:
    assert is_numeric("100")
    assert is_numeric("$1,234.56")
    assert is_numeric("7.00%")
    assert not is_numeric("Revenue")


def test_gazetteer_regex_matches_finance_keywords() -> None:
    gaz = build_gazetteer(["DebtInstrumentFaceAmount", "RevenueFromContract"])
    rx = build_gazetteer_regex(gaz)
    assert rx.search("The debt instrument face amount was disclosed.")


def test_score_strict_and_relaxed() -> None:
    sentence = _issued_100_million_sentence()
    preds = [[Span(2, 3, None, "100")]]
    result = score([sentence], preds)
    assert result.strict.tp == 1
    assert result.strict.f1 == 1.0
    assert result.relaxed.f1 == 1.0


def test_snap_char_range_to_numeric_spans_finds_minimal_numeric() -> None:
    sentence = _issued_100_million_sentence()
    spans = snap_char_range_to_numeric_spans(9, 20, sentence)
    assert len(spans) == 1
    assert spans[0].start == 2
    assert spans[0].end == 3
    assert spans[0].text == "100"


def test_align_surface_string_snaps_phrase_to_numeric_token_span() -> None:
    sentence = _issued_100_million_sentence()
    used: list[tuple[int, int]] = []
    spans = align_surface_string_to_tokens("100 million", sentence, used)
    assert len(spans) == 1
    assert spans[0].start == 2
    assert spans[0].end == 3


def test_align_surface_string_exact_numeric_span() -> None:
    sentence = Sentence(
        index=2,
        tokens=["total", "was", "$", "100"],
        text="total was $ 100",
        token_offsets=[(0, 5), (6, 9), (10, 11), (12, 15)],
        gold_spans=[Span(3, 4, "Amount", "100")],
    )
    used: list[tuple[int, int]] = []
    spans = align_surface_string_to_tokens("100", sentence, used)
    assert len(spans) == 1
    assert spans[0].start == 3
    assert spans[0].end == 4


def test_align_surface_phrase_scores_strict_when_snapped() -> None:
    sentence = _issued_100_million_sentence()
    used: list[tuple[int, int]] = []
    preds = align_surface_string_to_tokens("100 million", sentence, used)
    result = score([sentence], [preds])
    assert result.strict.tp == 1
    assert result.strict.fp == 0


def test_numeric_filter_drops_non_numeric_predictions() -> None:
    sentence = Sentence(
        index=1,
        tokens=["Apple", "reported", "100"],
        text="Apple reported 100",
        token_offsets=[(0, 5), (6, 14), (15, 18)],
        gold_spans=[Span(2, 3, "X", "100")],
    )
    preds = [[Span(0, 1, None, "Apple"), Span(2, 3, None, "100")]]
    result = score([sentence], preds)
    assert result.strict.tp == 1
    assert result.strict.fp == 0


def test_run_params_default_llm_is_openai() -> None:
    from experiments.finer139.runner import (
        DEFAULT_LLM_MODEL,
        DEFAULT_LLM_PROVIDER,
        RunParams,
    )

    params = RunParams()
    assert params.provider == DEFAULT_LLM_PROVIDER == "openai"
    assert params.model == DEFAULT_LLM_MODEL == "gpt-4o-mini"


def test_build_llm_uses_zero_temperature_for_structured_extraction() -> None:
    from unittest.mock import patch

    from experiments.finer139.runner import _build_llm

    with patch("adapters.create_llm_from_config") as mock_create:
        _build_llm("openai", "gpt-4o-mini")
    mock_create.assert_called_once()
    assert mock_create.call_args.kwargs["temperature"] == 0
