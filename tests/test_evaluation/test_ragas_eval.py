"""Tests for the lightweight RagasEvaluator (no ragas/datasets dependency)."""

from unittest.mock import MagicMock, patch

import pytest

from adapters.base import BaseLLM
from evaluation.ragas_eval import RagasEvaluator, get_evaluator


class FakeLLM(BaseLLM):
    """A fake LLM that returns canned responses for testing."""

    provider = "test"

    def __init__(self, responses: list[str] | None = None, **kwargs):
        super().__init__(model="test-model", **kwargs)
        self.responses = iter(responses or [""])
        self._supports_streaming = False

    @property
    def supports_streaming(self) -> bool:
        return self._supports_streaming

    def generate(self, prompt: str, **kwargs) -> str:
        return next(self.responses, "")

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        return next(self.responses, "")


def make_evaluator(responses: list[str] | None = None) -> RagasEvaluator:
    return RagasEvaluator(llm=FakeLLM(responses=responses))


class TestRagasEvaluator:
    def test_init_requires_llm(self) -> None:
        """Without an llm, construction should raise."""
        with pytest.raises(TypeError):
            RagasEvaluator()  # type: ignore[call-arg]

    def test_init_with_llm(self) -> None:
        """With a BaseLLM, construction succeeds."""
        evaluator = RagasEvaluator(llm=FakeLLM())
        assert evaluator.llm is not None
        assert evaluator.llm.model == "test-model"

    # ── Faithfulness ──────────────────────────────────────────────────────

    def test_faithfulness_returns_score(self) -> None:
        """Faithfulness parses the verification result correctly."""
        evaluator = make_evaluator(
            responses=[
                '["Claim 1", "Claim 2"]',
                '{"Claim 1": true, "Claim 2": false}',
            ]
        )
        score = evaluator._faithfulness(
            contexts=["Some context text."], response="Claim 1 and Claim 2."
        )
        assert score == 0.5  # 1 of 2 claims supported

    def test_faithfulness_empty_claims_fallback(self) -> None:
        """When claim extraction returns empty, score defaults to 0."""
        evaluator = make_evaluator(responses=["[]"])
        score = evaluator._faithfulness(
            contexts=["ctx"], response="Some answer."
        )
        assert score == 0.0

    # ── Answer relevancy ───────────────────────────────────────────────────

    def test_answer_relevancy_returns_score(self) -> None:
        """Answer relevancy parses a numeric score from the LLM."""
        evaluator = make_evaluator(responses=["0.85"])
        score = evaluator._answer_relevancy(
            question="What is X?", response="X is Y."
        )
        assert score == 0.85

    def test_answer_relevancy_bounds_clamping(self) -> None:
        """Scores outside 0-1 are clamped."""
        evaluator = make_evaluator(responses=["1.5"])
        score = evaluator._answer_relevancy("q", "a")
        assert score == 1.0

    # ── Context precision ──────────────────────────────────────────────────

    def test_context_precision_returns_score(self) -> None:
        """Context precision returns a rank-weighted score."""
        # Two contexts, first more relevant
        evaluator = make_evaluator(responses=["0.9", "0.5"])
        score = evaluator._context_precision(
            question="What is X?",
            contexts=["Very relevant chunk.", "Somewhat relevant chunk."],
        )
        # Rank-weighted: (0.9 * 1/1 + 0.5 * 1/2) / (1/1 + 1/2)
        # = (0.9 + 0.25) / 1.5 = 0.7667
        assert score == pytest.approx(0.7667, rel=1e-3)

    def test_context_precision_empty(self) -> None:
        """Empty context list returns 0."""
        evaluator = make_evaluator()
        score = evaluator._context_precision(question="q", contexts=[])
        assert score == 0.0

    # ── Context recall ─────────────────────────────────────────────────────

    def test_context_recall_returns_score(self) -> None:
        """Context recall parses a score from the LLM."""
        evaluator = make_evaluator(responses=["0.75"])
        score = evaluator._context_recall(
            contexts=["Some context."], ground_truth="Expected answer."
        )
        assert score == 0.75

    # ── evaluate_query integration ─────────────────────────────────────────

    @patch("evaluation.ragas_eval.RagasEvaluator._faithfulness")
    @patch("evaluation.ragas_eval.RagasEvaluator._answer_relevancy")
    @patch("evaluation.ragas_eval.RagasEvaluator._context_precision")
    def test_evaluate_query_no_ground_truth(
        self,
        mock_precision: MagicMock,
        mock_relevancy: MagicMock,
        mock_faithfulness: MagicMock,
    ) -> None:
        """Without ground truth, context_recall is skipped."""
        mock_faithfulness.return_value = 0.9
        mock_relevancy.return_value = 0.8
        mock_precision.return_value = 0.7

        evaluator = make_evaluator()
        scores = evaluator.evaluate_query("q", ["c1"], "a")

        assert scores == {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
        }
        assert "context_recall" not in scores

    @patch("evaluation.ragas_eval.RagasEvaluator._faithfulness")
    @patch("evaluation.ragas_eval.RagasEvaluator._answer_relevancy")
    @patch("evaluation.ragas_eval.RagasEvaluator._context_precision")
    @patch("evaluation.ragas_eval.RagasEvaluator._context_recall")
    def test_evaluate_query_with_ground_truth(
        self,
        mock_recall: MagicMock,
        mock_precision: MagicMock,
        mock_relevancy: MagicMock,
        mock_faithfulness: MagicMock,
    ) -> None:
        """With ground truth, all four metrics are computed."""
        mock_faithfulness.return_value = 0.9
        mock_relevancy.return_value = 0.8
        mock_precision.return_value = 0.7
        mock_recall.return_value = 0.6

        evaluator = make_evaluator()
        scores = evaluator.evaluate_query("q", ["c1"], "a", ground_truth="gt")

        assert scores == {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }

    @patch("evaluation.ragas_eval.RagasEvaluator._faithfulness")
    @patch("evaluation.ragas_eval.RagasEvaluator._answer_relevancy")
    @patch("evaluation.ragas_eval.RagasEvaluator._context_precision")
    def test_evaluate_query_graceful_error_handling(
        self,
        mock_precision: MagicMock,
        mock_relevancy: MagicMock,
        mock_faithfulness: MagicMock,
    ) -> None:
        """If a metric fails, it's skipped and others are still returned."""
        mock_faithfulness.side_effect = RuntimeError("API down")
        mock_relevancy.return_value = 0.8
        mock_precision.return_value = 0.7

        evaluator = make_evaluator()
        scores = evaluator.evaluate_query("q", ["c1"], "a")

        assert "faithfulness" not in scores
        assert scores["answer_relevancy"] == 0.8
        assert scores["context_precision"] == 0.7

    # ── get_evaluator factory ──────────────────────────────────────────────

    @patch("evaluation.ragas_eval.load_config")
    @patch("evaluation.ragas_eval.create_llm_from_config")
    def test_get_evaluator_creates_llm_from_config(
        self, mock_create_llm_from_config: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """get_evaluator creates an LLM from config."""
        mock_load_config.return_value = {}
        mock_llm = FakeLLM()
        mock_create_llm_from_config.return_value = mock_llm

        evaluator = get_evaluator()
        assert evaluator.llm is not None
        mock_create_llm_from_config.assert_called_once()

    @patch("evaluation.ragas_eval.load_config")
    @patch("evaluation.ragas_eval.create_llm_from_config")
    def test_get_evaluator_passes_override(
        self, mock_create_llm_from_config: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """get_evaluator passes provider/model overrides to create_llm_from_config."""
        mock_load_config.return_value = {}
        mock_llm = FakeLLM()
        mock_create_llm_from_config.return_value = mock_llm

        evaluator = get_evaluator(provider="ollama", model="llama3.2")
        assert evaluator.llm is not None
        mock_create_llm_from_config.assert_called_once_with(
            {}, "ollama", "llama3.2"
        )
