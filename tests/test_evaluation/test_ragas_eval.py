"""Tests for the lightweight RagasEvaluator (no ragas/datasets dependency)."""

from unittest.mock import MagicMock, patch

import pytest

from evaluation.ragas_eval import RagasEvaluator


class TestRagasEvaluator:
    def test_init_requires_api_key(self) -> None:
        """Without an API key, construction should raise."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                RagasEvaluator()

    def test_init_with_env_key(self) -> None:
        """With OPENAI_API_KEY set, construction succeeds and client is ready."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            evaluator = RagasEvaluator()
            assert evaluator.client is not None
            assert evaluator.model == "gpt-4o-mini"

    def test_init_with_explicit_key(self) -> None:
        """An explicitly passed key takes precedence."""
        evaluator = RagasEvaluator(openai_api_key="explicit-key")
        assert evaluator.client.api_key == "explicit-key"

    def test_init_custom_model(self) -> None:
        """A custom model is accepted and stored."""
        evaluator = RagasEvaluator(model="gpt-4", openai_api_key="key")
        assert evaluator.model == "gpt-4"

    # ── Faithfulness ──────────────────────────────────────────────────────

    @patch("evaluation.ragas_eval.RagasEvaluator._call_llm")
    def test_faithfulness_returns_score(self, mock_call: MagicMock) -> None:
        """Faithfulness parses the verification result correctly."""
        # First call: extract claims
        # Second call: verify claims against context
        mock_call.side_effect = [
            '["Claim 1", "Claim 2"]',
            '{"Claim 1": true, "Claim 2": false}',
        ]
        evaluator = RagasEvaluator(openai_api_key="key")
        score = evaluator._faithfulness(
            contexts=["Some context text."], response="Claim 1 and Claim 2."
        )
        assert score == 0.5  # 1 of 2 claims supported

    @patch("evaluation.ragas_eval.RagasEvaluator._call_llm")
    def test_faithfulness_empty_claims_fallback(self, mock_call: MagicMock) -> None:
        """When claim extraction returns empty, score defaults to 0."""
        mock_call.return_value = "[]"
        evaluator = RagasEvaluator(openai_api_key="key")
        score = evaluator._faithfulness(
            contexts=["ctx"], response="Some answer."
        )
        assert score == 0.0

    # ── Answer relevancy ───────────────────────────────────────────────────

    @patch("evaluation.ragas_eval.RagasEvaluator._call_llm")
    def test_answer_relevancy_returns_score(self, mock_call: MagicMock) -> None:
        """Answer relevancy parses a numeric score from the LLM."""
        mock_call.return_value = "0.85"
        evaluator = RagasEvaluator(openai_api_key="key")
        score = evaluator._answer_relevancy(
            question="What is X?", response="X is Y."
        )
        assert score == 0.85

    @patch("evaluation.ragas_eval.RagasEvaluator._call_llm")
    def test_answer_relevancy_bounds_clamping(self, mock_call: MagicMock) -> None:
        """Scores outside 0-1 are clamped."""
        mock_call.return_value = "1.5"
        evaluator = RagasEvaluator(openai_api_key="key")
        score = evaluator._answer_relevancy("q", "a")
        assert score == 1.0

    # ── Context precision ──────────────────────────────────────────────────

    @patch("evaluation.ragas_eval.RagasEvaluator._call_llm")
    def test_context_precision_returns_score(self, mock_call: MagicMock) -> None:
        """Context precision returns a rank-weighted score."""
        # Two contexts, first more relevant
        mock_call.side_effect = ["0.9", "0.5"]
        evaluator = RagasEvaluator(openai_api_key="key")
        score = evaluator._context_precision(
            question="What is X?",
            contexts=["Very relevant chunk.", "Somewhat relevant chunk."],
        )
        # Rank-weighted: (0.9 * 1/1 + 0.5 * 1/2) / (1/1 + 1/2)
        # = (0.9 + 0.25) / 1.5 = 0.7667
        assert score == pytest.approx(0.7667, rel=1e-3)

    @patch("evaluation.ragas_eval.RagasEvaluator._call_llm")
    def test_context_precision_empty(self, mock_call: MagicMock) -> None:
        """Empty context list returns 0."""
        evaluator = RagasEvaluator(openai_api_key="key")
        score = evaluator._context_precision(question="q", contexts=[])
        assert score == 0.0

    # ── Context recall ─────────────────────────────────────────────────────

    @patch("evaluation.ragas_eval.RagasEvaluator._call_llm")
    def test_context_recall_returns_score(self, mock_call: MagicMock) -> None:
        """Context recall parses a score from the LLM."""
        mock_call.return_value = "0.75"
        evaluator = RagasEvaluator(openai_api_key="key")
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

        evaluator = RagasEvaluator(openai_api_key="key")
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

        evaluator = RagasEvaluator(openai_api_key="key")
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

        evaluator = RagasEvaluator(openai_api_key="key")
        scores = evaluator.evaluate_query("q", ["c1"], "a")

        assert "faithfulness" not in scores
        assert scores["answer_relevancy"] == 0.8
        assert scores["context_precision"] == 0.7

    # ── get_evaluator factory ──────────────────────────────────────────────

    def test_get_evaluator_uses_env(self) -> None:
        """get_evaluator reads model from env and creates evaluator."""
        from evaluation.ragas_eval import get_evaluator

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "key", "RAGAS_MODEL": "gpt-4"},
        ):
            evaluator = get_evaluator()
            assert evaluator.model == "gpt-4"

    def test_get_evaluator_defaults(self) -> None:
        """get_evaluator falls back to default model when env is unset."""
        from evaluation.ragas_eval import get_evaluator

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "key"},
            clear=True,
        ):
            evaluator = get_evaluator()
            assert evaluator.model == "gpt-4o-mini"
