import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

from evaluation.ragas_eval import RagasEvaluator, Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall


from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from evaluation.ragas_eval import RagasEvaluator, Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding


class TestRagasEvaluator:
    @patch("evaluation.ragas_eval.OpenAI")
    @patch("evaluation.ragas_eval.llm_factory")
    @patch("evaluation.ragas_eval.OpenAIEmbeddings")
    def test_init_with_env_key(self, mock_embeddings_class, mock_llm_factory, mock_openai) -> None:
        mock_llm = MagicMock(spec=InstructorBaseRagasLLM)
        mock_llm_factory.return_value = mock_llm
        mock_embeddings = MagicMock(spec=BaseRagasEmbedding)
        mock_embeddings_class.return_value = mock_embeddings

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            evaluator = RagasEvaluator()
            assert evaluator.llm is not None
            assert evaluator.embeddings is not None

    def test_init_raises_if_no_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                RagasEvaluator()

    @patch("evaluation.ragas_eval.evaluate")
    @patch("evaluation.ragas_eval.OpenAI")
    @patch("evaluation.ragas_eval.llm_factory")
    @patch("evaluation.ragas_eval.OpenAIEmbeddings")
    def test_evaluate_query_no_ground_truth(self, mock_embeddings_class, mock_llm_factory, mock_openai, mock_evaluate) -> None:
        mock_llm = MagicMock(spec=InstructorBaseRagasLLM)
        mock_llm_factory.return_value = mock_llm
        mock_embeddings = MagicMock(spec=BaseRagasEmbedding)
        mock_embeddings_class.return_value = mock_embeddings
        
        # Mocking the result of evaluate
        mock_result = MagicMock()
        mock_result.to_pandas.return_value.iloc.__getitem__.return_value.to_dict.return_value = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8
        }
        mock_evaluate.return_value = mock_result

        evaluator = RagasEvaluator(openai_api_key="test-key")
        scores = evaluator.evaluate_query("q", ["c1"], "a")

        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_precision" not in scores # Should be filtered out

        # Check if active_metrics passed to evaluate did not include precision/recall
        args, kwargs = mock_evaluate.call_args
        metrics = kwargs["metrics"]
        assert all(not isinstance(m, (ContextPrecision, ContextRecall)) for m in metrics)

    @patch("evaluation.ragas_eval.evaluate")
    @patch("evaluation.ragas_eval.OpenAI")
    @patch("evaluation.ragas_eval.llm_factory")
    @patch("evaluation.ragas_eval.OpenAIEmbeddings")
    def test_evaluate_query_with_ground_truth(self, mock_embeddings_class, mock_llm_factory, mock_openai, mock_evaluate) -> None:
        mock_llm = MagicMock(spec=InstructorBaseRagasLLM)
        mock_llm_factory.return_value = mock_llm
        mock_embeddings = MagicMock(spec=BaseRagasEmbedding)
        mock_embeddings_class.return_value = mock_embeddings

        mock_result = MagicMock()
        mock_result.to_pandas.return_value.iloc.__getitem__.return_value.to_dict.return_value = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6
        }
        mock_evaluate.return_value = mock_result

        evaluator = RagasEvaluator(openai_api_key="test-key")
        scores = evaluator.evaluate_query("q", ["c1"], "a", ground_truth="gt")

        assert "context_precision" in scores
        assert "context_recall" in scores
