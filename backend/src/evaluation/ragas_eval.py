import logging
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """Evaluator using RAGAS framework with OpenAI as judge."""

    def __init__(
        self,
        model: str = "gpt-4o",
        embeddings_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
    ):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided or set in environment")

        self.client = OpenAI(api_key=api_key)
        self.llm = llm_factory(model, client=self.client)
        self.embeddings = OpenAIEmbeddings(model=embeddings_model, client=self.client)

        # Configure metrics with the judge LLM and embeddings
        self.metrics = [
            Faithfulness(llm=self.llm),
            AnswerRelevancy(llm=self.llm, embeddings=self.embeddings),
            ContextPrecision(llm=self.llm),
            ContextRecall(llm=self.llm),
        ]

    def evaluate_query(
        self,
        query: str,
        contexts: List[str],
        response: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate a single query-response pair.

        Args:
            query: The user question.
            contexts: List of retrieved text chunks.
            response: The generated answer.
            ground_truth: The reference answer (required for precision/recall).

        Returns:
            Dictionary of metrics and their scores.
        """
        data = {
            "question": [query],
            "contexts": [contexts],
            "answer": [response],
        }

        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Filter metrics if no ground truth is provided
        active_metrics = self.metrics
        if not ground_truth:
            active_metrics = [m for m in self.metrics if not isinstance(m, (ContextPrecision, ContextRecall))]
            logger.warning("No ground truth provided. Skipping context_precision and context_recall.")

        try:
            result = evaluate(
                dataset,
                metrics=active_metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            return result.to_pandas().iloc[0].to_dict()
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}

    def evaluate_query_batch(
        self,
        dataset: Dataset,
    ) -> Dict[str, Any]:
        """Evaluate a full dataset using RAGAS.

        Args:
            dataset: HF Dataset with question, contexts, answer, ground_truth.

        Returns:
            Dictionary of scores (averages).
        """
        # Determine metrics based on presence of ground_truth
        active_metrics = self.metrics
        if "ground_truth" not in dataset.column_names:
            active_metrics = [m for m in self.metrics if not isinstance(m, (ContextPrecision, ContextRecall))]
            logger.warning("No ground truth provided. Skipping context_precision and context_recall.")

        try:
            result = evaluate(
                dataset,
                metrics=active_metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            return result
        except Exception as e:
            logger.error(f"RAGAS batch evaluation failed: {e}")
            return {}


def get_evaluator() -> RagasEvaluator:
    """Create a RagasEvaluator instance from environment."""
    return RagasEvaluator()
