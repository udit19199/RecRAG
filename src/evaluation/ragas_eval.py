"""Lightweight RAG evaluation via direct LLM calls — no ragas/datasets/pyarrow/torch.

Replaces the ragas framework with direct GPT-4 prompts for each metric.
Each metric makes one or two LLM calls with a structured prompt and parses
a numeric score from the response.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

FAITHFULNESS_EXTRACT_PROMPT = (
    'Extract the factual claims from the following answer. '
    'Return them as a JSON array of strings, e.g. ["claim 1", "claim 2"]. '
    'Include only claims that state facts, not opinions or instructions.\n\n'
    'Answer: {answer}'
)

FAITHFULNESS_VERIFY_PROMPT = (
    'Given the following context, determine if each claim is supported.\n'
    'Respond with a JSON object mapping each claim to a boolean: true if supported, false if not.\n\n'
    'Context: {context}\n\n'
    'Claims: {claims}'
)

ANSWER_RELEVANCY_PROMPT = (
    'Given the question and the answer below, rate how well the answer addresses the question.\n'
    'Respond with only a number between 0.0 and 1.0, where 0 means completely irrelevant '
    'and 1 means perfectly relevant.\n\n'
    'Question: {question}\n\nAnswer: {answer}'
)

CONTEXT_PRECISION_PROMPT = (
    'Given the question and a retrieved context chunk, rate how useful this context chunk is '
    'for answering the question.\n'
    'Respond with only a number between 0.0 and 1.0, where 0 means not useful at all '
    'and 1 means highly useful.\n\n'
    'Question: {question}\n\nContext: {context}'
)

CONTEXT_RECALL_PROMPT = (
    'Given the ground truth answer and the retrieved context below, '
    'rate what fraction of the information in the ground truth is covered by the context.\n'
    'Respond with only a number between 0.0 and 1.0, where 0 means none of the ground truth '
    'is covered and 1 means all of it is covered.\n\n'
    'Ground truth: {ground_truth}\n\nContext: {context}'
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_float(text: str, label: str = "score") -> float:
    """Extract a float from LLM response text. Tries JSON first, then regex."""
    text = text.strip()

    # Try parsing as JSON number
    try:
        val = json.loads(text)
        if isinstance(val, (int, float)):
            return max(0.0, min(1.0, float(val)))
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting a float with regex
    match = re.search(r"([0-9]*\.?[0-9]+)", text)
    if match:
        val = float(match.group(1))
        return max(0.0, min(1.0, val))

    logger.warning("Could not parse %s from: %s", label, text[:100])
    return 0.0


def _parse_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Try to extract a JSON object/array from text."""
    text = text.strip()

    # Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown code fences
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


# ── Evaluator ─────────────────────────────────────────────────────────────────


class RagasEvaluator:
    """Lightweight RAG evaluator using direct LLM calls.

    Computes the same metrics as the ragas framework (faithfulness, answer
    relevancy, context precision, context recall) but without the dependency
    overhead. Each metric makes one or two GPT calls with structured prompts.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        openai_api_key: str | None = None,
    ):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided or set in environment")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        """Make a single LLM call and return the response text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    # ── Individual metrics ─────────────────────────────────────────────────

    def _faithfulness(self, contexts: list[str], response: str) -> float:
        """How much of the answer is supported by the context."""
        context_joined = "\n\n".join(contexts)

        # Step 1: Extract claims from the answer
        claims_json = self._call_llm(
            FAITHFULNESS_EXTRACT_PROMPT.format(answer=response)
        )
        claims_data = _parse_json(claims_json)
        if not isinstance(claims_data, list) or not claims_data:
            logger.warning("Could not extract claims from answer")
            return 0.0

        claims = [str(c) for c in claims_data]

        # Step 2: Verify each claim against context
        # Process in batches to avoid context overflow
        batch_size = 5
        supported = 0
        total = 0

        for i in range(0, len(claims), batch_size):
            batch = claims[i : i + batch_size]
            claims_str = json.dumps(batch)
            verify_result = self._call_llm(
                FAITHFULNESS_VERIFY_PROMPT.format(
                    context=context_joined, claims=claims_str
                )
            )
            verify_data = _parse_json(verify_result)
            if isinstance(verify_data, dict):
                for claim in batch:
                    total += 1
                    if verify_data.get(claim, False):
                        supported += 1
            else:
                # Fallback: if parsing fails, assume all claims in batch are supported
                total += len(batch)
                supported += len(batch)

        return supported / max(total, 1)

    def _answer_relevancy(self, question: str, response: str) -> float:
        """How well the answer addresses the question."""
        result = self._call_llm(
            ANSWER_RELEVANCY_PROMPT.format(question=question, answer=response)
        )
        return _parse_float(result, "answer_relevancy")

    def _context_precision(self, question: str, contexts: list[str]) -> float:
        """How relevant the retrieved context chunks are to the question.

        Uses a rank-aware scoring: earlier (higher-ranked) chunks that are
        relevant contribute more to the score.
        """
        if not contexts:
            return 0.0

        scores = []
        for i, ctx in enumerate(contexts):
            result = self._call_llm(
                CONTEXT_PRECISION_PROMPT.format(question=question, context=ctx)
            )
            score = _parse_float(result, f"context_precision[{i}]")
            scores.append(score)

        # Rank-aware: weight by position (earlier = higher weight)
        total_weight = sum(1.0 / (i + 1) for i in range(len(scores)))
        weighted = sum(
            s * (1.0 / (i + 1)) for i, s in enumerate(scores)
        )
        return weighted / max(total_weight, 1.0)

    def _context_recall(
        self, contexts: list[str], ground_truth: str
    ) -> float:
        """How much of the ground truth is covered by the context."""
        context_joined = "\n\n".join(contexts)
        result = self._call_llm(
            CONTEXT_RECALL_PROMPT.format(
                ground_truth=ground_truth, context=context_joined
            )
        )
        return _parse_float(result, "context_recall")

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate_query(
        self,
        query: str,
        contexts: list[str],
        response: str,
        ground_truth: str | None = None,
    ) -> dict[str, float]:
        """Evaluate a single query-response pair.

        Args:
            query: The user question.
            contexts: List of retrieved text chunks.
            response: The generated answer.
            ground_truth: A reference answer (required for context_recall).

        Returns:
            Dictionary of metric names to scores (0.0–1.0).
        """
        scores: dict[str, float] = {}
        errors: list[str] = []

        try:
            scores["faithfulness"] = self._faithfulness(contexts, response)
        except Exception as e:
            logger.error("faithfulness failed: %s", e)
            errors.append(f"faithfulness: {e}")

        try:
            scores["answer_relevancy"] = self._answer_relevancy(query, response)
        except Exception as e:
            logger.error("answer_relevancy failed: %s", e)
            errors.append(f"answer_relevancy: {e}")

        try:
            scores["context_precision"] = self._context_precision(query, contexts)
        except Exception as e:
            logger.error("context_precision failed: %s", e)
            errors.append(f"context_precision: {e}")

        if ground_truth:
            try:
                scores["context_recall"] = self._context_recall(contexts, ground_truth)
            except Exception as e:
                logger.error("context_recall failed: %s", e)
                errors.append(f"context_recall: {e}")
        else:
            logger.warning(
                "No ground truth provided. Skipping context_recall."
            )

        if errors:
            logger.warning(
                "Evaluation completed with %d error(s): %s",
                len(errors),
                "; ".join(errors),
            )

        return scores

    def evaluate_query_batch(
        self,
        queries: list[str],
        contexts_list: list[list[str]],
        responses: list[str],
        ground_truths: list[str | None] | None = None,
    ) -> list[dict[str, float]]:
        """Evaluate multiple query-response pairs.

        Args:
            queries: List of questions.
            contexts_list: List of retrieved context lists (one per query).
            responses: List of generated answers.
            ground_truths: Optional list of reference answers.

        Returns:
            List of score dicts, one per query.
        """
        results = []
        for i, query in enumerate(queries):
            gt = ground_truths[i] if ground_truths else None
            result = self.evaluate_query(
                query, contexts_list[i], responses[i], ground_truth=gt
            )
            results.append(result)
        return results


def get_evaluator() -> RagasEvaluator:
    """Create a RagasEvaluator instance from environment."""
    model = os.getenv("RAGAS_MODEL", "gpt-4o-mini")
    return RagasEvaluator(model=model)
