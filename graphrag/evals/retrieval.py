from __future__ import annotations

from typing import Any

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from neo4j_graphrag.generation.types import RagResultModel

from ..dataset_records.two_wiki_multihopqa import TwoWikiRecord
from ..graph_rag import DEFAULT_LLM_MODEL
from .construction import ResponsesOpenAIModel, _metric_result


def evaluate_retrieval(
    record: TwoWikiRecord,
    result: RagResultModel,
    *,
    judge_model: str = DEFAULT_LLM_MODEL,
    top_k: int = 5,
) -> dict[str, Any]:
    items = result.retriever_result.items if result.retriever_result else []
    retrieved_context = [str(item.content) for item in items[:top_k]]
    test_case = LLMTestCase(
        input=record.question,
        expected_output="\n".join(record.supporting_sentences()),
        retrieval_context=retrieved_context,
    )
    judge = ResponsesOpenAIModel(model=judge_model)
    return {
        "record_id": record.id,
        "top_k": top_k,
        "retrieved_context": retrieved_context,
        "deepeval": {
            "contextual_precision": _metric_result(
                ContextualPrecisionMetric(model=judge, threshold=None), test_case
            ),
            "contextual_recall": _metric_result(
                ContextualRecallMetric(model=judge, threshold=None), test_case
            ),
            "contextual_relevancy": _metric_result(
                ContextualRelevancyMetric(model=judge, threshold=None), test_case
            ),
        },
    }
