from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from langchain_openai import ChatOpenAI
from neo4j import Driver
from neo4j_graphrag.generation.types import RagResultModel

from ..dataset_records.two_wiki_multihopqa import TwoWikiRecord, answer_aliases
from ..graph_rag import DEFAULT_LLM_MODEL, DEFAULT_REASONING_EFFORT


class ResponsesOpenAIModel(DeepEvalBaseLLM):
    """Use LangChain's Responses model for DeepEval judges."""

    def __init__(self, model: str, reasoning_effort: str = DEFAULT_REASONING_EFFORT):
        self._model = ChatOpenAI(
            model=model,
            use_responses_api=True,
            reasoning={"effort": reasoning_effort},
        )
        super().__init__(model)

    def load_model(self) -> ChatOpenAI:
        return self._model

    def generate(self, prompt: str, schema=None):
        response = (
            self._model.with_structured_output(
                schema, method="function_calling"
            ).invoke(prompt)
            if schema
            else self._model.invoke(prompt)
        )
        return (response if schema else response.content), None

    async def a_generate(self, prompt: str, schema=None):
        model = (
            self._model.with_structured_output(schema, method="function_calling")
            if schema
            else self._model
        )
        response = await model.ainvoke(prompt)
        return (response if schema else response.content), None

    def get_model_name(self) -> str:
        return self.name

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True


GraphTriple = tuple[str, str, str]


def read_entity_triples(
    driver: Driver,
    *,
    database: str | None = None,
) -> list[GraphTriple]:
    result = driver.execute_query(
        """
        MATCH (subject:__Entity__)-[relation]->(object:__Entity__)
        RETURN subject.name AS subject,
               type(relation) AS relation,
               object.name AS object
        ORDER BY subject, relation, object
        """,
        database_=database,
    )
    return [
        (str(row["subject"]), str(row["relation"]), str(row["object"]))
        for row in result.records
    ]


def read_graph_statistics(
    driver: Driver,
    *,
    database: str | None = None,
) -> dict[str, int]:
    result = driver.execute_query(
        """
        CALL {
            MATCH (entity:__Entity__)
            RETURN count(entity) AS entity_count,
                   coalesce(
                       sum(CASE WHEN NOT (entity)--() THEN 1 ELSE 0 END), 0
                   ) AS isolated_entity_count
        }
        CALL {
            MATCH (entity:__Entity__)
            WITH entity.name AS name, count(*) AS name_count
            WHERE name_count > 1
            RETURN count(name) AS duplicate_entity_name_groups,
                   coalesce(sum(name_count), 0) AS duplicate_entity_nodes
        }
        CALL {
            MATCH (subject:__Entity__)-[relation]->(object:__Entity__)
            RETURN count(relation) AS relationship_count,
                   count(DISTINCT type(relation)) AS relation_type_count,
                   coalesce(
                       sum(CASE WHEN subject = object THEN 1 ELSE 0 END), 0
                   ) AS self_loop_count
        }
        RETURN entity_count,
               isolated_entity_count,
               duplicate_entity_name_groups,
               duplicate_entity_nodes,
               relationship_count,
               relation_type_count,
               self_loop_count
        """,
        database_=database,
    )
    row = result.records[0]
    return {
        key: int(row[key] or 0)
        for key in (
            "entity_count",
            "isolated_entity_count",
            "duplicate_entity_name_groups",
            "duplicate_entity_nodes",
            "relationship_count",
            "relation_type_count",
            "self_loop_count",
        )
    }


def _construction_metric(
    *,
    name: str,
    description: str,
    evaluation_steps: list[str],
    test_case: LLMTestCase,
    judge: ResponsesOpenAIModel,
) -> dict[str, Any]:
    metric = GEval(
        name=name,
        evaluation_steps=evaluation_steps,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.CONTEXT,
        ],
        model=judge,
        threshold=None,
    )
    metric.measure(test_case)
    return {
        "description": description,
        "score": metric.score,
        "reason": metric.reason,
    }


def evaluate_construction(
    record: TwoWikiRecord,
    graph_triples: Sequence[GraphTriple],
    *,
    judge_model: str = DEFAULT_LLM_MODEL,
    construction_seconds: float | None = None,
    graph_statistics: dict[str, int] | None = None,
) -> dict[str, Any]:
    source_context = [
        f"Page: {page.title}\n{paragraph}"
        for page in record.pages
        for paragraph in page.passages
    ]
    graph_output = json.dumps(graph_triples, ensure_ascii=False)
    full_context_case = LLMTestCase(
        input="Evaluate this graph against the supplied source context.",
        actual_output=graph_output,
        context=source_context,
    )
    supporting_context_case = LLMTestCase(
        input="Evaluate this graph against the supplied supporting passages.",
        actual_output=graph_output,
        context=record.supporting_sentences(),
    )
    judge = ResponsesOpenAIModel(model=judge_model)
    metrics = {
        "groundedness": _construction_metric(
            name="Graph groundedness",
            description="Are the facts in this graph supported by the source text?",
            evaluation_steps=[
                "Evaluate the graph as one representation, not as independently scored triples.",
                "Check whether the information expressed by the graph is supported by the source context.",
                "Consider entity identity, relation meaning, and relation direction.",
                "Accept equivalent wording and aliases when the meaning is the same.",
                "Give a short reason and mention representative unsupported or unclear information when present.",
            ],
            test_case=full_context_case,
            judge=judge,
        ),
        "completeness": _construction_metric(
            name="Graph completeness",
            description="How much factual information from the source text does the graph capture?",
            evaluation_steps=[
                "Evaluate the graph as one representation of all supplied source passages.",
                "Judge how much factual information across the context is represented in the graph.",
                "Treat facts in the context equally and do not use any question or answer to choose what matters.",
                "Accept equivalent wording and aliases when the meaning is the same.",
                "Give a short reason and mention representative missing areas when present.",
            ],
            test_case=full_context_case,
            judge=judge,
        ),
        "supporting_evidence_coverage": _construction_metric(
            name="Supporting evidence coverage",
            description="Does the graph preserve the facts in the dataset's supporting passages?",
            evaluation_steps=[
                "Evaluate the graph as one representation of the supplied supporting passages.",
                "Judge how much factual information in those passages is represented in the graph.",
                "Do not use the question, answer, or any gold graph triples.",
                "Accept equivalent wording and aliases when the meaning is the same.",
                "Give a short reason and mention representative missing areas when present.",
            ],
            test_case=supporting_context_case,
            judge=judge,
        ),
    }
    return {
        "deepeval": metrics,
        "graph_statistics": graph_statistics or {},
        "construction_seconds": construction_seconds,
    }


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def _metric_result(metric, test_case: LLMTestCase) -> dict[str, Any]:
    metric.measure(test_case)
    return {"score": metric.score, "reason": metric.reason}


def evaluate_answer(
    record: TwoWikiRecord,
    result: RagResultModel,
    *,
    judge_model: str = DEFAULT_LLM_MODEL,
) -> dict[str, Any]:
    """Score one 2Wiki answer after retrieval."""
    retriever_result = result.retriever_result
    retrieved_context = (
        [str(item.content) for item in retriever_result.items]
        if retriever_result is not None
        else []
    )
    gold_context = record.supporting_sentences()
    test_case = LLMTestCase(
        input=record.question,
        actual_output=result.answer,
        expected_output=record.answer,
        context=gold_context,
        retrieval_context=retrieved_context,
    )
    answer_alias_match = _normalise(result.answer) in {
        _normalise(alias) for alias in answer_aliases(record)
    }
    answer = {
        "expected": record.answer,
        "expected_id": record.answer_id,
        "actual": result.answer,
        "alias_match": answer_alias_match,
    }
    judge = ResponsesOpenAIModel(model=judge_model)
    if retrieved_context:
        answer["faithfulness"] = _metric_result(
            FaithfulnessMetric(model=judge, threshold=None), test_case
        )
    else:
        answer["faithfulness"] = {"reason": "No retrieved context."}
    answer["deepeval"] = {
        "relevancy": _metric_result(
            AnswerRelevancyMetric(model=judge, threshold=None), test_case
        ),
        "correctness": _metric_result(
            GEval(
                name="Answer correctness",
                evaluation_steps=[
                    "Check whether the answer answers the question.",
                    "Accept the reference answer and its listed aliases.",
                    "Reject contradictory or incomplete answers.",
                ],
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                model=judge,
                threshold=None,
            ),
            LLMTestCase(
                input=record.question,
                actual_output=result.answer,
                expected_output=json.dumps(
                    {"answer": record.answer, "aliases": answer_aliases(record)},
                    ensure_ascii=False,
                ),
            ),
        ),
    }
    return {"record_id": record.id, "answer": answer}
