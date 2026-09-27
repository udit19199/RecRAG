"""Agentic Neo4j retrieval and answer generation for GraphRAG."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

import neo4j
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from neo4j import Driver
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.generation import GraphRAG as Neo4jGraphRAG
from neo4j_graphrag.llm import LLMBase
from neo4j_graphrag.retrievers import (
    Text2CypherRetriever,
    ToolsRetriever,
    VectorCypherRetriever,
)
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage, RawSearchResult, RetrieverResultItem

RETRIEVAL_TOP_K = 5
VECTOR_RETRIEVAL_QUERY = """
CALL {
    WITH node
    MATCH (entity:__Entity__)-[:FROM_CHUNK]->(node)
    RETURN collect(DISTINCT entity.name) AS entities
}
CALL {
    WITH node
    MATCH (first:__Entity__)-[:FROM_CHUNK]->(node)
    MATCH path=(first)-[*1..2]-(last:__Entity__)
    WHERE all(
        relationship IN relationships(path)
        WHERE NOT (type(relationship) IN ["FROM_CHUNK", "NEXT_CHUNK", "FROM_DOCUMENT"])
    )
    WITH path
    LIMIT 25
    RETURN collect(DISTINCT {
        nodes: [item IN nodes(path) | item.name],
        relationships: [
            relationship IN relationships(path) |
            {
                source: startNode(relationship).name,
                type: type(relationship),
                target: endNode(relationship).name
            }
        ]
    }) AS graph_facts
}
RETURN node.text AS text, entities, graph_facts, score
"""
NO_CONTEXT = "I could not find supporting context for this question."
RetrievalMethod = Literal["agentic"]
RETRIEVAL_METHODS: list[RetrievalMethod] = ["agentic"]


def format_retrieval_record(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(
        content=json.dumps(record.data(), ensure_ascii=False, default=str)
    )


def _format_tool_record(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(content=record["content"], metadata=record["metadata"])


def _langchain_tool(tool: Tool) -> StructuredTool:
    def execute(**kwargs: Any) -> tuple[str, list[neo4j.Record]]:
        result = tool.execute(**kwargs)
        records = [
            neo4j.Record(
                {
                    "content": item.content,
                    "tool_name": tool.get_name(),
                    "metadata": {**(item.metadata or {}), "tool": tool.get_name()},
                }
            )
            for item in result.items
        ]
        return "\n".join(str(record["content"]) for record in records), records

    return StructuredTool.from_function(
        func=execute,
        name=tool.get_name(),
        description=tool.get_description(),
        args_schema=tool.get_parameters(),
        response_format="content_and_artifact",
    )


class AgenticToolsRetriever(ToolsRetriever):
    def __init__(self, *, agent_llm: BaseChatModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.result_formatter = _format_tool_record
        self._agent = create_agent(
            model=agent_llm,
            tools=[_langchain_tool(tool) for tool in self._tools],
            system_prompt=self.system_instruction,
            middleware=[ModelCallLimitMiddleware(run_limit=5)],
        )

    def get_search_results(
        self,
        query_text: str,
        message_history: list[LLMMessage] | None = None,
        top_k: int = RETRIEVAL_TOP_K,
        **kwargs,
    ) -> RawSearchResult:
        """Run the LangChain agent and return its Neo4j tool artifacts."""
        messages = [*(message_history or []), {"role": "user", "content": query_text}]
        result = self._agent.invoke({"messages": messages})
        records = [
            record
            for message in result["messages"]
            if isinstance(message, ToolMessage)
            for record in (message.artifact or [])
        ]
        return RawSearchResult(records=records[:top_k])


def build_agentic_retriever(
    driver: Driver,
    llm: LLMBase,
    embedder: Embedder,
    database: str,
    agent_llm: BaseChatModel,
    vector_index: str = "chunk_embeddings",
    neo4j_schema: str | None = None,
) -> ToolsRetriever:
    cypher = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=neo4j_schema,
        neo4j_database=database,
        result_formatter=format_retrieval_record,
    )
    vector = VectorCypherRetriever(
        driver=driver,
        index_name=vector_index,
        retrieval_query=VECTOR_RETRIEVAL_QUERY,
        embedder=embedder,
        result_formatter=format_retrieval_record,
        neo4j_database=database,
    )
    return AgenticToolsRetriever(
        driver=driver,
        llm=llm,
        agent_llm=agent_llm,
        tools=[
            vector.convert_to_tool(
                name="vector_search",
                description="Search the graph by semantic similarity.",
            ),
            cypher.convert_to_tool(
                name="cypher_search",
                description="Generate and run a read-only Cypher query.",
            ),
        ],
        neo4j_database=database,
        system_instruction=(
            "Before any evidence has been returned, call exactly one retrieval tool. "
            "After reviewing evidence, either call the next best tool with a refined "
            "query or make no tool call if the evidence is sufficient. Do not repeat "
            "a search that returned no new evidence."
        ),
    )


def answer_question(
    question: str,
    *,
    driver: Driver,
    database: str,
    llm: LLMBase,
    embedder: Embedder,
    answer_llm: BaseChatModel,
    retrieval_methods: Sequence[RetrievalMethod],
):
    if any(method != "agentic" for method in retrieval_methods):
        raise ValueError("Only agentic retrieval is supported.")
    if not retrieval_methods:
        return []
    neo4j_schema = get_schema(driver, database=database)
    retriever = build_agentic_retriever(
        driver=driver,
        llm=llm,
        embedder=embedder,
        database=database,
        agent_llm=answer_llm,
        neo4j_schema=neo4j_schema,
    )
    return [
        Neo4jGraphRAG(retriever, llm).search(
            question,
            retriever_config={"top_k": RETRIEVAL_TOP_K},
            return_context=True,
            response_fallback=NO_CONTEXT,
        )
        for _method in retrieval_methods
    ]
