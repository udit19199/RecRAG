from typing import Any

import neo4j
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from neo4j import Driver
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.llm import LLMBase
from neo4j_graphrag.retrievers import (
    Text2CypherRetriever,
    ToolsRetriever,
    VectorCypherRetriever,
)
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage, RawSearchResult, RetrieverResultItem

from .retrievers import VECTOR_RETRIEVAL_QUERY, format_retrieval_record


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
        self, query_text: str, message_history: list[LLMMessage] | None = None, **kwargs
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
        return RawSearchResult(records=records)


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
