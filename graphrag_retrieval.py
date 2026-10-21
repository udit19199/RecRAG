"""GraphRAG configuration, agentic retrieval, and answer generation."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tomllib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast

import neo4j
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.tools import StructuredTool, ToolException
from langchain_openai import ChatOpenAI
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.components.types import Neo4jGraph, PropertyValue
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import LLMGenerationError, Text2CypherRetrievalError
from neo4j_graphrag.generation import GraphRAG as Neo4jGraphRAG
from neo4j_graphrag.llm import LLMBase
from neo4j_graphrag.llm.types import LLMResponse, ToolCall, ToolCallResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.retrievers import (
    Text2CypherRetriever,
    ToolsRetriever,
    VectorCypherRetriever,
)
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage, RawSearchResult, RetrieverResultItem
from pydantic import BaseModel, SecretStr

from cost import (
    MeteredAsyncClient,
    MeteredClient,
    Pricing,
    QuestionCost,
    indexing,
    measure,
)
from dataset import SourcePage
from graphrag_construction import rebuild_graph

RETRIEVAL_TOP_K = 5


class GraphRAGEmbeddings(OpenAIEmbeddings):
    async def async_embed_query(self, text: str) -> list[float]:
        # The upstream async default calls synchronous HTTP on the event loop,
        # serializing the pipeline's otherwise concurrent chunk embedding stage.
        return await asyncio.to_thread(self.embed_query, text)


class SQLiteGenerationCache(BaseCache):
    """Persist successful model calls so failed graph builds can resume."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chat_generations ("
                "key TEXT PRIMARY KEY, result TEXT NOT NULL)"
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return sha256(f"{llm_string}\0{prompt}".encode()).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT result FROM chat_generations WHERE key = ?",
                [self._key(prompt, llm_string)],
            ).fetchone()
        return (
            [
                ChatGeneration.model_validate_json(value)
                for value in json.loads(row["result"])
            ]
            if row
            else None
        )

    def update(
        self, prompt: str, llm_string: str, return_val: Sequence[Generation]
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO chat_generations (key, result) VALUES (?, ?)",
                [
                    self._key(prompt, llm_string),
                    # Pydantic serializes structured parsed responses as JSON
                    # objects, which LangChain's parser accepts on cache hits.
                    json.dumps([value.model_dump_json() for value in return_val]),
                ],
            )

    def clear(self, **kwargs: Any) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM chat_generations")


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


RETRIEVAL_METHOD = "agentic"


def format_retrieval_record(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(
        content=json.dumps(record.data(), ensure_ascii=False, default=str)
    )


def _format_tool_record(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(content=record["content"], metadata=record["metadata"])


def _langchain_tool(tool: Tool) -> StructuredTool:
    def execute(**kwargs: Any) -> tuple[str, list[neo4j.Record]]:
        try:
            result = tool.execute(**kwargs)
        except (
            neo4j.exceptions.CypherTypeError,
            neo4j.exceptions.CypherSyntaxError,
        ) as exc:
            # Query mistakes are feedback for the bounded retrieval agent, not
            # reasons to discard the completed corpus graph or retry HTTP calls.
            raise ToolException(f"Cypher query rejected: {exc.message}") from exc
        except Text2CypherRetrievalError as exc:
            cause = exc.__cause__
            if not isinstance(
                cause,
                (
                    neo4j.exceptions.CypherTypeError,
                    neo4j.exceptions.CypherSyntaxError,
                ),
            ):
                raise
            raise ToolException(f"Cypher query rejected: {cause}") from exc
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
        handle_tool_error=True,
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
        examples=[
            (
                "Name properties can be a scalar or a list of aliases. Match each "
                "alias separately rather than passing a list to string functions. "
                "Example: MATCH (n:__Entity__) WHERE any(alias IN CASE WHEN "
                "valueType(n.name) STARTS WITH 'LIST' THEN n.name ELSE [n.name] END "
                "WHERE toLower(toString(alias)) CONTAINS 'example name') RETURN n LIMIT 10"
            )
        ],
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
    # neo4j-graphrag 1.9.x populates `_embedding_node_property` from SHOW
    # VECTOR INDEXES but its filtered-search path reads `_node_embedding_property`.
    # Keep both names aligned so an agent-supplied filter uses the index metadata.
    vector._node_embedding_property = vector._embedding_node_property
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
    agent_llm: BaseChatModel,
    embedder: Embedder,
):
    neo4j_schema = get_schema(driver, database=database)
    retriever = build_agentic_retriever(
        driver=driver,
        llm=llm,
        embedder=embedder,
        database=database,
        agent_llm=agent_llm,
        neo4j_schema=neo4j_schema,
    )
    return Neo4jGraphRAG(retriever, llm).search(
        question,
        retriever_config={"top_k": RETRIEVAL_TOP_K},
        return_context=True,
        response_fallback=NO_CONTEXT,
    )


DEFAULT_LLM_MODEL = "gpt-6-luna"


DEFAULT_REASONING_EFFORT = "medium"


LLMInput = str | list[LLMMessage]


LLMHistory = list[LLMMessage] | MessageHistory | None


class _GraphProperty(BaseModel):
    key: str
    value: PropertyValue


class _ExtractedNode(BaseModel):
    id: str
    label: str
    properties: list[_GraphProperty]


class _ExtractedRelationship(BaseModel):
    start_node_id: str
    end_node_id: str
    type: str
    properties: list[_GraphProperty]


class _ExtractedGraph(BaseModel):
    nodes: list[_ExtractedNode]
    relationships: list[_ExtractedRelationship]


class GraphRAGChatLLM(LLMBase):
    supports_structured_output = True

    def __init__(self, model: ChatOpenAI) -> None:
        super().__init__(model_name=model.model_name)
        self._model = model

    @staticmethod
    def _messages(
        input: LLMInput,
        history: LLMHistory,
        system_instruction: str | None,
    ) -> list[LLMMessage]:
        if isinstance(input, list):
            messages = list(input)
        else:
            previous = (
                history.messages
                if isinstance(history, MessageHistory)
                else history or []
            )
            messages = [*previous, LLMMessage(role="user", content=input)]
        if system_instruction:
            messages.insert(0, LLMMessage(role="system", content=system_instruction))
        return messages

    @staticmethod
    def _content(response: Any) -> str:
        if isinstance(response, BaseMessage):
            return response.text
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, BaseModel):
            return content.model_dump_json()
        return json.dumps(content)

    @staticmethod
    def _tool_schema(tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.get_name(),
                "description": tool.get_description(),
                "parameters": tool.get_parameters(),
            },
        }

    def _invoke(
        self,
        input: LLMInput,
        history: LLMHistory,
        system_instruction,
        response_format,
        **kwargs,
    ):
        messages = self._messages(input, history, system_instruction)
        if response_format is Neo4jGraph:
            # Strict generation cannot express arbitrary property maps. A list of
            # key/value entries preserves the same facts within a strict schema.
            messages.append(
                LLMMessage(
                    role="user",
                    content=(
                        "Extract the entities and relationships only from the "
                        "Input text section of the preceding message, never from "
                        "formatting examples, using the response schema. "
                        "Every node must have "
                        "a name property. Encode each "
                        "node and relationship's properties as a list of key/value "
                        "entries, using key name for each node's stated name and an "
                        "empty list when there are no properties. Each entry has "
                        "exactly two fields: key is the actual property name (such "
                        "as name or occupation), and value is its stated fact. "
                        "Do not use key or value as property names. Combine multiple "
                        "values of one property into a single array value."
                    ),
                )
            )
            local_chat = bool(os.environ.get("RECRAG_CHAT_BASE_URL"))
            structured = (
                self._model.with_structured_output(
                    _ExtractedGraph, method="function_calling"
                )
                if local_chat
                else self._model.with_structured_output(
                    _ExtractedGraph, method="json_schema", strict=True
                )
            )
            response = structured.invoke(cast(LanguageModelInput, messages), **kwargs)
            graph = json.loads(
                _ExtractedGraph.model_validate(response).model_dump_json(),
                object_hook=lambda value: SimpleNamespace(**value),
            )
            for element in [*graph.nodes, *graph.relationships]:
                properties = SimpleNamespace()
                for prop in element.properties:
                    if prop.key in vars(properties):
                        # Multiple occupations or aliases are facts, not malformed
                        # JSON: retain distinct values in Neo4j's property arrays.
                        previous = getattr(properties, prop.key)
                        values = previous if isinstance(previous, list) else [previous]
                        additions = (
                            prop.value if isinstance(prop.value, list) else [prop.value]
                        )
                        for value in additions:
                            if value not in values:
                                values.append(value)
                        setattr(properties, prop.key, values)
                    else:
                        setattr(properties, prop.key, prop.value)
                for key, value in vars(properties).items():
                    if (
                        isinstance(value, list)
                        and len({type(item) for item in value}) > 1
                    ):
                        # Neo4j property arrays require one type. Keep every fact
                        # when repeated extracted properties mix text and numbers.
                        setattr(properties, key, [str(item) for item in value])
                element.properties = properties
            return Neo4jGraph.model_validate_json(json.dumps(graph, default=vars))
        model = (
            self._model.with_structured_output(
                response_format, method="function_calling"
            )
            if response_format
            else self._model
        )
        response = model.invoke(
            cast(LanguageModelInput, messages),
            **kwargs,
        )
        return response

    def invoke(
        self,
        input: LLMInput,
        message_history: LLMHistory = None,
        system_instruction=None,
        response_format=None,
        **kwargs,
    ):
        try:
            return LLMResponse(
                content=self._content(
                    self._invoke(
                        input,
                        message_history,
                        system_instruction,
                        response_format,
                        **kwargs,
                    )
                )
            )
        except Exception as exc:
            raise LLMGenerationError(exc) from exc

    async def ainvoke(
        self,
        input: LLMInput,
        message_history: LLMHistory = None,
        system_instruction=None,
        response_format=None,
        **kwargs,
    ):
        return await asyncio.to_thread(
            self.invoke,
            input,
            message_history,
            system_instruction,
            response_format,
            **kwargs,
        )

    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: LLMHistory = None,
        system_instruction=None,
    ):
        try:
            response = self._model.bind_tools(
                [self._tool_schema(tool) for tool in tools]
            ).invoke(
                cast(
                    LanguageModelInput,
                    self._messages(input, message_history, system_instruction),
                )
            )
            return ToolCallResponse(
                tool_calls=[
                    ToolCall(name=call["name"], arguments=call["args"])
                    for call in response.tool_calls
                ],
                content=self._content(response) or None,
            )
        except Exception as exc:
            raise LLMGenerationError(exc) from exc

    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: LLMHistory = None,
        system_instruction=None,
    ):
        return await asyncio.to_thread(
            self.invoke_with_tools, input, tools, message_history, system_instruction
        )


class GraphRAG:
    def __init__(
        self,
        *,
        driver: Driver,
        llm: LLMBase,
        construction_llm: GraphRAGChatLLM,
        agent_llm: BaseChatModel,
        embedder: Embedder,
        answer_llm,
        embedding_dimensions: int,
        pricing: Pricing | None = None,
    ) -> None:
        self._driver = driver
        self._llm = llm
        self._construction_llm = construction_llm
        self._agent_llm = agent_llm
        self._embedder = embedder
        self._answer_llm = answer_llm
        self._embedding_dimensions = embedding_dimensions
        self._pricing = pricing or Pricing()
        self.last_question_cost: QuestionCost | None = None

    @classmethod
    def from_config(cls, config_path: Path | None = None) -> GraphRAG:
        project_root = Path(__file__).resolve().parent
        load_dotenv(dotenv_path=project_root / ".env", override=True)
        path = config_path or Path(
            os.environ.get("RECRAG_CONFIG", project_root / "config.toml")
        )
        config = tomllib.loads(path.read_text())
        auth = os.environ["NEO4J_AUTH"]
        if "/" in auth:
            username = auth.split("/", 1)[0]
            password = auth.split("/", 1)[1]
        else:
            username = "neo4j"
            password = auth
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(username, password),
        )

        def chat_model(model_name: str) -> ChatOpenAI:
            local_chat = bool(os.environ.get("RECRAG_CHAT_BASE_URL")) and (
                model_name == config["llm"].get("model", DEFAULT_LLM_MODEL)
            )
            return ChatOpenAI(
                model=model_name,
                timeout=config["llm"]["timeout"],
                base_url=os.environ.get("RECRAG_CHAT_BASE_URL") if local_chat else None,
                api_key=(SecretStr("ollama") if local_chat else None),
                use_responses_api=not local_chat,
                model_kwargs=(
                    {
                        "extra_body": {
                            "options": {
                                "num_ctx": int(
                                    os.environ.get("RECRAG_CONTEXT", "32768")
                                )
                            }
                        }
                    }
                    if local_chat
                    else {}
                ),
                streaming=False,
                disable_streaming=True,
                max_retries=0,
                http_client=MeteredClient(timeout=config["llm"]["timeout"]),
                http_async_client=MeteredAsyncClient(timeout=config["llm"]["timeout"]),
                reasoning=(
                    {
                        "effort": config["llm"].get(
                            "reasoning_effort", DEFAULT_REASONING_EFFORT
                        )
                    }
                    if not local_chat
                    and model_name == config["llm"].get("model", DEFAULT_LLM_MODEL)
                    else None
                ),
            )

        try:
            answer_model = chat_model(config["llm"].get("model", DEFAULT_LLM_MODEL))
            construction_model = config.get("graphrag", {}).get(
                "construction_model", config["llm"].get("model", DEFAULT_LLM_MODEL)
            )
            agent_model = config.get("graphrag", {}).get(
                "agent_model", construction_model
            )
            construction_chat = chat_model(construction_model)
            agent_chat = (
                construction_chat
                if agent_model == construction_model
                else chat_model(agent_model)
            )
            return cls(
                driver=driver,
                llm=GraphRAGChatLLM(answer_model),
                construction_llm=GraphRAGChatLLM(construction_chat),
                agent_llm=agent_chat,
                embedder=GraphRAGEmbeddings(
                    model=config["embedding"]["model"],
                    timeout=config["embedding"]["timeout"],
                    max_retries=0,
                    http_client=MeteredClient(timeout=config["embedding"]["timeout"]),
                ),
                answer_llm=answer_model,
                embedding_dimensions=config["embedding"]["dimensions"],
                pricing=Pricing.model_validate(config["pricing"])
                if "pricing" in config
                else Pricing(),
            )
        except Exception:
            driver.close()
            raise

    def construct(
        self,
        chunks: Sequence[str] | Sequence[SourcePage],
        *,
        database: str,
    ) -> None:
        if chunks and isinstance(chunks[0], SourcePage):
            from benchmark import split_pages

            source_chunks = split_pages(cast(Sequence[SourcePage], chunks))
        else:
            source_chunks = cast(Sequence[str], chunks)
        model = self._construction_llm._model
        original_cache = model.cache
        cache_root = Path(__file__).resolve().parent / "runs"
        namespace = os.environ.get("RECRAG_RUN_NAMESPACE")
        if namespace:
            cache_root /= namespace
        model.cache = SQLiteGenerationCache(
            cache_root / "graphrag" / "construction-cache" / f"{database}.sqlite"
        )
        try:
            rebuild_graph(
                source_chunks,
                driver=self._driver,
                llm=self._construction_llm,
                embedder=self._embedder,
                database=database,
                embedding_dimensions=self._embedding_dimensions,
            )
        finally:
            model.cache = original_cache

    def answer(
        self,
        question: str,
        *,
        database: str,
    ):
        with measure(self._pricing) as question_cost:
            try:
                return answer_question(
                    question,
                    driver=self._driver,
                    database=database,
                    llm=self._llm,
                    embedder=self._embedder,
                    agent_llm=self._agent_llm,
                )
            finally:
                self.last_question_cost = question_cost

    def close(self) -> None:
        self._driver.close()
        if isinstance(self._embedder, OpenAIEmbeddings):
            self._embedder.client.close()
        if isinstance(self._answer_llm, ChatOpenAI):
            self._answer_llm.root_client.close()
        construction_model = self._construction_llm._model
        if (
            isinstance(construction_model, ChatOpenAI)
            and construction_model is not self._answer_llm
        ):
            construction_model.root_client.close()
        if (
            isinstance(self._agent_llm, ChatOpenAI)
            and self._agent_llm is not self._answer_llm
            and self._agent_llm is not construction_model
        ):
            self._agent_llm.root_client.close()


@contextmanager
def setup(settings, index_dir: Path, dataset: str):
    from rag_retrieval import setup as setup_answer

    rag = GraphRAG.from_config()
    try:
        with setup_answer(settings, index_dir, dataset) as answer:
            yield SimpleNamespace(rag=rag, llm=answer.llm, embeddings=answer.embeddings)
    finally:
        rag.close()


def retrieve(
    *,
    rag: GraphRAG,
    chunks: Sequence[str],
    question: str,
    index_dir: Path,
    top_k: int,
):
    from benchmark import RetrievedContext

    database = f"recrag-benchmark-{index_dir.name[:24]}"
    marker = index_dir / "graph.complete"
    started = perf_counter()
    with indexing():
        # A local marker alone does not prove the Desktop database still exists.
        existing = rag._driver.execute_query(
            "SHOW DATABASES", database_="system"
        ).records
        available = any(
            row["name"] == database and row["currentStatus"] == "online"
            for row in existing
        )
        if marker.exists() and available:
            construction_seconds = 0.0
        else:
            rag.construct(chunks, database=database)
            marker.write_text(database)
            construction_seconds = perf_counter() - started
    started = perf_counter()
    retriever = build_agentic_retriever(
        rag._driver,
        rag._llm,
        rag._embedder,
        database,
        rag._agent_llm,
        neo4j_schema=get_schema(rag._driver, database=database),
    )
    result = retriever.search(question, top_k=top_k)
    return RetrievedContext(
        passages=[str(item.content) for item in result.items],
        scores=[
            float(item.metadata.get("score", 0.0)) if item.metadata else 0.0
            for item in result.items
        ],
        construction_seconds=construction_seconds,
        retrieval_seconds=perf_counter() - started,
    )


__all__ = [
    "DEFAULT_LLM_MODEL",
    "DEFAULT_REASONING_EFFORT",
    "GraphRAG",
    "GraphRAGChatLLM",
]
