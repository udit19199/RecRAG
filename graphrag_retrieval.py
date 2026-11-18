"""Agentic Hybrid RAG configuration, agentic retrieval, and answer generation."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
import tomllib
from array import array
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, LiteralString, cast

import neo4j
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.tools import StructuredTool, ToolException
from langchain_openai import ChatOpenAI
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.components.schema import GraphSchema
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.exceptions import EmbeddingsGenerationError, LLMGenerationError
from neo4j_graphrag.generation import GraphRAG as Neo4jGraphRAG
from neo4j_graphrag.generation.prompts import Text2CypherTemplate
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.llm import LLMBase
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.retrievers import (
    Text2CypherRetriever,
    ToolsRetriever,
    VectorCypherRetriever,
)
from neo4j_graphrag.retrievers.text2cypher import extract_cypher
from neo4j_graphrag.schema import get_structured_schema
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import (
    LLMMessage,
    RawSearchResult,
    RetrieverResult,
    RetrieverResultItem,
)
from neo4j_graphrag.utils.rate_limit import NoOpRateLimitHandler
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, SecretStr

from cost import (
    MeteredAsyncClient,
    MeteredClient,
    Pricing,
    QuestionCost,
    indexing,
    measure,
)
from dataset import SourcePage
from graphrag_construction import (
    EXTRACTION_INSTRUCTION,
    GENERAL_ONTOLOGY_SCHEMA,
    ExtractedGraph,
    OntologyChoice,
    PageChunks,
    construction_key,
    graph_from_extraction,
    is_junk_property,
    ontology_for,
    rebuild_graph,
)

if TYPE_CHECKING:
    from benchmark import RetrievedContext, Settings
    from rag_retrieval import LlamaEmbedding, ResponsesLLM

RETRIEVAL_TOP_K = 10
CYPHER_ROW_LIMIT = 10
ENTITY_LIMIT = 15
FACT_LIMIT = 5
FACTS_PER_ENTITY = 10
EMBEDDING_BATCH_SIZE = 512
MIN_PROPERTY_USES = 3
MIN_PROPERTY_SHARE = 0.01
NO_SCORE = -1.0

DEFAULT_LLM_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "medium"


class LLMConfig(BaseModel):
    model: str = DEFAULT_LLM_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    timeout: PositiveFloat


class EmbeddingConfig(BaseModel):
    model: str
    dimensions: PositiveInt
    timeout: PositiveFloat


class GraphRetrievalConfig(BaseModel):
    timeout: PositiveFloat = 10


class GraphConfig(BaseModel):
    timeout: PositiveFloat = 30
    index_timeout: PositiveFloat = 600
    extraction_concurrency: PositiveInt = 20
    ontology: OntologyChoice = "general"
    construction_model: str | None = None
    agent_model: str | None = None
    # Thinking level for the one call that writes a Cypher query from the
    # agent's lookup. None uses llm.reasoning_effort.
    cypher_reasoning_effort: str | None = None
    retrieval: GraphRetrievalConfig = Field(default_factory=GraphRetrievalConfig)


class GraphRAGConfig(BaseModel):
    llm: LLMConfig
    embedding: EmbeddingConfig
    graphrag: GraphConfig = Field(default_factory=GraphConfig)
    pricing: Pricing = Field(default_factory=Pricing)

    @classmethod
    def load(cls, path: Path | None = None) -> GraphRAGConfig:
        path = path or Path(
            os.environ.get(
                "RECRAG_CONFIG", Path(__file__).resolve().parent / "config.toml"
            )
        )
        return cls.model_validate(tomllib.loads(path.read_text()))

    @property
    def construction_model(self) -> str:
        return self.graphrag.construction_model or self.llm.model

    @property
    def agent_model(self) -> str:
        return self.graphrag.agent_model or self.construction_model

    def construction_key(self, ontology: GraphSchema) -> str:
        from benchmark import CHUNK_OVERLAP, CHUNK_SIZE

        return construction_key(
            model=self.construction_model,
            reasoning_effort=self.llm.reasoning_effort,
            endpoint=os.environ.get("RECRAG_CHAT_BASE_URL", ""),
            embedding_model=self.embedding.model,
            embedding_dimensions=self.embedding.dimensions,
            chunking=(
                f"llama-index SentenceSplitter {CHUNK_SIZE} tokens, {CHUNK_OVERLAP} "
                "overlap, per page, title-prefixed"
            ),
            ontology=ontology,
        )


def graph_index_key(corpus_key: str, dataset: str) -> str:
    config = GraphRAGConfig.load()
    ontology = ontology_for(dataset, config.graphrag.ontology)
    return sha256(
        f"{corpus_key}\0graphrag\0{config.construction_key(ontology)}".encode()
    ).hexdigest()


def graph_contents(ontology: GraphSchema) -> str:
    return ", ".join(
        node.label for node in ontology.node_types if node.label != "Thing"
    )


@contextmanager
def _sqlite(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _open_table(path: Path, ddl: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _sqlite(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(ddl)


class SQLiteEmbeddingCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        _open_table(
            path,
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "key TEXT PRIMARY KEY, vector BLOB NOT NULL)",
        )

    def lookup(self, keys: Sequence[str]) -> list[list[float] | None]:
        found: list[list[float] | None] = []
        with _sqlite(self._path) as connection:
            for key in keys:
                row = connection.execute(
                    "SELECT vector FROM embeddings WHERE key = ?", [key]
                ).fetchone()
                found.append(array("d", row["vector"]).tolist() if row else None)
        return found

    def update(self, keys: Sequence[str], vectors: Sequence[list[float]]) -> None:
        with _sqlite(self._path) as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                [
                    [key, array("d", vector).tobytes()]
                    for key, vector in zip(keys, vectors, strict=True)
                ],
            )


class GraphRAGEmbeddings(OpenAIEmbeddings):
    def __init__(self, *, model: str, dimensions: int, timeout: float) -> None:
        super().__init__(
            model=model,
            rate_limit_handler=NoOpRateLimitHandler(),
            timeout=timeout,
            max_retries=0,
            http_client=MeteredClient(timeout=timeout),
        )
        self.dimensions = dimensions
        self.store: SQLiteEmbeddingCache | None = None

    def _key(self, text: str) -> str:
        return sha256(f"{self.model}\0{self.dimensions}\0{text}".encode()).hexdigest()

    def _create(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.embeddings.create(
                input=texts, model=self.model, dimensions=self.dimensions
            )
        except Exception as exc:
            raise EmbeddingsGenerationError(
                f"Failed to generate embeddings with OpenAI: {exc}"
            ) from exc
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    def embed_all(self, texts: Sequence[str]) -> None:
        if self.store is None:
            raise RuntimeError("Batch embedding needs a construction store.")
        unique: list[str] = []
        seen: set[str] = set()
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique.append(text)
        keys = [self._key(text) for text in unique]
        missing = [
            text
            for text, vector in zip(unique, self.store.lookup(keys), strict=True)
            if vector is None
        ]
        for start in range(0, len(missing), EMBEDDING_BATCH_SIZE):
            batch = missing[start : start + EMBEDDING_BATCH_SIZE]
            self.store.update([self._key(text) for text in batch], self._create(batch))

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        key = self._key(text)
        if self.store is None:
            return self._create([text])[0]
        cached = self.store.lookup([key])[0]
        if cached is not None:
            return cached
        vector = self._create([text])[0]
        self.store.update([key], [vector])
        return vector

    async def async_embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


class SQLiteGenerationCache(BaseCache):
    def __init__(self, path: Path, identity: str) -> None:
        self._path = path
        self._identity = identity
        self._local = threading.local()
        _open_table(
            path,
            "CREATE TABLE IF NOT EXISTS chat_generations ("
            "key TEXT PRIMARY KEY, result TEXT NOT NULL)",
        )

    def _key(self, prompt: str) -> str:
        return sha256(f"{self._identity}\0{prompt}".encode()).hexdigest()

    def forget_hit(self) -> None:
        self._local.hit = None

    def evict_hit(self) -> bool:
        key: str | None = getattr(self._local, "hit", None)
        self._local.hit = None
        if key is None:
            return False
        with _sqlite(self._path) as connection:
            connection.execute("DELETE FROM chat_generations WHERE key = ?", [key])
        return True

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        key = self._key(prompt)
        with _sqlite(self._path) as connection:
            row = connection.execute(
                "SELECT result FROM chat_generations WHERE key = ?", [key]
            ).fetchone()
        self._local.hit = key if row else None
        if row is None:
            return None
        return [
            ChatGeneration.model_validate_json(value)
            for value in json.loads(row["result"])
        ]

    def update(
        self, prompt: str, llm_string: str, return_val: Sequence[Generation]
    ) -> None:
        with _sqlite(self._path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO chat_generations (key, result) VALUES (?, ?)",
                [
                    self._key(prompt),
                    json.dumps([value.model_dump_json() for value in return_val]),
                ],
            )

    def clear(self, **kwargs: Any) -> None:
        with _sqlite(self._path) as connection:
            connection.execute("DELETE FROM chat_generations")


VECTOR_RETRIEVAL_QUERY = f"""
OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(page:Document)
CALL (node) {{
    MATCH (entity:__Entity__)-[:FROM_CHUNK]->(node)
    WITH DISTINCT entity.name AS name
    LIMIT {ENTITY_LIMIT}
    RETURN collect(name) AS entities
}}
CALL (node) {{
    MATCH (entity:__Entity__)-[:FROM_CHUNK]->(node)
    CALL (entity) {{
        MATCH (entity)-[relation]-(:__Entity__)
        RETURN relation
        LIMIT {FACTS_PER_ENTITY}
    }}
    WITH DISTINCT relation, startNode(relation) AS source, endNode(relation) AS target
    WITH relation, source, target,
         EXISTS {{ (source)-[:FROM_CHUNK]->(node) }}
         AND EXISTS {{ (target)-[:FROM_CHUNK]->(node) }} AS local
    ORDER BY local
    LIMIT {FACT_LIMIT}
    RETURN collect(source.name + ' -' + type(relation) + '-> ' + target.name) AS facts
}}
RETURN node.text AS text, page.title AS title, entities, facts, score
"""


NO_CONTEXT = "I could not find supporting context for this question."


RETRIEVAL_METHOD = "agentic"


class VectorEvidence(BaseModel):
    title: str | None
    text: str
    entities: list[str]
    facts: list[str]


def format_vector_record(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(
        content=VectorEvidence(
            title=record["title"],
            text=record["text"],
            entities=[str(name) for name in record["entities"]],
            facts=[str(fact) for fact in record["facts"]],
        ).model_dump_json(),
        metadata={"score": record["score"]},
    )


# Graph rows carry graph facts only (13 Nov 2026). Before this, the generated
# Cypher often collected every chunk of an article (whole article text) and
# every linked entity into one row; 13 of the 54 graph rows the answer model
# got were over 1,000 tokens, up to 14,600. Passage text now comes only from
# vector_search, and long lists are cut.
CYPHER_LIST_LIMIT = 20
PASSAGE_TEXT = re.compile(r"\.\s*text\b|\[\s*['\"]text['\"]\s*\]", re.IGNORECASE)


def _graph_facts(value: Any) -> Any:
    """Drop passage text and keep at most CYPHER_LIST_LIMIT items per list."""
    if isinstance(value, dict):
        return {
            key: _graph_facts(item)
            for key, item in value.items()
            if key not in ("text", "embedding")
        }
    if isinstance(value, list):
        kept = [_graph_facts(item) for item in value[:CYPHER_LIST_LIMIT]]
        if len(value) > CYPHER_LIST_LIMIT:
            kept.append(f"(+{len(value) - CYPHER_LIST_LIMIT} more)")
        return kept
    return value


def format_cypher_record(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(
        content=json.dumps(_graph_facts(record.data()), ensure_ascii=False, default=str)
    )


def without_passage_text(schema: str) -> str:
    """The graph schema with Chunk.text removed, so queries cannot ask for it."""
    return re.sub(r",?\s*\btext: STRING", "", schema).replace("{, ", "{")


def _format_tool_record(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(content=record["content"], metadata=record["metadata"])


def parse_vector_evidence(content: str) -> VectorEvidence | None:
    """Return vector_search evidence, or None for a cypher_search row."""
    try:
        return VectorEvidence.model_validate_json(content)
    except ValueError:
        return None


def answer_text(content: str) -> str:
    """What the answer model reads for one item.

    A vector_search passage becomes its plain text, which starts with the
    page title line exactly as Normal RAG's chunks do; its entity list and
    graph facts stay in the saved evidence but are not sent. A cypher_search
    row has no plain form and is sent as returned.
    """
    evidence = parse_vector_evidence(content)
    return evidence.text if evidence else content


def _evidence_key(content: str) -> str:
    evidence = parse_vector_evidence(content)
    text = evidence.text if evidence else content
    return " ".join(text.lower().split())


EVIDENCE_ID = re.compile(r"\bE[0-9a-f]{6}\b")


def evidence_id(content: str) -> str:
    """Short ID the agent uses to pick an item; the same text gets the same ID."""
    return "E" + sha256(_evidence_key(content).encode()).hexdigest()[:6]


TOOL_PURPOSE = (
    "Use vector_search to find passages about a topic. Use cypher_search to "
    "look up specific named entities and how they are connected. "
)
# Passages first, then one graph lookup: on MultiHop-RAG temporal questions this
# order beat free tool choice (58% vs 50% exact match over 100 questions) because
# the agent otherwise almost never calls cypher_search.
AGENT_INSTRUCTION = TOOL_PURPOSE + (
    "First call vector_search with the question. Then call cypher_search once "
    "about the named entities in that evidence. After that, either call the next "
    "best tool with a refined query or make no tool call if the evidence is "
    "sufficient. Do not repeat a search that returned no new evidence."
    # Picking the evidence (13 Nov 2026). Before this, code kept the 10
    # vector passages with the highest similarity score and graph rows only
    # filled empty slots, so no graph row ever reached the answer model (0 of
    # 300 comparison runs) and the agent's own view of what was relevant was
    # discarded. In 59 of 111 runs missing a named article, the graph lookup
    # had returned that article.
    " Every result line starts with an ID such as [E1a2b3c]. When you stop "
    "searching, do not answer the question. Reply only with the IDs of the "
    f"results the answer needs, at most {RETRIEVAL_TOP_K}, most important first, "
    "one per line. Results from either tool count. Pick evidence for every "
    "article, person or part the question asks about."
)

# Graph ablation (18 Nov 2026, runs/nograph): RECRAG_AGENT_NO_GRAPH=1 runs the
# same agent with no graph at all. The cypher_search tool is removed, and
# vector_search results carry no entity names or graph facts. The instruction
# drops only the cypher_search sentences; everything else is word for word the
# same. Unset, the agent is the frozen final version.
NO_GRAPH_AGENT_INSTRUCTION = "Use vector_search to find passages about a topic. " + (
    "First call vector_search with the question. After that, either call "
    "vector_search again with a refined query or make no tool call if the "
    "evidence is sufficient. Do not repeat a search that returned no new evidence."
    " Every result line starts with an ID such as [E1a2b3c]. When you stop "
    "searching, do not answer the question. Reply only with the IDs of the "
    f"results the answer needs, at most {RETRIEVAL_TOP_K}, most important first, "
    "one per line. Pick evidence for every "
    "article, person or part the question asks about."
)
NO_GRAPH_VECTOR_QUERY = """
OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(page:Document)
RETURN node.text AS text, page.title AS title, [] AS entities, [] AS facts, score
"""


def no_graph_agent() -> bool:
    return os.environ.get("RECRAG_AGENT_NO_GRAPH") == "1"


CYPHER_PROMPT = f"""
Task: Generate one read-only Cypher statement for a Neo4j 5 graph database from a user input.

Schema:
{{schema}}

Examples:
{{examples}}

Rules:
- Use only node labels, relationship types, and properties listed in the schema.
- Every extracted entity also has the label __Entity__ with a name and optional aliases.
- Match names case-insensitively: compare toLower(n.name), and each alias in coalesce(n.aliases, []), with a lowercase string.
- Return graph facts only: names, titles, dates, short properties and relationship types. Never return passage text; vector_search finds passages.
- Write alternative relationship types as [:A|B].
- Return names and short properties, never embedding, and end with LIMIT {CYPHER_ROW_LIMIT} or less.

Input:
{{query_text}}

Return only the Cypher statement, with no triple backticks or other text.

Cypher query:
"""


CYPHER_INPUT_MARK = "Input:\n"
assert CYPHER_PROMPT.count(CYPHER_INPUT_MARK) == 1


def _name_matches(variable: str, name: str) -> str:
    return (
        f"(toLower({variable}.name) CONTAINS '{name}' OR any(alias IN "
        f"coalesce({variable}.aliases, []) WHERE toLower(alias) CONTAINS '{name}'))"
    )


CYPHER_EXAMPLES = [
    "USER INPUT: 'What is connected to Ada Lovelace?' QUERY: "
    "MATCH (n:__Entity__)-[r]-(m:__Entity__) WHERE "
    + _name_matches("n", "ada lovelace")
    + " RETURN n.name AS entity, type(r) AS relationship, m.name AS related LIMIT 10",
    "USER INPUT: 'Which pages mention Acme Corp?' QUERY: "
    "MATCH (n:__Entity__)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(d:Document) "
    "WHERE "
    + _name_matches("n", "acme corp")
    + " RETURN DISTINCT n.name AS entity, d.title AS page LIMIT 10",
    "USER INPUT: 'Which organizations is Jane Doe part of?' QUERY: "
    "MATCH (p:Person)-[:MEMBER_OF|WORKS_FOR]->(o:Organization) WHERE "
    + _name_matches("p", "jane doe")
    + " RETURN p.name AS person, o.name AS organization LIMIT 10",
]


_TOOL_ERROR_CODES = [
    "Neo.ClientError.Statement.",
    "Neo.ClientError.Transaction.TransactionTimedOut",
    "Neo.TransientError.General.MemoryPool",
]


def _query_error(exc: neo4j.exceptions.Neo4jError) -> bool:
    code = exc.code or ""
    return any(code.startswith(prefix) for prefix in _TOOL_ERROR_CODES)


class _ChunkVectorRetriever(VectorCypherRetriever):
    def __init__(
        self,
        *,
        driver: Driver,
        database: str,
        embedder: GraphRAGEmbeddings,
        retrieval_query: str = VECTOR_RETRIEVAL_QUERY,
    ) -> None:
        super().__init__(
            driver=driver,
            index_name="chunk_embeddings",
            retrieval_query=retrieval_query,
            embedder=embedder,
            result_formatter=format_vector_record,
            neo4j_database=database,
        )

    def get_search_results(self, query_text: str) -> RawSearchResult:
        """Find source passages similar in meaning to the query text."""
        return super().get_search_results(query_text=query_text, top_k=RETRIEVAL_TOP_K)


class _BoundedText2CypherRetriever(Text2CypherRetriever):
    def __init__(
        self,
        *,
        driver: Driver,
        llm: LLMBase,
        database: str,
        neo4j_schema: str,
        timeout: float,
    ) -> None:
        super().__init__(
            driver=driver,
            llm=llm,
            neo4j_schema=neo4j_schema,
            examples=CYPHER_EXAMPLES,
            neo4j_database=database,
            result_formatter=format_cypher_record,
        )
        self._timeout = timeout

    def get_search_results(self, query_text: str) -> RawSearchResult:
        """Generate a read-only Cypher query for the input and run it."""
        prompt = Text2CypherTemplate(template=CYPHER_PROMPT).format(
            schema=self.neo4j_schema,
            examples="\n".join(self.examples or []),
            query_text=query_text,
        )
        # The same prompt text in two messages: the part every question shares
        # (task, schema, examples, rules) as the system message, the lookup as
        # the user message. OpenAI only served the shared part from its prompt
        # cache when it was a message of its own (tested 10 Nov 2026: 0 of
        # 2,336 input tokens cached as one message, 2,304 as two). The shared
        # key sends these calls to the same cache.
        shared, lookup = prompt.split(CYPHER_INPUT_MARK, 1)
        cypher = extract_cypher(
            self.llm.invoke(
                [
                    {"role": "system", "content": shared},
                    {"role": "user", "content": CYPHER_INPUT_MARK + lookup},
                ],
                prompt_cache_key=f"recrag-cypher-{self.neo4j_database}",
            ).content
        )
        if PASSAGE_TEXT.search(cypher):
            raise ToolException(
                "Refused a query that returns passage text; cypher_search returns "
                f"graph facts only, vector_search finds passages.\nCypher: {cypher}"
            )
        try:
            with self.driver.session(
                database=self.neo4j_database, default_access_mode=neo4j.READ_ACCESS
            ) as session:
                transaction = session.begin_transaction(timeout=self._timeout)
                try:
                    plan = transaction.run(
                        cast(LiteralString, f"EXPLAIN {cypher}")
                    ).consume()
                    if plan.query_type != "r":
                        raise ToolException(
                            f"Refused a query that would write.\nCypher: {cypher}"
                        )
                    records = transaction.run(cast(LiteralString, cypher)).fetch(
                        CYPHER_ROW_LIMIT
                    )
                finally:
                    transaction.close()
        except neo4j.exceptions.Neo4jError as exc:
            if not _query_error(exc):
                raise
            raise ToolException(
                f"Cypher query failed: {exc.message}\nCypher: {cypher}"
            ) from exc
        return RawSearchResult(records=records, metadata={"cypher": cypher})


def _langchain_tool(tool: Tool) -> StructuredTool:
    def execute(query_text: str) -> tuple[str, list[neo4j.Record]]:
        result: RetrieverResult = tool.execute(query_text=query_text)
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
        rows = "\n".join(
            f"[{evidence_id(str(record['content']))}] {record['content']}"
            for record in records
        )
        cypher = result.metadata.get("cypher") if result.metadata else None
        if cypher is None:
            return rows, records
        header = (
            f"Cypher query: {cypher}\nRows returned (at most {CYPHER_ROW_LIMIT}): "
            f"{len(records)}"
        )
        return f"{header}\n{rows}", records

    return StructuredTool.from_function(
        func=execute,
        name=tool.get_name(),
        description=tool.get_description(),
        args_schema=tool.get_parameters(),
        response_format="content_and_artifact",
        handle_tool_error=True,
    )


@dataclass(frozen=True, slots=True)
class _AgentContext:
    cache_key: str


@wrap_model_call
def _question_cache_key(
    request: ModelRequest[_AgentContext],
    handler: Callable[[ModelRequest[_AgentContext]], ModelResponse],
) -> ModelResponse:
    """Give every agent call for one question the same prompt_cache_key.

    Each agent step resends the previous step's messages plus the newest tool
    result. The key routes those requests to the same OpenAI cache, so the
    repeated part is billed at the cached-input rate. What the model reads is
    unchanged.
    """
    settings = {
        **request.model_settings,
        "prompt_cache_key": request.runtime.context.cache_key,
    }
    return handler(request.override(model_settings=settings))


class AgenticToolsRetriever(ToolsRetriever):
    def __init__(
        self,
        *,
        driver: Driver,
        llm: LLMBase,
        agent_llm: BaseChatModel,
        tools: Sequence[Tool],
        neo4j_database: str,
        system_instruction: str,
    ) -> None:
        super().__init__(
            driver=driver,
            llm=llm,
            tools=tools,
            neo4j_database=neo4j_database,
            system_instruction=system_instruction,
        )
        self.result_formatter = _format_tool_record
        middleware: list[AgentMiddleware[Any, Any, Any]] = [
            ModelCallLimitMiddleware(run_limit=5),
            _question_cache_key,
        ]
        self._agent = create_agent(
            model=agent_llm,
            tools=[_langchain_tool(tool) for tool in self._tools],
            system_prompt=self.system_instruction,
            middleware=middleware,
            context_schema=_AgentContext,
        )

    def get_search_results(
        self,
        query_text: str,
        message_history: list[LLMMessage] | None = None,
        top_k: int = RETRIEVAL_TOP_K,
        **kwargs: Any,
    ) -> RawSearchResult:
        """Run the LangChain agent and pass on the evidence it picked.

        The agent ends by listing the IDs of the results the answer needs,
        from either tool; those come first, in its order. Remaining slots up
        to top_k are filled as before: pooled vector_search passages by best
        similarity score, then cypher_search rows in arrival order. If the
        agent names no valid ID, that old rule picks all top_k. The metadata
        records every tool call and, as a final "select_evidence" step, the
        agent's reply and how many of its picks were used.
        """
        messages = [*(message_history or []), {"role": "user", "content": query_text}]
        key = sha256(f"{self.neo4j_database}\0{query_text}".encode()).hexdigest()
        result = self._agent.invoke(
            {"messages": messages},
            context=_AgentContext(cache_key=f"recrag-agent-{key[:32]}"),
        )
        queries = {
            call["id"]: call["args"]
            for message in result["messages"]
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        }
        # A passage found by several searches keeps its best score.
        passages: dict[str, tuple[float, int, neo4j.Record]] = {}
        seen_rows: set[str] = set()
        rows: list[neo4j.Record] = []
        trace: list[dict[str, Any]] = []
        for message in result["messages"]:
            if not isinstance(message, ToolMessage):
                continue
            artifact: list[neo4j.Record] = message.artifact or []
            step = len(trace)
            trace.append(
                {
                    "step": step,
                    "tool": message.name,
                    "args": queries.get(message.tool_call_id, {}),
                    "status": message.status,
                    "output": str(message.content),
                    "results": len(artifact),
                }
            )
            for record in artifact:
                key = _evidence_key(str(record["content"]))
                score = (record["metadata"] or {}).get("score")
                if record["tool_name"] == "vector_search" and score is not None:
                    found = passages.get(key)
                    if found is None:
                        passages[key] = (float(score), len(passages), record)
                    elif float(score) > found[0]:
                        passages[key] = (float(score), found[1], record)
                elif key not in seen_rows:
                    seen_rows.add(key)
                    rows.append(record)
        # Highest score first; ties keep the order the agent found them in.
        ranked = sorted(passages.values(), key=lambda entry: (-entry[0], entry[1]))
        fallback = [record for _, _, record in ranked] + rows
        by_id = {evidence_id(str(record["content"])): record for record in fallback}
        final = result["messages"][-1]
        reply = final.text if isinstance(final, AIMessage) else ""
        picked: list[neo4j.Record] = []
        for found_id in EVIDENCE_ID.findall(reply):
            record = by_id.get(found_id)
            if record is not None and all(record is not p for p in picked):
                picked.append(record)
        kept = picked[:top_k]
        kept += [r for r in fallback if all(r is not k for k in kept)][
            : top_k - len(kept)
        ]
        trace.append(
            {
                "step": len(trace),
                "tool": "select_evidence",
                "args": {},
                "status": "success" if picked else "no_valid_ids",
                "output": reply,
                "results": min(len(picked), top_k),
            }
        )
        return RawSearchResult(
            records=kept,
            metadata={"trace": trace, "agent_messages": len(result["messages"])},
        )


@dataclass(slots=True)
class _PropertyUse:
    owner: str
    key: str
    uses: int
    total: int

    def common(self) -> bool:
        return self.key == "name" or self.uses >= max(
            MIN_PROPERTY_USES, self.total * MIN_PROPERTY_SHARE
        )


_ENTITY_PROPERTY_USES = """
MATCH (n:__Entity__)
UNWIND [label IN labels(n) WHERE NOT label STARTS WITH '__'] AS owner
WITH owner, count(n) AS total, collect(keys(n)) AS keys
UNWIND keys AS node_keys
UNWIND node_keys AS key
RETURN owner, key, count(*) AS uses, total
"""

_RELATIONSHIP_PROPERTY_USES = """
MATCH (:__Entity__)-[r]->(:__Entity__)
WITH type(r) AS owner, keys(r) AS relationship_keys
WITH owner, count(*) AS total, collect(relationship_keys) AS keys
UNWIND keys AS relationship_keys
UNWIND relationship_keys AS key
RETURN owner, key, count(*) AS uses, total
"""


def _property_uses(
    driver: Driver, database: str, query: LiteralString, timeout: float
) -> list[_PropertyUse]:
    records, _, _ = driver.execute_query(
        neo4j.Query(query, timeout=timeout),
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    return [
        _PropertyUse(record["owner"], record["key"], record["uses"], record["total"])
        for record in records
    ]


class _SchemaProperty(BaseModel):
    name: str = Field(alias="property")
    type: str


class _SchemaRelationship(BaseModel):
    start: str
    type: str
    end: str


class _StructuredSchema(BaseModel):
    node_props: dict[str, list[_SchemaProperty]]
    rel_props: dict[str, list[_SchemaProperty]]
    relationships: list[_SchemaRelationship]


def _schema_lines(
    properties: dict[str, list[_SchemaProperty]], uses: Sequence[_PropertyUse]
) -> list[str]:
    lines: list[str] = []
    for owner, entries in properties.items():
        counted = [use for use in uses if use.owner == owner]
        kept = [
            f"{entry.name}: {entry.type}"
            for entry in entries
            if not is_junk_property(entry.name)
            and (
                not counted
                or any(use.key == entry.name and use.common() for use in counted)
            )
        ]
        if kept:
            lines.append(f"{owner} {{{', '.join(kept)}}}")
    return lines


def graph_schema(driver: Driver, database: str, timeout: float) -> str:
    structured = _StructuredSchema.model_validate(
        get_structured_schema(driver, database=database, timeout=timeout)
    )
    node_uses = _property_uses(driver, database, _ENTITY_PROPERTY_USES, timeout)
    relationship_uses = _property_uses(
        driver, database, _RELATIONSHIP_PROPERTY_USES, timeout
    )
    return "\n".join(
        [
            "Node properties:",
            "__Entity__ {name: STRING, aliases: LIST}",
            *_schema_lines(structured.node_props, node_uses),
            "Relationship properties:",
            *_schema_lines(structured.rel_props, relationship_uses),
            "The relationships:",
            *[
                f"(:{pattern.start})-[:{pattern.type}]->(:{pattern.end})"
                for pattern in structured.relationships
            ],
        ]
    )


def build_agentic_retriever(
    driver: Driver,
    llm: LLMBase,
    embedder: GraphRAGEmbeddings,
    database: str,
    agent_llm: BaseChatModel,
    *,
    cypher_timeout: float,
    schema_timeout: float,
    neo4j_schema: str | None = None,
    ontology: GraphSchema = GENERAL_ONTOLOGY_SCHEMA,
) -> AgenticToolsRetriever:
    if no_graph_agent():
        vector_only = _ChunkVectorRetriever(
            driver=driver,
            database=database,
            embedder=embedder,
            retrieval_query=NO_GRAPH_VECTOR_QUERY,
        )
        return AgenticToolsRetriever(
            driver=driver,
            llm=llm,
            agent_llm=agent_llm,
            tools=[
                vector_only.convert_to_tool(
                    name="vector_search",
                    description=(
                        "Find source text passages similar in meaning to the "
                        "query. Each result has the passage and its page title."
                    ),
                    parameter_descriptions={
                        "query_text": "What to search for, in plain words.",
                    },
                ),
            ],
            neo4j_database=database,
            system_instruction=NO_GRAPH_AGENT_INSTRUCTION,
        )
    cypher = _BoundedText2CypherRetriever(
        driver=driver,
        llm=llm,
        database=database,
        neo4j_schema=without_passage_text(
            neo4j_schema or graph_schema(driver, database, schema_timeout)
        ),
        timeout=cypher_timeout,
    )
    vector = _ChunkVectorRetriever(driver=driver, database=database, embedder=embedder)
    return AgenticToolsRetriever(
        driver=driver,
        llm=llm,
        agent_llm=agent_llm,
        tools=[
            vector.convert_to_tool(
                name="vector_search",
                description=(
                    "Find source text passages similar in meaning to the query. "
                    "Each result has the passage, its page title, the entities it "
                    "mentions, and a few graph facts about them."
                ),
                parameter_descriptions={
                    "query_text": "What to search for, in plain words.",
                },
            ),
            cypher.convert_to_tool(
                name="cypher_search",
                description=(
                    "Query the knowledge graph extracted from the source text: "
                    f"{graph_contents(ontology)} and the relationships between them. "
                    "Generates and runs a read-only Cypher query. Returns graph "
                    "facts (names, titles, dates, relationships), never passage text."
                ),
                parameter_descriptions={
                    "query_text": (
                        "A short lookup about specific named entities, such as "
                        "'what is connected to <name>' or 'which articles "
                        "mention <name>'."
                    ),
                },
            ),
        ],
        neo4j_database=database,
        system_instruction=AGENT_INSTRUCTION,
    )


LLMInput = str | list[LLMMessage]
LLMHistory = list[LLMMessage] | MessageHistory | None
ResponseFormat = type[BaseModel] | dict[str, Any] | None
ModelOutput = BaseMessage | BaseModel | dict[str, Any]


class GraphRAGChatLLM(LLMBase):
    supports_structured_output = True

    def __init__(self, model: ChatOpenAI) -> None:
        super().__init__(
            model_name=model.model_name, rate_limit_handler=NoOpRateLimitHandler()
        )
        self._model = model

    @staticmethod
    def _chat_message(message: LLMMessage) -> BaseMessage:
        if message["role"] == "system":
            return SystemMessage(message["content"])
        if message["role"] == "assistant":
            return AIMessage(message["content"])
        return HumanMessage(message["content"])

    @classmethod
    def _messages(
        cls,
        input: LLMInput,
        history: LLMHistory,
        system_instruction: str | None,
    ) -> list[BaseMessage]:
        if isinstance(input, list):
            messages = [cls._chat_message(message) for message in input]
        else:
            previous = (
                history.messages
                if isinstance(history, MessageHistory)
                else history or []
            )
            messages = [
                *(cls._chat_message(message) for message in previous),
                HumanMessage(input),
            ]
        if system_instruction:
            messages.insert(0, SystemMessage(system_instruction))
        return messages

    @staticmethod
    def _content(response: ModelOutput) -> str:
        if isinstance(response, BaseMessage):
            return response.text
        if isinstance(response, BaseModel):
            return response.model_dump_json()
        return json.dumps(response)

    def _extract(self, messages: list[BaseMessage], **kwargs: Any) -> Neo4jGraph:
        messages.append(HumanMessage(EXTRACTION_INSTRUCTION))
        local_chat = bool(os.environ.get("RECRAG_CHAT_BASE_URL"))
        structured = self._model.with_structured_output(
            ExtractedGraph,
            method="function_calling" if local_chat else "json_schema",
            strict=None if local_chat else True,
        )

        def extract() -> Neo4jGraph:
            response = structured.invoke(messages, **kwargs)
            return graph_from_extraction(ExtractedGraph.model_validate(response))

        cache = self._model.cache
        if isinstance(cache, SQLiteGenerationCache):
            cache.forget_hit()
            try:
                return extract()
            except Exception:
                if not cache.evict_hit():
                    raise
        return extract()

    def _invoke(
        self,
        input: LLMInput,
        history: LLMHistory,
        system_instruction: str | None,
        response_format: ResponseFormat,
        **kwargs: Any,
    ) -> ModelOutput:
        messages = self._messages(input, history, system_instruction)
        if response_format is Neo4jGraph:
            return self._extract(messages, **kwargs)
        model = (
            self._model.with_structured_output(
                response_format, method="function_calling"
            )
            if response_format
            else self._model
        )
        return model.invoke(messages, **kwargs)

    def invoke(
        self,
        input: LLMInput,
        message_history: LLMHistory = None,
        system_instruction: str | None = None,
        response_format: ResponseFormat = None,
        **kwargs: Any,
    ) -> LLMResponse:
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
        system_instruction: str | None = None,
        response_format: ResponseFormat = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await asyncio.to_thread(
            self.invoke,
            input,
            message_history,
            system_instruction,
            response_format,
            **kwargs,
        )


class GraphRAG:
    def __init__(
        self,
        *,
        driver: Driver,
        llm: GraphRAGChatLLM,
        cypher_llm: GraphRAGChatLLM,
        construction_llm: GraphRAGChatLLM,
        agent_llm: BaseChatModel,
        embedder: GraphRAGEmbeddings,
        answer_llm: ChatOpenAI,
        config: GraphRAGConfig,
        ontology: GraphSchema,
    ) -> None:
        self._driver = driver
        self._llm = llm
        self._cypher_llm = cypher_llm
        self._construction_llm = construction_llm
        self._agent_llm = agent_llm
        self._embedder = embedder
        self._answer_llm = answer_llm
        self._config = config
        self._ontology = ontology
        self._retrievers: list[AgenticToolsRetriever] = []
        self.last_question_cost: QuestionCost | None = None

    @classmethod
    def from_config(
        cls,
        config_path: Path | None = None,
        dataset: str | None = None,
    ) -> GraphRAG:
        load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
        config = GraphRAGConfig.load(config_path)
        auth = os.environ["NEO4J_AUTH"]
        username, password = auth.split("/", 1) if "/" in auth else ["neo4j", auth]
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(username, password),
            connection_timeout=config.graphrag.timeout,
        )

        base_url = os.environ.get("RECRAG_CHAT_BASE_URL")

        def chat_model(model_name: str, effort: str | None = None) -> ChatOpenAI:
            configured = model_name == config.llm.model
            local_chat = bool(base_url) and configured
            return ChatOpenAI(
                model=model_name,
                timeout=config.llm.timeout,
                base_url=base_url if local_chat else None,
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
                http_client=MeteredClient(timeout=config.llm.timeout),
                http_async_client=MeteredAsyncClient(timeout=config.llm.timeout),
                reasoning=(
                    {"effort": effort or config.llm.reasoning_effort}
                    if configured and not local_chat
                    else None
                ),
            )

        try:
            answer_model = chat_model(config.llm.model)
            cypher_effort = config.graphrag.cypher_reasoning_effort
            cypher_chat = (
                chat_model(config.llm.model, cypher_effort)
                if cypher_effort not in {None, config.llm.reasoning_effort}
                else answer_model
            )
            construction_chat = chat_model(config.construction_model)
            agent_chat = (
                construction_chat
                if config.agent_model == config.construction_model
                else chat_model(config.agent_model)
            )
            return cls(
                driver=driver,
                llm=GraphRAGChatLLM(answer_model),
                cypher_llm=GraphRAGChatLLM(cypher_chat),
                construction_llm=GraphRAGChatLLM(construction_chat),
                agent_llm=agent_chat,
                embedder=GraphRAGEmbeddings(
                    model=config.embedding.model,
                    dimensions=config.embedding.dimensions,
                    timeout=config.embedding.timeout,
                ),
                answer_llm=answer_model,
                config=config,
                ontology=ontology_for(dataset, config.graphrag.ontology),
            )
        except Exception:
            driver.close()
            raise

    def retriever(
        self, database: str, neo4j_schema: str | None = None
    ) -> AgenticToolsRetriever:
        for retriever in self._retrievers:
            if retriever.neo4j_database == database:
                return retriever
        retriever = build_agentic_retriever(
            self._driver,
            self._cypher_llm,
            self._embedder,
            database,
            self._agent_llm,
            neo4j_schema=neo4j_schema,
            ontology=self._ontology,
            cypher_timeout=self._config.graphrag.retrieval.timeout,
            schema_timeout=self._config.graphrag.timeout,
        )
        self._retrievers.append(retriever)
        return retriever

    def construct(self, pages: Sequence[SourcePage], *, database: str) -> None:
        from benchmark import split_pages

        source = [
            PageChunks(title=page.title, chunks=split_pages([page])) for page in pages
        ]
        self._retrievers = [
            retriever
            for retriever in self._retrievers
            if retriever.neo4j_database != database
        ]
        model = self._construction_llm._model
        original_cache = model.cache
        cache_root = Path(__file__).resolve().parent / "runs"
        namespace = os.environ.get("RECRAG_RUN_NAMESPACE")
        if namespace:
            cache_root /= namespace
        cache_path = (
            cache_root / "graphrag" / "construction-cache" / f"{database}.sqlite"
        )
        model.cache = SQLiteGenerationCache(
            cache_path, identity=self._config.construction_key(self._ontology)
        )
        self._embedder.store = SQLiteEmbeddingCache(cache_path)
        try:
            self._embedder.embed_all(
                [chunk for page in source for chunk in page.chunks]
            )
            rebuild_graph(
                source,
                driver=self._driver,
                llm=self._construction_llm,
                embedder=self._embedder,
                database=database,
                embedding_dimensions=self._config.embedding.dimensions,
                schema=self._ontology,
                index_timeout=self._config.graphrag.index_timeout,
                extraction_concurrency=self._config.graphrag.extraction_concurrency,
            )
        finally:
            model.cache = original_cache
            self._embedder.store = None

    def answer(self, question: str, *, database: str) -> RagResultModel:
        with measure(self._config.pricing) as question_cost:
            try:
                return Neo4jGraphRAG(self.retriever(database), self._llm).search(
                    question,
                    retriever_config={"top_k": RETRIEVAL_TOP_K},
                    return_context=True,
                    response_fallback=NO_CONTEXT,
                )
            finally:
                self.last_question_cost = question_cost

    def close(self) -> None:
        self._driver.close()
        self._embedder.client.close()
        self._answer_llm.root_client.close()
        cypher_model = self._cypher_llm._model
        if cypher_model is not self._answer_llm:
            cypher_model.root_client.close()
        construction_model = self._construction_llm._model
        if construction_model is not self._answer_llm:
            construction_model.root_client.close()
        if (
            isinstance(self._agent_llm, ChatOpenAI)
            and self._agent_llm is not self._answer_llm
            and self._agent_llm is not construction_model
        ):
            self._agent_llm.root_client.close()


@dataclass(frozen=True, slots=True)
class GraphPipeline:
    rag: GraphRAG
    llm: ResponsesLLM
    embeddings: LlamaEmbedding


@contextmanager
def setup(settings: Settings, index_dir: Path, dataset: str) -> Iterator[GraphPipeline]:
    from rag_retrieval import setup as setup_answer

    rag = GraphRAG.from_config(dataset=dataset)
    try:
        with setup_answer(settings, index_dir, dataset) as answer:
            yield GraphPipeline(rag=rag, llm=answer.llm, embeddings=answer.embeddings)
    finally:
        rag.close()


def retrieve(
    *,
    rag: GraphRAG,
    pages: Sequence[SourcePage],
    question: str,
    index_dir: Path,
    top_k: int,
) -> RetrievedContext:
    from benchmark import RetrievedContext

    database = f"recrag-benchmark-{index_dir.name[:24]}"
    marker = index_dir / "graph.complete"
    started = perf_counter()
    with indexing():
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
            rag.construct(pages, database=database)
            marker.write_text(database)
            construction_seconds = perf_counter() - started
        # The schema scans every entity, so work it out once per built graph,
        # not once per question; the benchmark builds a new GraphRAG per record.
        schema_path = index_dir / "graph.schema"
        if construction_seconds or not schema_path.exists():
            staged = schema_path.with_suffix(f".{os.getpid()}.tmp")
            staged.write_text(
                graph_schema(rag._driver, database, rag._config.graphrag.timeout)
            )
            staged.replace(schema_path)
        schema = schema_path.read_text()
    started = perf_counter()
    result = rag.retriever(database, neo4j_schema=schema).search(question, top_k=top_k)
    metadata = result.metadata or {}
    return RetrievedContext(
        passages=[answer_text(str(item.content)) for item in result.items],
        evidence=[str(item.content) for item in result.items],
        trace=metadata.get("trace", []),
        scores=[
            float(item.metadata["score"])
            if item.metadata and item.metadata.get("score") is not None
            else NO_SCORE
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
