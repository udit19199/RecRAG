"""Ontology-guided graph construction in Neo4j."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version
from typing import Literal

from neo4j import Driver
from neo4j_graphrag.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.components.kg_writer import KGWriterModel, Neo4jWriter
from neo4j_graphrag.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.types import (
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
    PropertyValue,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import LLMBase
from pydantic import BaseModel

OntologyChoice = Literal["general", "dataset"]

GENERAL_ONTOLOGY_SCHEMA = GraphSchema(
    node_types=(
        NodeType(
            label="Person",
            description="A human.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="birth_date",
                    type="STRING",
                    description="Birth date as written.",
                ),
                PropertyType(
                    name="death_date",
                    type="STRING",
                    description="Death date as written.",
                ),
                PropertyType(
                    name="nationality",
                    type="STRING",
                    description="Nationality as written.",
                ),
                PropertyType(
                    name="occupation",
                    type="STRING",
                    description="Job or role as written.",
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Organization",
            description="A company, school, team, agency, band, party, or other group.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="kind", type="STRING", description="Kind of group as written."
                ),
                PropertyType(
                    name="founded_on",
                    type="STRING",
                    description="Founding date as written.",
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Place",
            description="A country, region, city, venue, or building.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="kind", type="STRING", description="Kind of place as written."
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Event",
            description="A dated happening: war, election, game, trial, launch, deal, or ceremony.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="date", type="STRING", description="Event date as written."
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Work",
            description="A made thing: book, film, song, show, article, product, or app.",
            properties=[
                PropertyType(name="name", type="STRING", description="Title or name."),
                PropertyType(
                    name="kind", type="STRING", description="Kind of work as written."
                ),
                PropertyType(
                    name="date",
                    type="STRING",
                    description="Release or publish date as written.",
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Concept",
            description="A topic, field, genre, language, award, or idea.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="kind",
                    type="STRING",
                    description="Kind of concept as written.",
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Thing",
            description="Fallback when no narrower kind fits.",
            properties=[PropertyType(name="name", type="STRING", description="Name.")],
            additional_properties=True,
        ),
    ),
    relationship_types=(
        RelationshipType(
            label="MEMBER_OF", description="Person or group belongs to a group."
        ),
        RelationshipType(
            label="WORKS_FOR", description="Person works for or leads an organization."
        ),
        RelationshipType(label="LOCATED_IN", description="Inside or based in a place."),
        RelationshipType(label="PART_OF", description="Part of a larger whole."),
        RelationshipType(
            label="CREATED", description="Person or organization made a work."
        ),
        RelationshipType(label="PARTICIPATED_IN", description="Took part in an event."),
        RelationshipType(
            label="RELATED_TO",
            description="Any other stated link; prefer a specific type.",
        ),
    ),
    additional_node_types=True,
    additional_relationship_types=True,
)


BIOGRAPHY_ONTOLOGY_SCHEMA = GraphSchema(
    node_types=(
        NodeType(
            label="Person",
            description="A human.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="gender",
                    type="STRING",
                    description="Male, female, or other identity as stated. Omit when not stated.",
                ),
                PropertyType(
                    name="birth_date",
                    type="STRING",
                    description="Birth date as written.",
                ),
                PropertyType(
                    name="death_date",
                    type="STRING",
                    description="Death date as written.",
                ),
                PropertyType(
                    name="nationality",
                    type="STRING",
                    description="Nationality as written.",
                ),
                PropertyType(
                    name="occupation",
                    type="STRING",
                    description="Job or role as written.",
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Organization",
            description="A group, firm, or school.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="kind",
                    type="STRING",
                    description="Company, school, or group as written.",
                ),
                PropertyType(
                    name="founded_on",
                    type="STRING",
                    description="Founding date as written.",
                ),
                PropertyType(
                    name="headquarters",
                    type="STRING",
                    description="Head place as written.",
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Place",
            description="A city, country, or building.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="kind",
                    type="STRING",
                    description="City, country, or building as written.",
                ),
                PropertyType(
                    name="country", type="STRING", description="Country as written."
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="CreativeWork",
            description="A book, film, or song.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="kind",
                    type="STRING",
                    description="Book, film, or song as written.",
                ),
                PropertyType(
                    name="published_on",
                    type="STRING",
                    description="Publish date as written.",
                ),
                PropertyType(
                    name="language", type="STRING", description="Language as written."
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Event",
            description="A war, award ceremony, or election.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="held_on", type="STRING", description="Event date as written."
                ),
                PropertyType(
                    name="location",
                    type="STRING",
                    description="Event place as written.",
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Concept",
            description="An award, field, genre, or language.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="kind",
                    type="STRING",
                    description="Award, field, or genre as written.",
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Thing",
            description="Fallback when no narrower kind fits.",
            properties=[PropertyType(name="name", type="STRING", description="Name.")],
            additional_properties=True,
        ),
    ),
    relationship_types=(
        RelationshipType(label="BORN_IN", description="Person born in Place."),
        RelationshipType(label="DIED_IN", description="Person died in Place."),
        RelationshipType(label="LOCATED_IN", description="Inside a Place."),
        RelationshipType(label="PART_OF", description="Part of a whole."),
        RelationshipType(label="MEMBER_OF", description="Person in Organization."),
        RelationshipType(
            label="CREATED_BY", description="Work made by Person or Organization."
        ),
        RelationshipType(label="SPOUSE_OF", description="Married to."),
        RelationshipType(label="CHILD_OF", description="Child of."),
        RelationshipType(label="AWARDED", description="Given award."),
        RelationshipType(
            label="HAS_NATIONALITY", description="Person linked to Place."
        ),
    ),
    additional_node_types=True,
    additional_relationship_types=True,
)


NEWS_ONTOLOGY_SCHEMA = GraphSchema(
    node_types=(
        NodeType(
            label="Article",
            description="A news article.",
            properties=[
                PropertyType(name="name", type="STRING", description="Headline."),
                PropertyType(
                    name="published_on",
                    type="STRING",
                    description="Publish date as written.",
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Publisher",
            description="A news outlet that publishes articles.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Person",
            description="A human.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="role", type="STRING", description="Job or title as written."
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Organization",
            description="A company, team, agency, or group.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="kind",
                    type="STRING",
                    description="Company, team, agency, or group as written.",
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Product",
            description="A product, app, service, or platform.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Event",
            description="A trial, game, launch, deal, or other dated happening.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
                PropertyType(
                    name="held_on", type="STRING", description="Event date as written."
                ),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Place",
            description="A city, country, or venue.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Topic",
            description="A subject, field, or issue.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
            ],
            additional_properties=True,
        ),
        NodeType(
            label="Thing",
            description="Fallback when no narrower kind fits.",
            properties=[
                PropertyType(name="name", type="STRING", description="Name."),
            ],
            additional_properties=True,
        ),
    ),
    relationship_types=(
        RelationshipType(
            label="PUBLISHED_BY", description="Article published by Publisher."
        ),
        RelationshipType(label="MENTIONS", description="Article mentions an entity."),
        RelationshipType(
            label="WORKS_FOR", description="Person works for or leads Organization."
        ),
        RelationshipType(
            label="FOUNDED",
            description="Person or Organization founded Organization or Product.",
        ),
        RelationshipType(
            label="OWNS",
            description="Organization owns or acquired Organization or Product.",
        ),
        RelationshipType(
            label="MADE", description="Organization made or offers Product."
        ),
        RelationshipType(
            label="PARTICIPATED_IN",
            description="Person or Organization took part in Event.",
        ),
        RelationshipType(
            label="ACCUSED_OF",
            description="Person or Organization accused of or charged with Topic or Event.",
        ),
        RelationshipType(
            label="SAID",
            description="Person or Organization stated a claim about an entity.",
        ),
        RelationshipType(label="LOCATED_IN", description="Inside or based in a Place."),
        RelationshipType(label="PART_OF", description="Part of a whole."),
        RelationshipType(label="COMPETES_WITH", description="Rival or opponent."),
    ),
    additional_node_types=True,
    additional_relationship_types=True,
)


def ontology_for(
    dataset: str | None, choice: OntologyChoice = "general"
) -> GraphSchema:
    if choice == "general":
        return GENERAL_ONTOLOGY_SCHEMA
    if dataset == "multihop_rag":
        return NEWS_ONTOLOGY_SCHEMA
    if dataset == "hotpotqa":
        return BIOGRAPHY_ONTOLOGY_SCHEMA
    # Natural Questions and NovelQA pages cover any topic (decided 10 Nov 2026).
    return GENERAL_ONTOLOGY_SCHEMA


ONTOLOGY_EXTRACTION_PROMPT = ERExtractionTemplate.DEFAULT_TEMPLATE + (
    "\n\nRules:\n"
    "- Every node needs a name.\n"
    "- Fill a field only when text states it. Omit it when not stated.\n"
    "- Use Thing only when no narrower kind fits.\n"
    "- Link name is UPPER_SNAKE, short, from text.\n"
    "- One fact per link. No facts outside text.\n"
)


EXTRACTION_INSTRUCTION = (
    "Extract the entities and relationships only from the Input text section of "
    "the preceding message, never from formatting examples, using the response "
    "schema. Encode each node's and relationship's properties as a list of "
    "entries, or an empty list when there are none. In each entry, key is the "
    "property name in lower_snake_case, such as name, aliases, or occupation, "
    "and value is the stated fact. Every node has exactly one name entry whose "
    "value is a single string: the entity's main name. Put other names for the "
    "same entity in one aliases entry whose value is a list of strings. When any "
    "other property has several stated values, give them as one list value."
)


CONSTRUCTION_METHOD = "ontology_guided"


def database_name(record_id: str, run_id: str) -> str:
    record_key = sha256(record_id.encode()).hexdigest()[:8]
    return f"recrag-{CONSTRUCTION_METHOD.replace('_', '-')}-{record_key}-{run_id}"


class ExtractedProperty(BaseModel):
    key: str
    value: PropertyValue


class ExtractedNode(BaseModel):
    id: str
    label: str
    properties: list[ExtractedProperty]


class ExtractedRelationship(BaseModel):
    start_node_id: str
    end_node_id: str
    type: str
    properties: list[ExtractedProperty]


class ExtractedGraph(BaseModel):
    nodes: list[ExtractedNode]
    relationships: list[ExtractedRelationship]


_JUNK_PROPERTY = re.compile(
    r"key|value|key[^a-z].*|.*[\s:?].*|__.*|embedding", re.IGNORECASE
)


def is_junk_property(key: str) -> bool:
    return not key or _JUNK_PROPERTY.fullmatch(key) is not None


_PRIMITIVE = str | int | float | bool


@dataclass(slots=True)
class _Property:
    key: str
    values: list[PropertyValue] = field(default_factory=list)

    def add(self, value: PropertyValue) -> None:
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str):
                item = item.strip()
            if item == "" or item is None or item in self.values:
                continue
            self.values.append(item)

    def value(self) -> PropertyValue:
        if len(self.values) == 1:
            return self.values[0]
        if len({type(item) for item in self.values}) > 1 or not all(
            isinstance(item, _PRIMITIVE) for item in self.values
        ):
            return [str(item) for item in self.values]
        return [item for item in self.values if isinstance(item, _PRIMITIVE)]


def _properties(entries: Sequence[ExtractedProperty]) -> list[_Property]:
    properties: list[_Property] = []
    pending: str | None = None

    def add(key: str, value: PropertyValue) -> None:
        match = next((item for item in properties if item.key == key), None)
        if match is None:
            match = _Property(key)
            properties.append(match)
        match.add(value)

    for entry in entries:
        key = entry.key.strip()
        if key.lower() == "key" and isinstance(entry.value, str):
            candidate = entry.value.strip()
            pending = None if is_junk_property(candidate) else candidate
            continue
        if key.lower() == "value" and pending is not None:
            add(pending, entry.value)
            pending = None
            continue
        pending = None
        if not is_junk_property(key):
            add(key, entry.value)
    return [item for item in properties if item.values]


def _node(extracted: ExtractedNode) -> Neo4jNode | None:
    properties = _properties(extracted.properties)
    names = [
        str(value) for item in properties if item.key == "name" for value in item.values
    ]
    if not names:
        return None
    seen = {names[0].lower()}
    aliases: list[str] = []
    for value in names[1:] + [
        str(value)
        for item in properties
        if item.key in {"alias", "aliases"}
        for value in item.values
    ]:
        if value.lower() not in seen:
            seen.add(value.lower())
            aliases.append(value)
    node_properties: dict[str, PropertyValue] = {"name": names[0]}
    if aliases:
        node_properties["aliases"] = aliases
    for item in properties:
        if item.key not in {"name", "alias", "aliases"}:
            node_properties[item.key] = item.value()
    return Neo4jNode(id=extracted.id, label=extracted.label, properties=node_properties)


def graph_from_extraction(extracted: ExtractedGraph) -> Neo4jGraph:
    nodes = [node for node in map(_node, extracted.nodes) if node is not None]
    ids = {node.id for node in nodes}
    return Neo4jGraph(
        nodes=nodes,
        relationships=[
            Neo4jRelationship(
                start_node_id=relationship.start_node_id,
                end_node_id=relationship.end_node_id,
                type=relationship.type,
                properties={
                    item.key: item.value()
                    for item in _properties(relationship.properties)
                },
            )
            for relationship in extracted.relationships
            if relationship.start_node_id in ids and relationship.end_node_id in ids
        ],
    )


def construction_key(
    *,
    model: str,
    reasoning_effort: str,
    endpoint: str,
    embedding_model: str,
    embedding_dimensions: int,
    chunking: str,
    ontology: GraphSchema,
) -> str:
    return sha256(
        "\0".join(
            [
                model,
                reasoning_effort,
                endpoint,
                embedding_model,
                str(embedding_dimensions),
                chunking,
                ontology.model_dump_json(),
                ONTOLOGY_EXTRACTION_PROMPT,
                EXTRACTION_INSTRUCTION,
                json.dumps(ExtractedGraph.model_json_schema(), sort_keys=True),
                version("neo4j-graphrag"),
            ]
        ).encode()
    ).hexdigest()


@dataclass(slots=True)
class PageChunks:
    title: str
    chunks: list[str]


class _RaisingNeo4jWriter(Neo4jWriter):
    async def run(
        self, graph: Neo4jGraph, lexical_graph_config: LexicalGraphConfig
    ) -> KGWriterModel:
        result = await super().run(graph, lexical_graph_config)
        if result.status != "SUCCESS":
            raise RuntimeError(
                str(result.metadata.get("error", "Graph write failed."))
                if result.metadata
                else "Graph write failed."
            )
        return result


class _PreparedChunksSplitter(TextSplitter):
    def __init__(self, chunks: Sequence[str]) -> None:
        self._chunks = list(chunks)

    async def run(self, text: str) -> TextChunks:
        return TextChunks(
            chunks=[
                TextChunk(text=chunk, index=index)
                for index, chunk in enumerate(self._chunks)
            ]
        )


_CLEAR_TEMPORARY_IDS = """
MATCH (n:__KGBuilder__) WHERE n.__tmp_internal_id IS NOT NULL
CALL (n) { SET n.__tmp_internal_id = null } IN TRANSACTIONS OF 10000 ROWS
"""

_DELETE_INLINE_DOCUMENT_LINKS = """
MATCH (:Chunk)-[link:FROM_DOCUMENT]->(:Document)
CALL (link) { DELETE link } IN TRANSACTIONS OF 10000 ROWS
"""

_CREATE_PAGE_DOCUMENTS = """
MATCH (inline:Document) DELETE inline
WITH count(*) AS deleted
UNWIND range(0, size($titles) - 1) AS page
CREATE (:__KGBuilder__:Document {page: page, title: $titles[page], name: $titles[page]})
"""

_LINK_CHUNKS_TO_PAGES = """
MATCH (document:Document)
WITH document ORDER BY document.page
WITH collect(document) AS documents
MATCH (chunk:Chunk)
CALL (chunk, documents) {
    WITH chunk, documents[$pages[chunk.index]] AS document
    CREATE (chunk)-[:FROM_DOCUMENT]->(document)
} IN TRANSACTIONS OF 10000 ROWS
"""

_UNLINK_CROSS_PAGE_CHUNKS = """
MATCH (previous:Chunk)-[link:NEXT_CHUNK]->(next:Chunk)
WHERE $pages[previous.index] <> $pages[next.index]
CALL (link) { DELETE link } IN TRANSACTIONS OF 10000 ROWS
"""

_MERGE_CASE_VARIANTS = """
MATCH (entity:__Entity__) WHERE entity.name IS :: STRING
WITH entity,
     [label IN labels(entity) WHERE NOT label STARTS WITH '__'][0] AS label,
     toLower(trim(entity.name)) AS key
WITH label, key, entity ORDER BY COUNT { (entity)--() } DESC
WITH label, key, collect(entity) AS entities
WHERE size(entities) > 1
CALL (entities) {
    WITH entities,
         toLower(trim(entities[0].name)) AS name,
         apoc.coll.toSet(reduce(
             collected = [], entity IN entities | collected + coalesce(entity.aliases, [])
         )) AS aliases
    CALL apoc.refactor.mergeNodes(
        entities, {properties: 'discard', mergeRels: true}
    ) YIELD node
    WITH node, [alias IN aliases WHERE toLower(trim(alias)) <> name] AS aliases
    SET node.aliases = CASE WHEN size(aliases) > 0 THEN aliases END
} IN TRANSACTIONS OF 200 ROWS
"""

_DELETE_SELF_LOOPS = "MATCH (entity:__Entity__)-[loop]->(entity) DELETE loop"

_COUNT_EMPTY_CHUNKS = """
MATCH (chunk:Chunk)
RETURN count(chunk) AS chunks,
       sum(CASE WHEN EXISTS { (:__Entity__)-[:FROM_CHUNK]->(chunk) } THEN 0 ELSE 1 END)
           AS empty
"""


async def _run_pipeline(pipeline: SimpleKGPipeline, workers: int) -> None:
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=workers + 4)
    )
    await pipeline.run_async(text=" ")


def rebuild_graph(
    pages: Sequence[PageChunks],
    *,
    driver: Driver,
    llm: LLMBase,
    embedder: Embedder,
    database: str,
    embedding_dimensions: int,
    schema: GraphSchema,
    index_timeout: float,
    extraction_concurrency: int,
) -> None:
    pages = [page for page in pages if page.chunks]
    if not pages:
        raise ValueError("Agentic Hybrid RAG needs nonempty source chunks.")
    chunks = [chunk for page in pages for chunk in page.chunks]
    page_of_chunk = [number for number, page in enumerate(pages) for _ in page.chunks]

    driver.execute_query(
        "CREATE OR REPLACE DATABASE $name WAIT", name=database, database_="system"
    )
    driver.execute_query(
        "CREATE INDEX __entity__tmp_internal_id IF NOT EXISTS "
        "FOR (n:__KGBuilder__) ON (n.__tmp_internal_id)",
        database_=database,
    )
    pipeline = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=schema,
        prompt_template=ONTOLOGY_EXTRACTION_PROMPT,
        on_error="RAISE",
        from_file=False,
        kg_writer=_RaisingNeo4jWriter(driver, neo4j_database=database, clean_db=False),
        text_splitter=_PreparedChunksSplitter(chunks),
        lexical_graph_config=LexicalGraphConfig(),
        perform_entity_resolution=False,
        neo4j_database=database,
    )
    extractor = pipeline.runner.pipeline.get_node_by_name("extractor").component
    if not isinstance(extractor, LLMEntityRelationExtractor):
        raise TypeError("SimpleKGPipeline has no LLM extractor to configure.")
    extractor.max_concurrency = extraction_concurrency
    asyncio.run(_run_pipeline(pipeline, extraction_concurrency))

    with driver.session(database=database) as session:
        session.run(_CLEAR_TEMPORARY_IDS).consume()
        session.run("DROP INDEX __entity__tmp_internal_id IF EXISTS").consume()
        session.run(_DELETE_INLINE_DOCUMENT_LINKS).consume()
        session.run(
            _CREATE_PAGE_DOCUMENTS, titles=[page.title for page in pages]
        ).consume()
        session.run(_LINK_CHUNKS_TO_PAGES, pages=page_of_chunk).consume()
        session.run(_UNLINK_CROSS_PAGE_CHUNKS, pages=page_of_chunk).consume()
        session.run("MATCH (document:Document) SET document.page = null").consume()
        session.run(_MERGE_CASE_VARIANTS).consume()
        session.run(_DELETE_SELF_LOOPS).consume()
        counts = session.run(_COUNT_EMPTY_CHUNKS).single(strict=True)
    print(
        f"Agentic Hybrid RAG: {counts['empty']} of {counts['chunks']} chunks "
        f"have no extracted entities ({database}).",
        file=sys.stderr,
        flush=True,
    )

    create_vector_index(
        driver,
        "chunk_embeddings",
        label="Chunk",
        embedding_property="embedding",
        dimensions=embedding_dimensions,
        similarity_fn="cosine",
        neo4j_database=database,
    )
    driver.execute_query(
        "CALL db.awaitIndexes($timeout)",
        timeout=int(index_timeout),
        database_=database,
    )
