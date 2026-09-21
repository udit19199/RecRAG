from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from neo4j import Driver
from neo4j_graphrag.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import LLMBase

ONTOLOGY_SCHEMA = GraphSchema(
    node_types=[
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
    ],
    relationship_types=[
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
    ],
    additional_node_types=True,
    additional_relationship_types=True,
)

ONTOLOGY_EXTRACTION_PROMPT = ERExtractionTemplate.DEFAULT_TEMPLATE + (
    "\n\nRules:\n"
    "- Every node needs a name.\n"
    "- Fill a field only when text states it. Omit it when not stated.\n"
    "- Use Thing only when no narrower kind fits.\n"
    "- Link name is UPPER_SNAKE, short, from text.\n"
    "- One fact per link. No facts outside text.\n"
)


class ConstructionMethod(StrEnum):
    STANDARD = "standard"
    ONTOLOGY_GUIDED = "ontology_guided"

    def database_name(self, record_id: str) -> str:
        return f"recrag-{self.value.replace('_', '-')}-{record_id}"


@dataclass(slots=True)
class SourcePage:
    title: str
    passages: list[str]


def rebuild_graph(
    pages: Sequence[SourcePage],
    *,
    driver: Driver,
    method: ConstructionMethod,
    llm: LLMBase,
    embedder: Embedder,
    database: str,
    embedding_dimensions: int,
) -> None:
    match method:
        case ConstructionMethod.STANDARD:
            schema = None
            prompt_template = ERExtractionTemplate.DEFAULT_TEMPLATE
        case ConstructionMethod.ONTOLOGY_GUIDED:
            schema = ONTOLOGY_SCHEMA
            prompt_template = ONTOLOGY_EXTRACTION_PROMPT
        case _:
            raise ValueError(f"Unsupported construction method: {method}")

    page_texts = []
    for page in pages:
        passages_text = "\n\n".join(page.passages)
        page_texts.append(f"Page: {page.title}\n{passages_text}")
    text = "\n\n".join(page_texts)

    # Keep each entity-to-chunk embedding independently searchable.
    driver.execute_query(
        "CREATE DATABASE $name IF NOT EXISTS WAIT", name=database, database_="system"
    )
    pipeline = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=schema,
        prompt_template=prompt_template,
        from_file=False,
        text_splitter=FixedSizeSplitter(chunk_size=1000, chunk_overlap=100),
        neo4j_database=database,
    )
    asyncio.run(pipeline.run_async(text=text))

    driver.execute_query(
        """
        MATCH (entity:__Entity__)-[:FROM_CHUNK]->(chunk:Chunk)
        WHERE chunk.embedding IS NOT NULL
        CREATE (entity)-[:HAS_EMBEDDING]->(:EntityEmbedding {
            text: chunk.text,
            embedding: chunk.embedding
        })
        """,
        database_=database,
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

    create_vector_index(
        driver,
        "entity_embeddings",
        label="EntityEmbedding",
        embedding_property="embedding",
        dimensions=embedding_dimensions,
        similarity_fn="cosine",
        neo4j_database=database,
    )

    driver.execute_query("CALL db.awaitIndexes(60)", database_=database)
