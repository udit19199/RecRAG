from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from neo4j import Driver
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import LLMBase

from .ontology import ONTOLOGY_EXTRACTION_PROMPT, ONTOLOGY_SCHEMA


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
