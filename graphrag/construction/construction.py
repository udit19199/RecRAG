from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from neo4j import Driver
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.llm import LLMBase

from . import default_extraction, ontology_guided


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
            approach = default_extraction
        case ConstructionMethod.ONTOLOGY_GUIDED:
            approach = ontology_guided
        case _:
            raise ValueError(f"Unsupported construction method: {method}")

    page_texts = []
    for page in pages:
        passages_text = "\n\n".join(page.passages)
        page_texts.append(f"Page: {page.title}\n{passages_text}")
    text = "\n\n".join(page_texts)

    driver.execute_query(
        "CREATE DATABASE $name IF NOT EXISTS WAIT", name=database, database_="system"
    )
    pipeline = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=approach.SCHEMA,
        prompt_template=approach.EXTRACTION_PROMPT,
        from_file=False,
        text_splitter=FixedSizeSplitter(chunk_size=1000, chunk_overlap=100),
        neo4j_database=database,
    )
    asyncio.run(pipeline.run_async(text=text))

    create_vector_index(
        driver,
        "chunk_embeddings",
        label="Chunk",
        embedding_property="embedding",
        dimensions=embedding_dimensions,
        similarity_fn="cosine",
        neo4j_database=database,
    )

    create_fulltext_index(
        driver,
        "chunk_fulltext",
        label="Chunk",
        node_properties=["text"],
        neo4j_database=database,
    )

    driver.execute_query("CALL db.awaitIndexes(60)", database_=database)
