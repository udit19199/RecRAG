from __future__ import annotations

from typing import Literal

from neo4j import Driver
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.generation import GraphRAG as Neo4jGraphRAG
from neo4j_graphrag.llm import LLMBase
from neo4j_graphrag.schema import get_schema

from ..milvus_store import MilvusStore
from .agentic import build_agentic_retriever
from .retrievers import (
    RETRIEVAL_TOP_K,
    MilvusRetriever,
    build_entity_vector_retriever,
    build_vector_retriever,
)

NO_CONTEXT = "I could not find supporting context for this question."
RetrievalMethod = Literal["agentic", "vector", "entity_vector", "milvus_vector"]
RETRIEVAL_METHODS: list[RetrievalMethod] = [
    "agentic",
    "vector",
    "entity_vector",
    "milvus_vector",
]


def answer_question(
    question: str,
    *,
    driver: Driver,
    database: str,
    llm: LLMBase,
    embedder: Embedder,
    answer_llm,
    retrieval_methods: list[RetrievalMethod],
    milvus: MilvusStore,
    embedding_dimensions: int,
):
    neo4j_schema = None
    if "agentic" in retrieval_methods:
        neo4j_schema = get_schema(driver, database=database)
    results = []
    for method in retrieval_methods:
        if method == "vector":
            retriever = build_vector_retriever(
                driver=driver, embedder=embedder, database=database
            )
        elif method == "entity_vector":
            retriever = build_entity_vector_retriever(
                driver=driver, embedder=embedder, database=database
            )
        elif method == "milvus_vector":
            retriever = MilvusRetriever(
                driver=driver,
                embedder=embedder,
                database=database,
                store=milvus,
                dimensions=embedding_dimensions,
            )
        elif method == "agentic":
            retriever = build_agentic_retriever(
                driver=driver,
                llm=llm,
                embedder=embedder,
                database=database,
                agent_llm=answer_llm,
                neo4j_schema=neo4j_schema,
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        results.append(
            Neo4jGraphRAG(retriever, llm).search(
                question,
                retriever_config={"top_k": RETRIEVAL_TOP_K},
                return_context=True,
                response_fallback=NO_CONTEXT,
            )
        )
    return results
