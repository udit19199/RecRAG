from __future__ import annotations

from typing import Literal

from neo4j import Driver
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import LLMBase
from neo4j_graphrag.schema import get_schema

from .agentic import build_agentic_retriever
from .retrievers import (
    build_hybrid_retriever,
    build_text2cypher_retriever,
    build_vector_retriever,
)

NO_CONTEXT = "I could not find supporting context for this question."
RetrievalMethod = Literal[
    "text2cypher", "agentic", "vector", "hybrid"
]
RETRIEVAL_METHODS: list[RetrievalMethod] = [
    "text2cypher",
    "agentic",
    "vector",
    "hybrid",
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
):
    neo4j_schema = None
    if "text2cypher" in retrieval_methods or "agentic" in retrieval_methods:
        neo4j_schema = get_schema(driver, database=database)
    results = []
    for method in retrieval_methods:
        if method == "text2cypher":
            retriever = build_text2cypher_retriever(
                driver=driver,
                llm=llm,
                neo4j_schema=neo4j_schema,
                database=database,
            )
        elif method == "vector":
            retriever = build_vector_retriever(
                driver=driver, embedder=embedder, database=database
            )
        elif method == "hybrid":
            retriever = build_hybrid_retriever(
                driver=driver, embedder=embedder, database=database
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
            GraphRAG(retriever, llm).search(
                question, return_context=True, response_fallback=NO_CONTEXT
            )
        )
    return results
