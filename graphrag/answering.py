from __future__ import annotations

from typing import Literal

from neo4j import Driver
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.generation import GraphRAG as Neo4jGraphRAG
from neo4j_graphrag.llm import LLMBase
from neo4j_graphrag.schema import get_schema

from .agentic import build_agentic_retriever
from .retrievers import RETRIEVAL_TOP_K

NO_CONTEXT = "I could not find supporting context for this question."
RetrievalMethod = Literal["agentic"]
RETRIEVAL_METHODS: list[RetrievalMethod] = ["agentic"]


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
    if any(method != "agentic" for method in retrieval_methods):
        raise ValueError("Only agentic retrieval is supported.")
    results = []
    if not retrieval_methods:
        return results
    neo4j_schema = get_schema(driver, database=database)
    retriever = build_agentic_retriever(
        driver=driver,
        llm=llm,
        embedder=embedder,
        database=database,
        agent_llm=answer_llm,
        neo4j_schema=neo4j_schema,
    )
    for method in retrieval_methods:
        results.append(
            Neo4jGraphRAG(retriever, llm).search(
                question,
                retriever_config={"top_k": RETRIEVAL_TOP_K},
                return_context=True,
                response_fallback=NO_CONTEXT,
            )
        )
    return results
