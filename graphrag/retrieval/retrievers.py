from __future__ import annotations

import json

import neo4j
from neo4j import Driver
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

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

ENTITY_VECTOR_RETRIEVAL_QUERY = """
MATCH (entity:__Entity__)-[:HAS_EMBEDDING]->(node)
RETURN node.text AS text, [entity.name] AS entities, [] AS graph_facts, score
"""


def format_retrieval_record(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(
        content=json.dumps(record.data(), ensure_ascii=False, default=str)
    )


def build_vector_retriever(
    *, driver: Driver, embedder: Embedder, database: str
) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        driver=driver,
        index_name="chunk_embeddings",
        retrieval_query=VECTOR_RETRIEVAL_QUERY,
        result_formatter=format_retrieval_record,
        embedder=embedder,
        neo4j_database=database,
    )


def build_entity_vector_retriever(
    *, driver: Driver, embedder: Embedder, database: str
) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        driver=driver,
        index_name="entity_embeddings",
        retrieval_query=ENTITY_VECTOR_RETRIEVAL_QUERY,
        result_formatter=format_retrieval_record,
        embedder=embedder,
        neo4j_database=database,
    )
