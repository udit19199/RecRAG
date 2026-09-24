from __future__ import annotations

import json

import neo4j
from neo4j import Driver
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResultItem

from ..milvus_store import MilvusStore

RETRIEVAL_TOP_K = 5

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


class MilvusRetriever(Retriever):
    def __init__(
        self,
        *,
        driver: Driver,
        embedder: Embedder,
        database: str,
        store: MilvusStore,
        dimensions: int,
    ) -> None:
        super().__init__(driver, neo4j_database=database)
        self._embedder = embedder
        self._collection = store.collection(database.replace("-", "_"), dimensions)

    def get_search_results(
        self, query_text: str, top_k: int = 5, **kwargs
    ) -> RawSearchResult:
        hits = self._collection.search(
            [self._embedder.embed_query(query_text)],
            "embedding",
            {"metric_type": "COSINE", "params": {}},
            top_k,
            output_fields=[],
        )[0]
        chunk_ids = [hit.id for hit in hits]
        scores = [hit.score for hit in hits]
        records = self.driver.execute_query(
            "UNWIND range(0, size($chunk_ids) - 1) AS rank "
            "WITH rank, $chunk_ids[rank] AS chunk_id, $scores[rank] AS score "
            "MATCH (node:Chunk) WHERE elementId(node) = chunk_id "
            "CALL { WITH node MATCH (entity:__Entity__)-[:FROM_CHUNK]->(node) "
            "RETURN collect(DISTINCT entity.name) AS entities } "
            "CALL { WITH node MATCH (first:__Entity__)-[:FROM_CHUNK]->(node) "
            "MATCH path=(first)-[*1..2]-(last:__Entity__) "
            "WHERE all(r IN relationships(path) WHERE NOT type(r) IN "
            '["FROM_CHUNK", "NEXT_CHUNK", "FROM_DOCUMENT"]) '
            "WITH path LIMIT 25 RETURN collect(DISTINCT {nodes: "
            "[n IN nodes(path) | n.name], relationships: "
            "[r IN relationships(path) | {source: startNode(r).name, "
            "type: type(r), target: endNode(r).name}]}) AS graph_facts } "
            "RETURN node.text AS text, entities, graph_facts, score ORDER BY rank",
            chunk_ids=chunk_ids,
            scores=scores,
            database_=self.neo4j_database,
        ).records
        return RawSearchResult(records=records)

    def get_result_formatter(self):
        return format_retrieval_record
