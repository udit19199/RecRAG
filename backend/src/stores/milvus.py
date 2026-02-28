import logging
from typing import Any, Optional

from pymilvus import DataType, MilvusClient

from models.chunk import RetrievalResult

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    """Milvus-backed vector store.

    Supports both Milvus Lite (local embedded file) and Milvus Standalone /
    Distributed (remote server) via a single URI parameter:

    - Lite:   uri="./storage/milvus_lite.db"
    - Server: uri="http://milvus:19530"

    All metadata (text, source, extra fields) is stored as Milvus payload
    fields — no external JSON files required.
    """

    def __init__(
        self,
        dimension: int,
        collection_name: str = "recrag_default",
        uri: str = "./storage/milvus_lite.db",
        metric_type: str = "L2",
        index_params: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(dimension)
        self._collection_name = collection_name
        self._metric_type = metric_type
        self._index_params = index_params or {
            "index_type": "FLAT",
            "metric_type": metric_type,
            "params": {},
        }
        self._client = MilvusClient(uri=uri)
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create collection with schema + index if it does not already exist."""
        if self._client.has_collection(self._collection_name):
            return

        schema = self._client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=65_535)
        schema.add_field("source", DataType.VARCHAR, max_length=2_048)
        schema.add_field("extra_meta", DataType.JSON)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=self._index_params.get("index_type", "FLAT"),
            metric_type=self._metric_type,
            params=self._index_params.get("params", {}),
        )

        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Created Milvus collection '%s'", self._collection_name)

    @staticmethod
    def _build_source_filter(sources: set[str]) -> str:
        """Build a Milvus filter expression for a set of source paths."""
        escaped = ", ".join(f'"{s}"' for s in sources)
        return f"source in [{escaped}]"

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadata_list: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Upsert documents: remove existing vectors for the same sources,
        then insert the new ones.

        Args:
            embeddings: Dense vectors of shape (n, dimension).
            documents: Corresponding text content for each vector.
            metadata_list: Optional per-document metadata dicts.
                Each dict may contain a ``source`` key and arbitrary extras.
        """
        if metadata_list is None:
            metadata_list = [{} for _ in documents]

        # Remove stale vectors for sources that are being re-added.
        sources: set[str] = {
            str(m["source"]) for m in metadata_list if m.get("source")
        }
        if sources:
            self.delete_by_filter(self._build_source_filter(sources))

        rows = [
            {
                "vector": embedding,
                "text": doc,
                "source": str(meta.get("source", "unknown")),
                "extra_meta": {k: v for k, v in meta.items() if k != "source"},
            }
            for embedding, doc, meta in zip(embeddings, documents, metadata_list)
        ]

        if rows:
            self._client.insert(collection_name=self._collection_name, data=rows)

    def search(
        self,
        query_embedding: list[float],
        k: int = 4,
    ) -> tuple[list[float], list[RetrievalResult]]:
        """Return the top-k most similar documents.

        Args:
            query_embedding: Query vector of length ``dimension``.
            k: Number of results to return.

        Returns:
            Tuple of (distances, results) where distances are L2 distances
            (lower = more similar) and results contain text + metadata.
        """
        hits = self._client.search(
            collection_name=self._collection_name,
            data=[query_embedding],
            limit=k,
            output_fields=["text", "source", "extra_meta"],
        )

        distances: list[float] = []
        results: list[RetrievalResult] = []

        for hit in hits[0]:
            dist = float(hit.get("distance", 0.0))
            entity = hit.get("entity", {})
            distances.append(dist)
            results.append(
                RetrievalResult(
                    text=entity.get("text", ""),
                    source=entity.get("source", "unknown"),
                    distance=dist,
                    metadata=entity.get("extra_meta") or {},
                )
            )

        return distances, results

    def delete_all(self) -> None:
        """Drop the collection and recreate it (empty)."""
        self._client.drop_collection(self._collection_name)
        self._ensure_collection()

    def delete_by_filter(self, filter_expr: str) -> int:
        """Delete all entities matching a Milvus filter expression.

        Args:
            filter_expr: Milvus boolean expression, e.g. ``source in ["a.pdf"]``.

        Returns:
            Number of entities deleted (0 if unavailable from response).
        """
        result = self._client.delete(
            collection_name=self._collection_name,
            filter=filter_expr,
        )
        if isinstance(result, dict):
            return int(result.get("delete_count", 0))
        return 0

    @property
    def count(self) -> int:
        """Return the number of vectors currently stored in the collection."""
        stats = self._client.get_collection_stats(self._collection_name)
        return int(stats.get("row_count", 0))

    def save(self) -> None:
        """No-op — Milvus persists data server-side automatically."""

    def close(self) -> None:
        """Close the Milvus client connection."""
        self._client.close()
