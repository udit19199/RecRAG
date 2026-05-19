"""Vector store implementations for RecRAG."""

import logging
import os
import tempfile
import urllib.parse
from typing import Any, Optional

from pymilvus import DataType, MilvusClient

from models.chunk import RetrievalResult

logger = logging.getLogger(__name__)

# Track temp directories for cleanup on shutdown
_DEV_TEMP_DIRS: list[str] = []

# Increase gRPC keepalive interval to avoid "too_many_pings" errors from Milvus Lite
os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "120000")   # 120s (default 10s)
os.environ.setdefault("GRPC_KEEPALIVE_TIMEOUT_MS", "20000") # 20s
os.environ.setdefault("GRPC_HTTP2_MIN_TIME_BETWEEN_PINGS_MS", "120000")
os.environ.setdefault("GRPC_HTTP2_MAX_PINGS_WITHOUT_DATA", "0")
DEV_MODE = os.environ.get("RECRAG_DEV", "").lower() in ("1", "true", "yes")


def _get_milvus_uri(uri: str) -> str:
    """If dev mode is active, redirect to a temporary directory."""
    if not DEV_MODE:
        return uri
    # Check if this is a file-based URI (Milvus Lite lite mode)
    if uri.endswith(".db") or "/" in uri and not uri.startswith("http"):
        tmp_dir = tempfile.mkdtemp(prefix="recrag_milvus_")
        _DEV_TEMP_DIRS.append(tmp_dir)
        db_path = os.path.join(tmp_dir, "milvus_lite.db")
        logger.info("DEV MODE: Using in-memory Milvus Lite at %s", db_path)
        return db_path
    return uri


def cleanup_dev_temp_dirs() -> None:
    """Remove all temp directories created during dev mode.

    Call this during application shutdown.
    """
    import shutil

    for d in _DEV_TEMP_DIRS:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    _DEV_TEMP_DIRS.clear()


class VectorStore:
    """Milvus-backed vector store (standalone or distributed server)."""

    def __init__(
        self,
        dimension: int,
        collection_name: str = "recrag_default",
        uri: str = "http://localhost:19530",
        metric_type: str = "L2",
        index_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self.dimension = dimension
        self._collection_name = collection_name
        self._metric_type = metric_type
        self._index_params = index_params or {
            "index_type": "FLAT",
            "metric_type": metric_type,
            "params": {},
        }

        resolved_uri = _get_milvus_uri(uri)

        env_user = os.environ.get("MILVUS_USERNAME")
        env_pass = os.environ.get("MILVUS_PASSWORD")

        if env_user and env_pass:
            self._client = MilvusClient(uri=resolved_uri, token=f"{env_user}:{env_pass}")
        else:
            parsed = urllib.parse.urlparse(resolved_uri)
            if parsed.username and parsed.password:
                token = f"{urllib.parse.unquote(parsed.username)}:{urllib.parse.unquote(parsed.password)}"
                netloc: str = parsed.hostname or ""
                if parsed.port:
                    netloc = f"{netloc}:{parsed.port}"
                cleaned_uri = urllib.parse.urlunparse(parsed._replace(netloc=netloc))
                self._client = MilvusClient(uri=cleaned_uri, token=token)
            else:
                self._client = MilvusClient(uri=resolved_uri)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        exists = self._client.has_collection(self._collection_name)

        if not exists:
            schema = self._client.create_schema(auto_id=True, enable_dynamic_field=False)
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

        # Load the collection into memory so search works (required for Milvus Lite).
        self._client.load_collection(self._collection_name)

    @staticmethod
    def _build_source_filter(sources: set[str]) -> str:
        escaped = ", ".join(f'"{s}"' for s in sources)
        return f"source in [{escaped}]"

    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadata_list: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        if metadata_list is None:
            metadata_list = [{} for _ in documents]

        sources: set[str] = {str(m["source"]) for m in metadata_list if m.get("source")}
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
        """Drop and recreate the collection (empties it)."""
        self._client.drop_collection(self._collection_name)
        self._ensure_collection()

    def delete_by_filter(self, filter_expr: str) -> int:
        result = self._client.delete(
            collection_name=self._collection_name, filter=filter_expr
        )
        return int(result.get("delete_count", 0)) if isinstance(result, dict) else 0

    def delete_document(self, source: str) -> int:
        """Delete all vectors associated with a given document source.

        Args:
            source: The document source (filename) to delete.

        Returns:
            Number of vectors deleted.
        """
        return self.delete_by_filter(f'source == "{source}"')

    @property
    def count(self) -> int:
        stats = self._client.get_collection_stats(self._collection_name)
        return int(stats.get("row_count", 0))

    def save(self) -> None:
        """No-op — Milvus persists server-side automatically."""

    def close(self) -> None:
        self._client.close()


__all__ = ["VectorStore"]
