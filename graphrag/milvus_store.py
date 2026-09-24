from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


class MilvusStore:
    def __init__(self, uri: str, timeout: int) -> None:
        self._timeout = timeout
        self._alias = "recrag"
        connections.connect(alias=self._alias, uri=uri, timeout=timeout)

    def collection(self, name: str, dimensions: int) -> Collection:
        if not utility.has_collection(name, using=self._alias, timeout=self._timeout):
            schema = CollectionSchema(
                [
                    FieldSchema(
                        "chunk_id", DataType.VARCHAR, is_primary=True, max_length=512
                    ),
                    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dimensions),
                ]
            )
            collection = Collection(name, schema, using=self._alias)
            collection.create_index(
                "embedding",
                {"index_type": "FLAT", "metric_type": "COSINE", "params": {}},
            )
        collection = Collection(name, using=self._alias)
        collection.load(timeout=self._timeout)
        return collection

    def write(
        self,
        name: str,
        dimensions: int,
        chunk_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        collection = self.collection(name, dimensions)
        collection.upsert([chunk_ids, embeddings], timeout=self._timeout)
        collection.flush(timeout=self._timeout)

    def close(self) -> None:
        connections.disconnect(self._alias)
