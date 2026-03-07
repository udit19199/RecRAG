import logging
from pathlib import Path
from typing import Any

from adapters import BaseEmbedder
from config import get_config_value, get_ingestion_dir, get_storage_dir
from loaders import BaseDocumentLoader, DocumentLoader
from models.chunk import Chunk
from splitters import BaseTextSplitter, TextSplitter
from stores import BaseVectorStore, VectorStore
from .base import (
    DEFAULT_BATCH_SIZE,
    create_embedder_from_config,
    get_collection_name,
    get_milvus_uri,
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting documents and creating embeddings.

    Supports dependency injection and full-corpus batch ingestion.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        splitter: BaseTextSplitter,
        loader: BaseDocumentLoader,
        vector_store: BaseVectorStore,
        config: dict[str, Any] | None = None,
        config_path: Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.embedder = embedder
        self.splitter = splitter
        self.loader = loader
        self.vector_store = vector_store
        self.config = config or {}
        self.config_path = config_path
        self.batch_size = batch_size

    @classmethod
    def from_config(
        cls, config: dict[str, Any], config_path: Path
    ) -> "IngestionPipeline":
        """Create pipeline from configuration dictionary."""
        embedder = create_embedder_from_config(config)

        chunk_size = get_config_value(config, "ingestion.chunk_size", 1024)
        chunk_overlap = get_config_value(config, "ingestion.chunk_overlap", 50)
        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        collection_name = get_collection_name(config, embedder.model)
        uri = get_milvus_uri(config, config_path)
        vector_store = VectorStore(
            dimension=embedder.dimension,
            collection_name=collection_name,
            uri=uri,
            metric_type=config.get("storage", {}).get("metric_type", "L2"),
        )

        ingestion_dir = get_ingestion_dir(config, config_path)
        loader = DocumentLoader(str(ingestion_dir))
        batch_size = get_config_value(
            config, "ingestion.batch_size", DEFAULT_BATCH_SIZE
        )

        return cls(
            embedder=embedder,
            splitter=splitter,
            loader=loader,
            vector_store=vector_store,
            config=config,
            config_path=config_path,
            batch_size=batch_size,
        )

    # ── File discovery ────────────────────────────────────────────────────────

    def _discover_files(self, directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(directory.glob("*.pdf"))

    # ── Core processing ───────────────────────────────────────────────────────

    def _embed_and_store(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        embeddings = self.embedder.embed_batch([c.text for c in chunks])
        self.vector_store.add(
            embeddings, [c.text for c in chunks], [{"source": c.source} for c in chunks]
        )
        return len(embeddings)

    def _process_file_batch(self, batch: list[Path], batch_num: int) -> tuple[int, int]:
        documents = []
        for fp in batch:
            try:
                documents.extend(self.loader.load_file(fp))
            except Exception as e:
                logger.warning("Failed to load %s: %s", fp, e)
        if not documents:
            return 0, 0
        chunks = self.splitter.split_documents(documents)
        embeddings_count = self._embed_and_store(chunks)
        logger.info("Processed batch %d: %d chunks", batch_num, len(chunks))
        return len(chunks), embeddings_count

    def _process_files_in_batches(self, files: list[Path]) -> tuple[int, int]:
        total_chunks = total_embeddings = 0
        for i in range(0, len(files), self.batch_size):
            batch = files[i : i + self.batch_size]
            chunks, embeddings = self._process_file_batch(
                batch, i // self.batch_size + 1
            )
            total_chunks += chunks
            total_embeddings += embeddings
        return total_chunks, total_embeddings

    def _prepare_for_ingestion(self, force: bool) -> None:
        if force:
            logger.info("Force re-indexing — clearing existing index")
            self.vector_store.delete_all()

    # ── Public API ────────────────────────────────────────────────────────────

    def process_documents_streaming(self, force: bool = False) -> dict[str, Any]:
        """Process all documents in the ingestion directory, optionally forcing re-index."""
        self._prepare_for_ingestion(force)
        ingestion_dir = get_ingestion_dir(self.config, self.config_path or Path("."))
        all_files = self._discover_files(ingestion_dir)

        if not all_files:
            logger.info("No files to process")
            return {
                "documents": 0,
                "chunks": 0,
                "embeddings": 0,
                "total_vectors": self.vector_store.count,
            }

        total_chunks, total_embeddings = self._process_files_in_batches(all_files)
        return {
            "documents": len(all_files),
            "chunks": total_chunks,
            "embeddings": total_embeddings,
            "total_vectors": self.vector_store.count,
        }

    def process_all_documents(self, force: bool = False) -> dict[str, Any]:
        """Load all documents at once; for large corpora use process_documents_streaming."""
        self._prepare_for_ingestion(force)
        documents = self.loader.load()
        chunks = self.splitter.split_documents(documents)
        embeddings_count = self._embed_and_store(chunks)
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "embeddings": embeddings_count,
            "total_vectors": self.vector_store.count,
        }


def run_ingestion(
    config_path: Path = Path("config.toml"),
    force: bool = False,
) -> dict[str, Any]:
    from config import load_config

    config = load_config(config_path)
    pipeline = IngestionPipeline.from_config(config, config_path)
    return pipeline.process_documents_streaming(force=force)
