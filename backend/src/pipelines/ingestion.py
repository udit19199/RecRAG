import json
import logging
from pathlib import Path
from typing import Any

from adapters import BaseEmbedder
from config import get_config_value, get_ingestion_dir, get_storage_dir
from loaders import BaseDocumentLoader, DocumentLoader
from models import Chunk
from splitters import BaseTextSplitter, TextSplitter
from stores import BaseVectorStore, VectorStore
from .base import (
    DEFAULT_BATCH_SIZE,
    create_embedder_from_config,
    get_vector_store_paths,
)
from .utils import _compute_file_hash

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting documents and creating embeddings.

    Supports dependency injection, incremental indexing, and batch streaming.
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

        # Initialize incremental indexing tracking
        self._processed_files: dict[str, str] = {}  # filepath -> hash

    @classmethod
    def from_config(
        cls, config: dict[str, Any], config_path: Path
    ) -> "IngestionPipeline":
        """Create pipeline from configuration dictionary."""
        embedder = create_embedder_from_config(config)

        chunk_size = get_config_value(config, "ingestion.chunk_size", 1024)
        chunk_overlap = get_config_value(config, "ingestion.chunk_overlap", 50)
        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        index_path, metadata_path = get_vector_store_paths(
            config, config_path, embedder.model
        )
        vector_store = VectorStore(
            dimension=embedder.dimension,
            index_path=index_path,
            metadata_path=metadata_path,
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

    def _load_processed_files(self, storage_dir: Path) -> dict[str, str]:
        """Load processed file tracking from storage."""
        tracking_file = storage_dir / "processed_files.json"
        if tracking_file.exists():
            try:
                with open(tracking_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_processed_files(self, storage_dir: Path) -> None:
        """Save processed file tracking to storage."""
        tracking_file = storage_dir / "processed_files.json"
        tracking_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tracking_file, "w") as f:
            json.dump(self._processed_files, f, indent=2)

    def _discover_files(self, directory: Path) -> list[Path]:
        """Discover all supported files in directory."""
        supported_extensions = {".pdf"}  # Can be extended
        files = []
        if directory.exists():
            for ext in supported_extensions:
                files.extend(directory.glob(f"*{ext}"))
        return sorted(files)

    def _get_changed_files(
        self, directory: Path, processed_files: dict[str, str]
    ) -> list[tuple[Path, bool]]:
        """Get list of files that are new or changed.

        Returns:
            List of (file_path, is_new) tuples
        """
        current_files = self._discover_files(directory)
        results = []

        for file_path in current_files:
            file_hash = _compute_file_hash(file_path)
            str_path = str(file_path)

            if str_path not in processed_files:
                # New file
                results.append((file_path, True))
            elif processed_files[str_path] != file_hash:
                # Changed file
                results.append((file_path, False))

        return results

    def _load_files_batch(self, file_paths: list[Path]) -> list[Any]:
        """Load documents from a batch of files."""
        documents = []
        for file_path in file_paths:
            try:
                docs = self.loader.load_file(file_path)
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        return documents

    def _embed_and_store(self, chunks: list[Chunk]) -> int:
        """Generate embeddings for chunks and store them. Returns number of embeddings."""
        if not chunks:
            return 0

        chunk_texts = [c.text for c in chunks]
        # Phase 5: Rename metadatas -> metadata_list
        metadata_list = [{"source": c.source} for c in chunks]

        embeddings = self.embedder.embed_batch(chunk_texts)
        self.vector_store.add(embeddings, chunk_texts, metadata_list)

        return len(embeddings)

    def _process_file_batch(self, batch: list[Path], batch_num: int) -> tuple[int, int]:
        """Process a batch of files: load, split, embed, store.

        Returns:
            Tuple of (chunks_count, embeddings_count).
        """
        documents = self._load_files_batch(batch)
        if not documents:
            return 0, 0

        chunks = self.splitter.split_documents(documents)
        embeddings_count = self._embed_and_store(chunks)

        logger.info(f"Processed batch {batch_num}: {len(chunks)} chunks")
        return len(chunks), embeddings_count

    def process_new_and_changed_documents(
        self,
        files: list[Path] | None = None,
    ) -> dict[str, Any]:
        """Run incremental ingestion - only process new or changed files.

        This method is memory-efficient as it processes files in batches.
        """
        storage_dir = self._get_storage_dir()
        if storage_dir:
            self._processed_files = self._load_processed_files(storage_dir)

        changed_files = self._get_files_to_process(files)

        if not changed_files:
            logger.info("No new or changed files to process")
            return {
                "documents": 0,
                "chunks": 0,
                "embeddings": 0,
                "total_vectors": self.vector_store.count,
                "skipped": 0,
            }

        new_files = [f for f, is_new in changed_files if is_new]
        updated_files = [f for f, is_new in changed_files if not is_new]
        all_files = new_files + updated_files

        logger.info(
            f"Processing {len(new_files)} new files, {len(updated_files)} updated files"
        )

        total_chunks, total_embeddings = self._process_files_in_batches(all_files)

        if storage_dir:
            self._save_processed_files(storage_dir)

        return {
            "documents": len(all_files),
            "new_documents": len(new_files),
            "updated_documents": len(updated_files),
            "chunks": total_chunks,
            "embeddings": total_embeddings,
            "total_vectors": self.vector_store.count,
        }

    def _get_storage_dir(self) -> Path | None:
        """Get storage directory from config."""
        if not self.config_path:
            return None
        return get_storage_dir(self.config, self.config_path)

    def _get_files_to_process(
        self, files: list[Path] | None
    ) -> list[tuple[Path, bool]]:
        """Determine which files need processing."""
        if files is not None:
            return [(f, str(f) not in self._processed_files) for f in files]

        ingestion_dir = get_ingestion_dir(self.config, self.config_path or Path("."))
        return self._get_changed_files(ingestion_dir, self._processed_files)

    def _process_files_in_batches(self, files: list[Path]) -> tuple[int, int]:
        """Process files in batches and update tracking.

        Returns:
            Tuple of (total_chunks, total_embeddings).
        """
        total_chunks = 0
        total_embeddings = 0

        for i in range(0, len(files), self.batch_size):
            batch = files[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            chunks_count, embeddings_count = self._process_file_batch(batch, batch_num)
            total_chunks += chunks_count
            total_embeddings += embeddings_count

            for file_path in batch:
                self._processed_files[str(file_path)] = _compute_file_hash(file_path)

        return total_chunks, total_embeddings

    def _prepare_for_ingestion(self, force: bool) -> None:
        """Clear index if force=True."""
        if force:
            logger.info("Force re-indexing - clearing existing index")
            self.vector_store.delete_all()
            self._processed_files = {}

    def process_documents_streaming(self, force: bool = False) -> dict[str, Any]:
        """Run ingestion with streaming/batched processing for lower memory usage.

        This method processes files in configured batch sizes.
        """
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
        """Run full ingestion - loads all documents at once into memory.

        For large corpora, use process_documents_streaming() instead.
        """
        self._prepare_for_ingestion(force)

        logger.info("Loading documents...")
        documents = self.loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        logger.info("Splitting documents into chunks...")
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info("Generating embeddings...")
        embeddings_count = self._embed_and_store(chunks)
        logger.info(f"Generated {embeddings_count} embeddings")

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "embeddings": embeddings_count,
            "total_vectors": self.vector_store.count,
        }

    # Backward compatibility
    def run(self, force: bool = False) -> dict[str, Any]:
        return self.process_all_documents(force=force)

    def run_streaming(self, force: bool = False) -> dict[str, Any]:
        return self.process_documents_streaming(force=force)

    def run_incremental(self, files: list[Path] | None = None) -> dict[str, Any]:
        return self.process_new_and_changed_documents(files=files)


def run_ingestion(
    config_path: Path = Path("config.toml"),
    force: bool = False,
    incremental: bool = False,
    files: list[Path] | None = None,
) -> dict[str, Any]:
    """Run the ingestion pipeline.

    Args:
        config_path: Path to configuration file.
        force: If True, re-index all documents.
        incremental: If True, only process new/changed files.
        files: Optional list of specific files to process.

    Returns:
        Dictionary with ingestion results.
    """
    from config import load_config

    config = load_config(config_path)
    pipeline = IngestionPipeline.from_config(config, config_path)

    if incremental:
        return pipeline.process_new_and_changed_documents(files=files)
    else:
        return pipeline.process_all_documents(force=force)
