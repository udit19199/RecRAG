import hashlib
import json
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


def _compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
        self._processed_files: dict[str, str] = {}

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

    # ── Tracking file helpers ─────────────────────────────────────────────────

    def _load_processed_files(self, storage_dir: Path) -> dict[str, str]:
        tracking_file = storage_dir / "processed_files.json"
        if tracking_file.exists():
            try:
                with open(tracking_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_processed_files(self, storage_dir: Path) -> None:
        tracking_file = storage_dir / "processed_files.json"
        tracking_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tracking_file, "w") as f:
            json.dump(self._processed_files, f, indent=2)

    def _get_storage_dir(self) -> Path | None:
        if not self.config_path:
            return None
        return get_storage_dir(self.config, self.config_path)

    # ── File discovery ────────────────────────────────────────────────────────

    def _discover_files(self, directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(directory.glob("*.pdf"))

    def _get_changed_files(
        self, directory: Path, processed: dict[str, str]
    ) -> list[tuple[Path, bool]]:
        """Return (file_path, is_new) tuples for new or modified files."""
        results = []
        for fp in self._discover_files(directory):
            key = str(fp)
            is_new = key not in processed
            if is_new or processed[key] != _compute_file_hash(fp):
                results.append((fp, is_new))
        return results

    def _get_files_to_process(
        self, files: list[Path] | None
    ) -> list[tuple[Path, bool]]:
        if files is not None:
            return [(f, str(f) not in self._processed_files) for f in files]
        ingestion_dir = get_ingestion_dir(self.config, self.config_path or Path("."))
        return self._get_changed_files(ingestion_dir, self._processed_files)

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
            for fp in batch:
                self._processed_files[str(fp)] = _compute_file_hash(fp)
        return total_chunks, total_embeddings

    def _prepare_for_ingestion(self, force: bool) -> None:
        if force:
            logger.info("Force re-indexing — clearing existing index")
            self.vector_store.delete_all()
            self._processed_files = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def process_new_and_changed_documents(
        self, files: list[Path] | None = None
    ) -> dict[str, Any]:
        """Incremental ingestion — only process new or changed files."""
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
            "Processing %d new, %d updated files", len(new_files), len(updated_files)
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
    incremental: bool = False,
    files: list[Path] | None = None,
) -> dict[str, Any]:
    from config import load_config

    config = load_config(config_path)
    pipeline = IngestionPipeline.from_config(config, config_path)

    if incremental:
        return pipeline.process_new_and_changed_documents(files=files)
    return pipeline.process_all_documents(force=force)
