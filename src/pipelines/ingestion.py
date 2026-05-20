import logging
from pathlib import Path
from typing import Any

from adapters import BaseEmbedder
from config import get_config_value, get_ingestion_dir
from loaders import (
    BaseDocumentLoader,
    DocumentLoader,
    LiteparseLoader,
    VisionPDFLoader,
)
from models.api import ExtractionMode
from models.chunk import Chunk
from splitters import BaseTextSplitter, TextSplitter
from stores import VectorStore
from .base import (
    DEFAULT_BATCH_SIZE,
    create_embedder_from_config,
    create_vector_store_from_config,
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
        vector_store: VectorStore,
        config: dict[str, Any] | None = None,
        config_path: Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        extraction_mode: ExtractionMode = ExtractionMode.TEXT_ONLY,
        vision_provider: str | None = None,
        vision_model: str | None = None,
    ):
        self.embedder = embedder
        self.splitter = splitter
        self.loader = loader
        self.vector_store = vector_store
        self.config = config or {}
        self.config_path = config_path
        self.batch_size = batch_size
        self.extraction_mode = extraction_mode
        self.vision_provider = vision_provider
        self.vision_model = vision_model

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        config_path: Path,
        extraction_mode: ExtractionMode = ExtractionMode.TEXT_ONLY,
        vision_provider: str | None = None,
        vision_model: str | None = None,
    ) -> "IngestionPipeline":
        """Create pipeline from configuration dictionary."""
        embedder = create_embedder_from_config(config)

        chunk_size = get_config_value(config, "ingestion.chunk_size", 1024)
        chunk_overlap = get_config_value(config, "ingestion.chunk_overlap", 50)
        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Get vision config from config or use provided overrides
        vision_cfg = config.get("vision", {})
        provider = vision_provider or vision_cfg.get("provider", "openai")
        model = vision_model or vision_cfg.get("model", "gpt-4o-mini")

        vector_store = create_vector_store_from_config(
            config,
            config_path,
            embedder,
            model if extraction_mode == ExtractionMode.VISION_ASSISTED else None,
        )

        ingestion_dir = get_ingestion_dir(config, config_path)

        # Select loader based on extraction mode
        loader: BaseDocumentLoader
        if extraction_mode == ExtractionMode.VISION_ASSISTED:
            vision_kwargs = {
                k: v for k, v in vision_cfg.items() if k not in ("provider", "model")
            }
            loader = VisionPDFLoader(
                str(ingestion_dir),
                vision_provider=provider,
                vision_model=model,
                vision_kwargs=vision_kwargs,
            )
            logger.info("Using vision-assisted extraction with %s/%s", provider, model)
        else:
            from stores import DEV_MODE

            if DEV_MODE:
                try:
                    loader = LiteparseLoader(str(ingestion_dir))
                    logger.info("DEV MODE: Using LiteParse for document extraction")
                except Exception as exc:
                    logger.warning(
                        "DEV MODE: LiteParse unavailable (%s), falling back to PDFLoader",
                        exc,
                    )
                    loader = DocumentLoader(str(ingestion_dir))
            else:
                loader = DocumentLoader(str(ingestion_dir))
                logger.info("Using text-only extraction")

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
            extraction_mode=extraction_mode,
            vision_provider=vision_provider,
            vision_model=vision_model,
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
                "extraction_mode": self.extraction_mode.value,
            }

        total_chunks, total_embeddings = self._process_files_in_batches(all_files)
        return {
            "documents": len(all_files),
            "chunks": total_chunks,
            "embeddings": total_embeddings,
            "total_vectors": self.vector_store.count,
            "extraction_mode": self.extraction_mode.value,
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
            "extraction_mode": self.extraction_mode.value,
        }


def run_ingestion(
    config_path: Path = Path("config.toml"),
    force: bool = False,
    extraction_mode: ExtractionMode = ExtractionMode.TEXT_ONLY,
    vision_provider: str | None = None,
    vision_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    from config import load_config

    config = load_config(config_path)

    if embedding_provider and embedding_model:
        config.setdefault("embedding", {})["provider"] = embedding_provider
        config.setdefault("embedding", {})["model"] = embedding_model

    pipeline = IngestionPipeline.from_config(
        config,
        config_path,
        extraction_mode=extraction_mode,
        vision_provider=vision_provider,
        vision_model=vision_model,
    )
    return pipeline.process_documents_streaming(force=force)
