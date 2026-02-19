import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from adapters import BaseEmbedder, BaseLLM, create_embedder, create_llm
from config import get_config_value, load_config, resolve_path
from core import (
    BaseDocumentLoader,
    BaseTextSplitter,
    BaseVectorStore,
    Chunk,
    DocumentLoader,
    TextSplitter,
    VectorStore,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TEMPLATE = """Context information:
{context}

Question: {question}

Answer:"""

DEFAULT_BATCH_SIZE = 100
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 4


def _compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _create_adapter_from_config(
    config: dict[str, Any],
    section: str,
    create_fn: Callable[..., Any],
    defaults: dict[str, str],
) -> Any:
    """Create an adapter (embedder or LLM) from configuration.

    Args:
        config: Configuration dictionary.
        section: Config section name ('embedding' or 'llm').
        create_fn: Factory function (create_embedder or create_llm).
        defaults: Default values for provider and model.

    Returns:
        Adapter instance.
    """
    section_config = config.get(section, {})
    provider = section_config.get("provider", defaults["provider"])
    model = section_config.get("model", defaults["model"])

    extra_kwargs = {
        k: v for k, v in section_config.items() if k not in ("provider", "model")
    }

    return create_fn(provider, model=model, **extra_kwargs)


def create_embedder_from_config(config: dict[str, Any]) -> BaseEmbedder:
    """Create an embedder instance from configuration."""
    defaults = {"provider": "openai", "model": "text-embedding-3-small"}
    return _create_adapter_from_config(config, "embedding", create_embedder, defaults)


def create_llm_from_config(config: dict[str, Any]) -> BaseLLM:
    """Create an LLM instance from configuration."""
    defaults = {"provider": "openai", "model": "gpt-4o-mini"}
    return _create_adapter_from_config(config, "llm", create_llm, defaults)


def get_vector_store_paths(
    config: dict[str, Any], config_path: Path, embedder_model: str
) -> tuple[Path, Path]:
    """Get index and metadata paths for the vector store.

    Args:
        config: Configuration dictionary.
        config_path: Path to configuration file.
        embedder_model: Embedder model name for path generation.

    Returns:
        Tuple of (index_path, metadata_path).
    """
    storage_dir = resolve_path(
        config.get("storage", {}).get("directory", "storage"), config_path
    )
    embedder_id = embedder_model.replace("/", "_").replace("-", "_")
    return (
        storage_dir / f"faiss_{embedder_id}.index",
        storage_dir / f"faiss_{embedder_id}.json",
    )


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

        ingestion_dir = resolve_path(
            config.get("ingestion", {}).get("directory", "data/pdfs"), config_path
        )
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
        chunk_metadatas = [{"source": c.source} for c in chunks]

        embeddings = self.embedder.embed_batch(chunk_texts)
        self.vector_store.add(embeddings, chunk_texts, chunk_metadatas)

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

    def run_incremental(
        self,
        files: list[Path] | None = None,
    ) -> dict[str, Any]:
        """Run incremental ingestion - only process new or changed files."""
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
        return resolve_path(
            self.config.get("storage", {}).get("directory", "storage"),
            self.config_path,
        )

    def _get_files_to_process(
        self, files: list[Path] | None
    ) -> list[tuple[Path, bool]]:
        """Determine which files need processing."""
        if files is not None:
            return [(f, str(f) not in self._processed_files) for f in files]

        ingestion_dir = resolve_path(
            self.config.get("ingestion", {}).get("directory", "data/pdfs"),
            self.config_path or Path("."),
        )
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

    def run_streaming(self, force: bool = False) -> dict[str, Any]:
        """Run ingestion with streaming/batched processing for lower memory usage."""
        self._prepare_for_ingestion(force)

        ingestion_dir = resolve_path(
            self.config.get("ingestion", {}).get("directory", "data/pdfs"),
            self.config_path or Path("."),
        )
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

    def run(self, force: bool = False) -> dict[str, Any]:
        """Run full ingestion - loads all documents at once.

        For large corpora, use run_streaming() instead.
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
    config = load_config(config_path)
    pipeline = IngestionPipeline.from_config(config, config_path)

    if incremental:
        return pipeline.run_incremental(files=files)
    else:
        return pipeline.run(force=force)


class RetrievalPipeline:
    """Pipeline for retrieving and generating responses.

    Supports dependency injection for flexible composition.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        llm: BaseLLM,
        vector_store: BaseVectorStore,
        top_k: int = DEFAULT_TOP_K,
        context_template: str = DEFAULT_CONTEXT_TEMPLATE,
        config: dict[str, Any] | None = None,
        config_path: Path | None = None,
    ):
        self.embedder = embedder
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k
        self.context_template = context_template
        self.config = config or {}
        self.config_path = config_path

    @classmethod
    def from_config(
        cls, config: dict[str, Any], config_path: Path
    ) -> "RetrievalPipeline":
        """Create pipeline from configuration dictionary."""
        embedder = create_embedder_from_config(config)
        llm = create_llm_from_config(config)

        index_path, metadata_path = get_vector_store_paths(
            config, config_path, embedder.model
        )
        vector_store = VectorStore(
            dimension=embedder.dimension,
            index_path=index_path,
            metadata_path=metadata_path,
        )

        top_k = get_config_value(config, "retrieval.top_k", DEFAULT_TOP_K)
        context_template = config.get("retrieval", {}).get(
            "context_template", DEFAULT_CONTEXT_TEMPLATE
        )

        return cls(
            embedder=embedder,
            llm=llm,
            vector_store=vector_store,
            top_k=top_k,
            context_template=context_template,
            config=config,
            config_path=config_path,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        k = top_k or self.top_k
        logger.info(f"Embedding query: {query[:50]}...")

        query_embedding = self.embedder.embed(query)
        _, results = self.vector_store.search(query_embedding, k=k)

        logger.info(f"Found {len(results)} results")
        return results

    def generate(
        self,
        query: str,
        context: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Generate a response using retrieved context."""
        context = context or self.retrieve(query)
        context_text = "\n\n".join(doc.get("text", "") for doc in context)

        prompt = self.context_template.format(
            context=context_text,
            question=query,
        )

        logger.info("Generating response...")
        return self.llm.generate(prompt)

    def query(self, query: str) -> dict[str, Any]:
        """Execute a full RAG query: retrieve and generate."""
        context = self.retrieve(query)
        response = self.generate(query, context)

        return {"response": response, "context": context}


def get_retrieval_pipeline(
    config_path: Path = Path("config.toml"),
) -> RetrievalPipeline:
    """Create a retrieval pipeline from config.

    Args:
        config_path: Path to configuration file.

    Returns:
        RetrievalPipeline instance.
    """
    config = load_config(config_path)
    return RetrievalPipeline.from_config(config, config_path)
