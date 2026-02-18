import logging
from pathlib import Path
from typing import Any, Optional

from adapters import BaseEmbedder, BaseLLM, create_embedder, create_llm
from config import get_config_value, load_config, resolve_path
from core import VectorStore, TextSplitter, DocumentLoader


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


DEFAULT_CONTEXT_TEMPLATE = """Context information:
{context}

Question: {question}

Answer:"""


def create_embedder_from_config(config: dict[str, Any]) -> BaseEmbedder:
    """Create an embedder instance from configuration.

    Args:
        config: Configuration dictionary containing 'embedding' section.

    Returns:
        BaseEmbedder instance.
    """
    embedder_config = config.get("embedding", {})
    provider = embedder_config.get("provider", "openai")
    model = embedder_config.get("model", "text-embedding-3-small")

    extra_kwargs = {
        k: v for k, v in embedder_config.items() if k not in ("provider", "model")
    }

    return create_embedder(provider, model=model, **extra_kwargs)


def create_llm_from_config(config: dict[str, Any]) -> BaseLLM:
    """Create an LLM instance from configuration.

    Args:
        config: Configuration dictionary containing 'llm' section.

    Returns:
        BaseLLM instance.
    """
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "openai")
    model = llm_config.get("model", "gpt-4o-mini")

    extra_kwargs = {
        k: v for k, v in llm_config.items() if k not in ("provider", "model")
    }

    return create_llm(provider, model=model, **extra_kwargs)


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
    """Pipeline for ingesting documents and creating embeddings."""

    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path

        self.embedder = create_embedder_from_config(config)

        chunk_size = get_config_value(config, "ingestion.chunk_size", 1024)
        chunk_overlap = get_config_value(config, "ingestion.chunk_overlap", 50)
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        index_path, metadata_path = get_vector_store_paths(
            config, config_path, self.embedder.model
        )
        self.vector_store = VectorStore(
            dimension=self.embedder.dimension,
            index_path=index_path,
            metadata_path=metadata_path,
        )

        ingestion_dir = resolve_path(
            config.get("ingestion", {}).get("directory", "data/pdfs"), config_path
        )
        self.loader = DocumentLoader(str(ingestion_dir))

    def run(self, force: bool = False) -> dict[str, Any]:
        if force:
            logger.info("Force re-indexing - clearing existing index")
            self.vector_store.delete_all()

        logger.info("Loading documents...")
        documents = self.loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        logger.info("Splitting documents into chunks...")
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        doc_source_map = {}
        for doc in documents:
            source = doc.metadata.get("file_name", "unknown")
            doc_text = doc.text if hasattr(doc, "text") else str(doc)
            doc_source_map[doc_text] = source

        chunk_metadatas = []
        for chunk in chunks:
            source = "unknown"
            for doc_text, doc_source in doc_source_map.items():
                if chunk in doc_text or doc_text in chunk:
                    source = doc_source
                    break
            chunk_metadatas.append({"source": source})

        logger.info("Generating embeddings...")
        embeddings = self.embedder.embed_batch(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        logger.info("Adding to vector store...")
        self.vector_store.add(embeddings, chunks, chunk_metadatas)

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "total_vectors": self.vector_store.count,
        }


def run_ingestion(
    config_path: Path = Path("config.toml"),
    force: bool = False,
) -> dict[str, Any]:
    """Run the ingestion pipeline.

    Args:
        config_path: Path to configuration file.
        force: If True, re-index all documents.

    Returns:
        Dictionary with ingestion results.
    """
    config = load_config(config_path)
    pipeline = IngestionPipeline(config, config_path)
    return pipeline.run(force=force)


class RetrievalPipeline:
    """Pipeline for retrieving and generating responses."""

    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path

        self.embedder = create_embedder_from_config(config)
        self.llm = create_llm_from_config(config)

        index_path, metadata_path = get_vector_store_paths(
            config, config_path, self.embedder.model
        )
        self.vector_store = VectorStore(
            dimension=self.embedder.dimension,
            index_path=index_path,
            metadata_path=metadata_path,
        )

        self.top_k = get_config_value(config, "retrieval.top_k", 4)
        self.context_template = config.get("retrieval", {}).get(
            "context_template", DEFAULT_CONTEXT_TEMPLATE
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
        k = top_k or self.top_k

        logger.info(f"Embedding query: {query[:50]}...")
        query_embedding = self.embedder.embed(query)

        logger.info(f"Searching top {k} results...")
        _, results = self.vector_store.search(query_embedding, k=k)

        return results

    def generate(
        self,
        query: str,
        context: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        if context is None:
            context = self.retrieve(query)

        context_text = "\n\n".join([doc.get("text", "") for doc in context])

        prompt = self.context_template.format(
            context=context_text,
            question=query,
        )

        logger.info("Generating response...")
        response = self.llm.generate(prompt)

        return response

    def query(self, query: str) -> dict[str, Any]:
        context = self.retrieve(query)
        response = self.generate(query, context)

        return {
            "response": response,
            "context": context,
        }


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
    return RetrievalPipeline(config, config_path)
