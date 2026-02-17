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


class IngestionPipeline:
    """Pipeline for ingesting documents and creating embeddings."""

    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path

        embedder_config = config.get("embedding", {})
        embedder_provider = embedder_config.get("provider", "openai")
        embedder_model = embedder_config.get("model", "text-embedding-3-small")

        self.embedder: BaseEmbedder = create_embedder(
            embedder_provider,
            model=embedder_model,
            **{
                k: v
                for k, v in embedder_config.items()
                if k not in ("provider", "model")
            },
        )

        chunk_size = get_config_value(config, "ingestion.chunk_size", 1024)
        chunk_overlap = get_config_value(config, "ingestion.chunk_overlap", 50)

        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        storage_dir = resolve_path(
            config.get("storage", {}).get("directory", "storage"), config_path
        )
        embedder_id = embedder_model.replace("/", "_").replace("-", "_")
        index_path = storage_dir / f"faiss_{embedder_id}.index"
        metadata_path = storage_dir / f"faiss_{embedder_id}.json"

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

        logger.info("Generating embeddings...")
        texts = [chunk for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")

        logger.info("Adding to vector store...")
        metadatas = [
            {"source": doc.metadata.get("file_name", "unknown")} for doc in documents
        ]
        self.vector_store.add(embeddings, texts, metadatas)

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

        embedder_config = config.get("embedding", {})
        embedder_provider = embedder_config.get("provider", "openai")
        embedder_model = embedder_config.get("model", "text-embedding-3-small")

        self.embedder: BaseEmbedder = create_embedder(
            embedder_provider,
            model=embedder_model,
            **{
                k: v
                for k, v in embedder_config.items()
                if k not in ("provider", "model")
            },
        )

        llm_config = config.get("llm", {})
        llm_provider = llm_config.get("provider", "openai")
        llm_model = llm_config.get("model", "gpt-4o-mini")

        self.llm: BaseLLM = create_llm(
            llm_provider,
            model=llm_model,
            **{k: v for k, v in llm_config.items() if k not in ("provider", "model")},
        )

        storage_dir = resolve_path(
            config.get("storage", {}).get("directory", "storage"), config_path
        )
        embedder_id = embedder_model.replace("/", "_").replace("-", "_")
        index_path = storage_dir / f"faiss_{embedder_id}.index"
        metadata_path = storage_dir / f"faiss_{embedder_id}.json"

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
