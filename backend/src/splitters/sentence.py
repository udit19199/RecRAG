from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument
from .base import BaseTextSplitter

class SentenceTextSplitter(BaseTextSplitter):
    """Text splitter for chunking documents while preserving metadata.

    Uses llama-index's SentenceSplitter internally, which preserves
    document metadata in each resulting node. This allows O(1) source
    lookup instead of O(n×m) substring matching.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: list[LlamaDocument]) -> list["Chunk"]:
        """Split documents into chunks with preserved metadata."""
        from models.chunk import Chunk

        nodes = self.splitter.get_nodes_from_documents(documents)
        chunks = []

        for node in nodes:
            metadata = dict(node.metadata) if node.metadata else {}
            source = metadata.get("file_name", "unknown")

            chunks.append(
                Chunk(
                    text=node.get_content(),
                    source=source,
                    metadata=metadata,
                )
            )

        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split raw text into chunks without metadata."""
        return self.splitter.split_text(text)

# Backward compatibility alias
TextSplitter = SentenceTextSplitter
