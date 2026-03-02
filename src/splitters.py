"""Text splitters for RecRAG."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument

if TYPE_CHECKING:
    from models.chunk import Chunk


class BaseTextSplitter(ABC):
    @abstractmethod
    def split_documents(self, documents: list[LlamaDocument]) -> list["Chunk"]:
        pass

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        pass


class SentenceTextSplitter(BaseTextSplitter):
    """Chunks documents while preserving metadata via llama-index SentenceSplitter.

    Preserves ``file_name`` metadata so source lookup is O(1) per chunk.
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents: list[LlamaDocument]) -> list["Chunk"]:
        from models.chunk import Chunk

        chunks = []
        for node in self.splitter.get_nodes_from_documents(documents):
            metadata = dict(node.metadata) if node.metadata else {}
            chunks.append(
                Chunk(
                    text=node.get_content(),
                    source=metadata.get("file_name", "unknown"),
                    metadata=metadata,
                )
            )
        return chunks

    def split_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)


# Default alias
TextSplitter = SentenceTextSplitter

__all__ = ["BaseTextSplitter", "SentenceTextSplitter", "TextSplitter"]
