"""Core components for RecRAG.

This module provides backward compatibility by re-exporting from focused submodules.
New code should import directly from the specific submodule (e.g., `from models import Chunk`).
"""

# Re-export for backward compatibility
from models import Chunk
from loaders import BaseDocumentLoader, DocumentLoader, PDFLoader
from splitters import BaseTextSplitter, SentenceTextSplitter, TextSplitter
from stores import BaseVectorStore, FAISSVectorStore, VectorStore

__all__ = [
    "Chunk",
    "BaseDocumentLoader",
    "DocumentLoader",
    "PDFLoader",
    "BaseTextSplitter",
    "TextSplitter",
    "SentenceTextSplitter",
    "BaseVectorStore",
    "VectorStore",
    "FAISSVectorStore",
]
