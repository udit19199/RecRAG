"""Document loaders for RecRAG."""

from abc import ABC, abstractmethod
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument


class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self) -> list[LlamaDocument]:
        pass

    @abstractmethod
    def load_file(self, file_path: Path | str) -> list[LlamaDocument]:
        pass


class PDFLoader(BaseDocumentLoader):
    """Document loader for PDF files using llama-index."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def load(self) -> list[LlamaDocument]:
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")
        return SimpleDirectoryReader(str(self.directory)).load_data()

    def load_file(self, file_path: Path | str) -> list[LlamaDocument]:
        return SimpleDirectoryReader(input_files=[str(file_path)]).load_data()


# Default alias
DocumentLoader = PDFLoader

__all__ = ["BaseDocumentLoader", "PDFLoader", "DocumentLoader"]
