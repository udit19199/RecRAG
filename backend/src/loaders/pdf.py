from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument
from .base import BaseDocumentLoader


class PDFLoader(BaseDocumentLoader):
    """Document loader for PDF files using llama-index."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def load(self) -> list[LlamaDocument]:
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        reader = SimpleDirectoryReader(str(self.directory))
        return reader.load_data()

    def load_file(self, file_path: Path | str) -> list[LlamaDocument]:
        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        return reader.load_data()
