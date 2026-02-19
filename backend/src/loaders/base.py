from abc import ABC, abstractmethod
from pathlib import Path
from llama_index.core.schema import Document as LlamaDocument

class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self) -> list[LlamaDocument]:
        """Load all documents from the configured directory."""
        pass

    @abstractmethod
    def load_file(self, file_path: Path | str) -> list[LlamaDocument]:
        """Load a single file."""
        pass
