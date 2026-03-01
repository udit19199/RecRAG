from abc import ABC, abstractmethod
from pathlib import Path
from llama_index.core.schema import Document as LlamaDocument

class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self) -> list[LlamaDocument]:
        pass

    @abstractmethod
    def load_file(self, file_path: Path | str) -> list[LlamaDocument]:
        pass
