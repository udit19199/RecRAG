from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
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
