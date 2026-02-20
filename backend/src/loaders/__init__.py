from .base import BaseDocumentLoader
from .pdf import PDFLoader

DocumentLoader = PDFLoader

__all__ = ["BaseDocumentLoader", "PDFLoader", "DocumentLoader"]
