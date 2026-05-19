"""Document loaders for RecRAG."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import logging
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument

logger = logging.getLogger(__name__)


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


class VisionPDFLoader(BaseDocumentLoader):
    """Document loader that uses vision models to extract text from PDF pages.

    Renders each page as an image and uses a VLM to extract text,
    which is better for PDFs with infographics, charts, and scanned content.
    """

    def __init__(
        self,
        directory: Path | str,
        vision_provider: str = "openai",
        vision_model: str = "gpt-4o-mini",
        vision_kwargs: dict[str, Any] | None = None,
        dpi: int = 150,
    ):
        self.directory = Path(directory)
        self.vision_provider = vision_provider
        self.vision_model = vision_model
        self.vision_kwargs = vision_kwargs or {}
        self.dpi = dpi
        self._extractor = None

    def _get_extractor(self):
        if self._extractor is None:
            from adapters.vision import create_vision_extractor

            self._extractor = create_vision_extractor(
                self.vision_provider,
                model=self.vision_model,
                **self.vision_kwargs,
            )
        return self._extractor

    def load(self) -> list[LlamaDocument]:
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        documents = []
        for pdf_path in sorted(self.directory.glob("*.pdf")):
            try:
                documents.extend(self.load_file(pdf_path))
            except Exception as e:
                logger.warning("Failed to load %s with vision: %s", pdf_path, e)
        return documents

    def load_file(self, file_path: Path | str) -> list[LlamaDocument]:
        from utils.pdf_images import pdf_to_images

        file_path = Path(file_path)
        extractor = self._get_extractor()
        documents = []

        for page_num, image_bytes in pdf_to_images(file_path, dpi=self.dpi):
            try:
                text = extractor.extract_text(image_bytes)
                if text.strip():
                    doc = LlamaDocument(
                        text=text,
                        metadata={
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "page_label": str(page_num),
                            "extraction_mode": "vision_assisted",
                            "vision_provider": self.vision_provider,
                            "vision_model": self.vision_model,
                        },
                    )
                    documents.append(doc)
                    logger.debug(
                        "Extracted %d chars from %s page %d",
                        len(text),
                        file_path.name,
                        page_num,
                    )
            except Exception as e:
                logger.warning(
                    "Vision extraction failed for %s page %d: %s",
                    file_path.name,
                    page_num,
                    e,
                )

        return documents


class HybridPDFLoader(BaseDocumentLoader):
    """Document loader that combines text extraction with vision fallback.

    First attempts standard text extraction, then uses vision for pages
    with low text yield. Useful when you want the best of both approaches.
    """

    def __init__(
        self,
        directory: Path | str,
        vision_provider: str = "openai",
        vision_model: str = "gpt-4o-mini",
        vision_kwargs: dict[str, Any] | None = None,
        min_chars_per_page: int = 100,
        dpi: int = 150,
    ):
        self.directory = Path(directory)
        self.text_loader = PDFLoader(directory)
        self.vision_loader = VisionPDFLoader(
            directory,
            vision_provider=vision_provider,
            vision_model=vision_model,
            vision_kwargs=vision_kwargs,
            dpi=dpi,
        )
        self.min_chars_per_page = min_chars_per_page

    def load(self) -> list[LlamaDocument]:
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        documents = []
        for pdf_path in sorted(self.directory.glob("*.pdf")):
            try:
                documents.extend(self.load_file(pdf_path))
            except Exception as e:
                logger.warning("Failed to load %s: %s", pdf_path, e)
        return documents

    def load_file(self, file_path: Path | str) -> list[LlamaDocument]:
        file_path = Path(file_path)

        # Try text extraction first
        text_docs = self.text_loader.load_file(file_path)

        # Check if text extraction yielded enough content
        total_chars = sum(len(doc.text) for doc in text_docs)
        from utils.pdf_images import get_pdf_page_count

        try:
            page_count = get_pdf_page_count(file_path)
        except Exception:
            page_count = max(len(text_docs), 1)

        avg_chars_per_page = total_chars / max(page_count, 1)

        if avg_chars_per_page >= self.min_chars_per_page:
            # Text extraction is sufficient
            for doc in text_docs:
                doc.metadata["extraction_mode"] = "text_only"
            return text_docs

        # Fall back to vision extraction
        logger.info(
            "Text extraction yielded %.0f chars/page for %s, using vision fallback",
            avg_chars_per_page,
            file_path.name,
        )
        return self.vision_loader.load_file(file_path)


# Default alias
DocumentLoader = PDFLoader


class LlamaParseLoader(BaseDocumentLoader):
    """Document loader using LlamaParse cloud API.

    Requires ``LLAMA_CLOUD_API_KEY`` environment variable.
    Only available when ``RECRAG_DEV=1`` is set.
    Provides better parsing for complex PDFs, tables, and scanned docs.
    """

    def __init__(
        self,
        directory: Path | str,
        result_type: str = "markdown",
        tier: str = "agentic",
        **kwargs: Any,
    ):
        self.directory = Path(directory)
        self.result_type = result_type
        self.tier = tier
        self._extra_kwargs = kwargs
        self._parser = None

    def _get_parser(self):
        if self._parser is not None:
            return self._parser
        try:
            from llama_parse import LlamaParse
        except ImportError:
            raise ImportError(
                "llama-parse package is required. Install with: uv pip install llama-parse"
            )

        api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY environment variable is required for LlamaParseLoader"
            )

        self._parser = LlamaParse(
            result_type=self.result_type,
            tier=self.tier,
            api_key=api_key,
            **self._extra_kwargs,
        )
        return self._parser

    def load(self) -> list[LlamaDocument]:
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        documents = []
        for pdf_path in sorted(self.directory.glob("*.pdf")):
            try:
                documents.extend(self.load_file(pdf_path))
            except Exception as e:
                logger.warning("LlamaParse failed for %s: %s", pdf_path, e)
        return documents

    def load_file(self, file_path: Path | str) -> list[LlamaDocument]:
        parser = self._get_parser()
        file_path = Path(file_path)

        # LlamaParse returns a list of Document objects
        docs = parser.load_data(str(file_path))

        # Convert to LlamaDocument if needed
        result = []
        for doc in docs:
            text = getattr(doc, "text", getattr(doc, "content", str(doc)))
            metadata = dict(getattr(doc, "metadata", {}))
            metadata.setdefault("file_name", file_path.name)
            metadata.setdefault("file_path", str(file_path))
            metadata.setdefault("extraction_mode", "llamaparse")
            metadata.setdefault("tier", self.tier)

            llama_doc = LlamaDocument(text=text, metadata=metadata)
            result.append(llama_doc)

        if not result:
            logger.warning("LlamaParse returned no content for %s", file_path)

        return result


__all__ = [
    "BaseDocumentLoader",
    "PDFLoader",
    "VisionPDFLoader",
    "HybridPDFLoader",
    "LlamaParseLoader",
    "DocumentLoader",
]
