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


class LiteparseLoader(BaseDocumentLoader):
    """Document loader using LiteParse CLI (lit) for local PDF parsing.

    Parses PDFs locally - no cloud dependencies, no API keys.
    Supports OCR, bounding boxes, page ranges.

    Requires ``@llamaindex/liteparse`` to be installed globally:
        npm i -g @llamaindex/liteparse

    For Office docs (DOCX, PPTX, XLSX), LibreOffice is required.
    For image parsing, ImageMagick is required.
    """

    def __init__(
        self,
        directory: Path | str,
        dpi: int = 150,
        ocr_enabled: bool = True,
        ocr_language: str = "en",
        target_pages: str | None = None,
        max_pages: int | None = None,
        precise_bbox: bool = True,
        skip_diagonal_text: bool = False,
        preserve_small_text: bool = False,
    ):
        self.directory = Path(directory)
        self.dpi = dpi
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.target_pages = target_pages
        self.max_pages = max_pages
        self.precise_bbox = precise_bbox
        self.skip_diagonal_text = skip_diagonal_text
        self.preserve_small_text = preserve_small_text

    def _build_cmd(self, file_path: Path) -> list[str]:
        cmd = ["npx", "--yes", "@llamaindex/liteparse", "parse", str(file_path), "--format", "json"]

        if not self.ocr_enabled:
            cmd.append("--no-ocr")
        if self.ocr_language != "en":
            cmd.extend(["--ocr-language", self.ocr_language])
        if self.dpi != 150:
            cmd.extend(["--dpi", str(self.dpi)])
        if self.target_pages:
            cmd.extend(["--target-pages", self.target_pages])
        if self.max_pages is not None:
            cmd.extend(["--max-pages", str(self.max_pages)])
        if not self.precise_bbox:
            cmd.append("--no-precise-bbox")
        if self.skip_diagonal_text:
            cmd.append("--skip-diagonal-text")
        if self.preserve_small_text:
            cmd.append("--preserve-small-text")

        return cmd

    def _extract_from_json(self, data: Any, file_path: Path) -> list[LlamaDocument]:
        """Convert LiteParse JSON output to LlamaDocuments."""
        documents: list[LlamaDocument] = []

        if isinstance(data, list):
            # List of pages
            for i, page in enumerate(data):
                text = page.get("text", page.get("content", "")) or ""
                if text.strip():
                    doc = LlamaDocument(
                        text=text,
                        metadata={
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "page_label": str(page.get("page", i + 1)),
                            "extraction_mode": "liteparse",
                        },
                    )
                    documents.append(doc)
        elif isinstance(data, dict):
            # Single document or a container with pages key
            pages = data.get("pages", data.get("results", [data]))
            if isinstance(pages, list):
                return self._extract_from_json(pages, file_path)

            text = data.get("text", data.get("content", "")) or ""
            if text.strip():
                doc = LlamaDocument(
                    text=text,
                    metadata={
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "page_label": str(data.get("page", 1)),
                        "extraction_mode": "liteparse",
                    },
                )
                documents.append(doc)

        return documents

    def load(self) -> list[LlamaDocument]:
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        documents = []
        for pdf_path in sorted(self.directory.glob("*.pdf")):
            try:
                documents.extend(self.load_file(pdf_path))
            except Exception as e:
                logger.warning("LiteParse failed for %s: %s", pdf_path, e)
        return documents

    def load_file(self, file_path: Path | str) -> list[LlamaDocument]:
        import json
        import subprocess

        file_path = Path(file_path)
        cmd = self._build_cmd(file_path)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=300
            )
        except FileNotFoundError:
            raise RuntimeError(
                "npx not found. Ensure Node.js is installed."
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip()
            raise RuntimeError(
                f"LiteParse failed for {file_path.name}: {stderr or e}"
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning(
                "LiteParse returned non-JSON output for %s, treating as plain text",
                file_path.name,
            )
            text = result.stdout.strip()
            if text:
                return [
                    LlamaDocument(
                        text=text,
                        metadata={
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "extraction_mode": "liteparse",
                        },
                    )
                ]
            return []

        documents = self._extract_from_json(data, file_path)

        if not documents:
            logger.warning("LiteParse returned no content for %s", file_path.name)

        return documents


__all__ = [
    "BaseDocumentLoader",
    "PDFLoader",
    "VisionPDFLoader",
    "HybridPDFLoader",
    "LlamaParseLoader",
    "LiteparseLoader",
    "DocumentLoader",
]
