"""PDF to image conversion utilities."""

from pathlib import Path
from collections.abc import Iterator
import logging

logger = logging.getLogger(__name__)


def pdf_to_images(
    pdf_path: Path | str,
    dpi: int = 150,
    fmt: str = "PNG",
) -> Iterator[tuple[int, bytes]]:
    """Convert PDF pages to images.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (default 150).
        fmt: Image format (PNG or JPEG).

    Yields:
        Tuple of (page_number, image_bytes) for each page.
        Page numbers are 1-indexed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for PDF-to-image conversion. "
            "Install with: pip install pymupdf"
        ) from e

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        zoom = dpi / 72  # Default PDF resolution is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix)

            if fmt.upper() == "PNG":
                image_bytes = pix.tobytes("png")
            elif fmt.upper() in ("JPEG", "JPG"):
                image_bytes = pix.tobytes("jpeg")
            else:
                raise ValueError(f"Unsupported image format: {fmt}")

            yield page_num + 1, image_bytes
    finally:
        doc.close()


def pdf_page_to_image(
    pdf_path: Path | str,
    page_num: int,
    dpi: int = 150,
    fmt: str = "PNG",
) -> bytes:
    """Convert a single PDF page to an image.

    Args:
        pdf_path: Path to the PDF file.
        page_num: Page number (1-indexed).
        dpi: Resolution for rendering.
        fmt: Image format (PNG or JPEG).

    Returns:
        Image bytes.
    """
    try:
        import fitz
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for PDF-to-image conversion. "
            "Install with: pip install pymupdf"
        ) from e

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        if page_num < 1 or page_num > len(doc):
            raise ValueError(f"Page {page_num} out of range. PDF has {len(doc)} pages.")

        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=matrix)

        if fmt.upper() == "PNG":
            return pix.tobytes("png")
        elif fmt.upper() in ("JPEG", "JPG"):
            return pix.tobytes("jpeg")
        else:
            raise ValueError(f"Unsupported image format: {fmt}")
    finally:
        doc.close()


def get_pdf_page_count(pdf_path: Path | str) -> int:
    """Get the number of pages in a PDF."""
    try:
        import fitz
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required. Install with: pip install pymupdf"
        ) from e

    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


__all__ = ["get_pdf_page_count", "pdf_page_to_image", "pdf_to_images"]
