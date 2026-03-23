"""Tests for PDF image utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


class TestPdfToImages:
    @patch("fitz.open")
    def test_pdf_to_images_yields_pages(self, mock_fitz_open: MagicMock) -> None:
        from utils.pdf_images import pdf_to_images

        # Mock PDF document with 2 pages
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__iter__ = MagicMock(return_value=iter([0, 1]))

        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"fake png data"
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_fitz_open.return_value = mock_doc
        mock_doc.close = MagicMock()

        # Create a fake path that "exists"
        with patch.object(Path, "exists", return_value=True):
            results = list(pdf_to_images(Path("/fake/test.pdf")))

        assert len(results) == 2
        assert results[0][0] == 1  # Page number (1-indexed)
        assert results[1][0] == 2
        assert results[0][1] == b"fake png data"

    def test_pdf_to_images_file_not_found(self) -> None:
        from utils.pdf_images import pdf_to_images

        with pytest.raises(FileNotFoundError):
            list(pdf_to_images(Path("/nonexistent/file.pdf")))


class TestPdfPageToImage:
    @patch("fitz.open")
    def test_single_page_extraction(self, mock_fitz_open: MagicMock) -> None:
        from utils.pdf_images import pdf_page_to_image

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=5)

        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"page 3 image"
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_fitz_open.return_value = mock_doc
        mock_doc.close = MagicMock()

        with patch.object(Path, "exists", return_value=True):
            result = pdf_page_to_image(Path("/fake/test.pdf"), page_num=3)

        assert result == b"page 3 image"
        # Should access page index 2 (0-indexed) for page_num 3
        mock_doc.__getitem__.assert_called_with(2)

    @patch("fitz.open")
    def test_invalid_page_number(self, mock_fitz_open: MagicMock) -> None:
        from utils.pdf_images import pdf_page_to_image

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_fitz_open.return_value = mock_doc
        mock_doc.close = MagicMock()

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ValueError, match="out of range"):
                pdf_page_to_image(Path("/fake/test.pdf"), page_num=10)


class TestGetPdfPageCount:
    @patch("fitz.open")
    def test_returns_page_count(self, mock_fitz_open: MagicMock) -> None:
        from utils.pdf_images import get_pdf_page_count

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=42)
        mock_fitz_open.return_value = mock_doc
        mock_doc.close = MagicMock()

        count = get_pdf_page_count(Path("/fake/test.pdf"))
        assert count == 42
