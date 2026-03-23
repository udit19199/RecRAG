"""Tests for vision extraction components."""

from unittest.mock import MagicMock, patch
import pytest

from adapters.vision import (
    OpenAIVisionExtractor,
    OllamaVisionExtractor,
    NIMVisionExtractor,
    create_vision_extractor,
    list_vision_providers,
    DEFAULT_EXTRACTION_PROMPT,
)
from models.api import ExtractionMode


class TestVisionExtractorRegistry:
    def test_list_vision_providers(self) -> None:
        providers = list_vision_providers()
        assert "openai" in providers
        assert "ollama" in providers
        assert "nim" in providers

    def test_create_vision_extractor_openai(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            extractor = create_vision_extractor("openai", model="gpt-4o-mini")
            assert isinstance(extractor, OpenAIVisionExtractor)
            assert extractor.model == "gpt-4o-mini"
            assert extractor.provider == "openai"

    def test_create_vision_extractor_ollama(self) -> None:
        extractor = create_vision_extractor("ollama", model="llava:latest")
        assert isinstance(extractor, OllamaVisionExtractor)
        assert extractor.model == "llava:latest"
        assert extractor.provider == "ollama"

    def test_create_vision_extractor_nim(self) -> None:
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            extractor = create_vision_extractor("nim", model="test-model")
            assert isinstance(extractor, NIMVisionExtractor)
            assert extractor.model == "test-model"
            assert extractor.provider == "nim"

    def test_create_vision_extractor_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown vision extractor provider"):
            create_vision_extractor("unknown_provider", model="test")


class TestOpenAIVisionExtractor:
    def test_init_requires_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Remove OPENAI_API_KEY if present
            import os

            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OpenAI API key required"):
                OpenAIVisionExtractor(model="gpt-4o-mini")

    def test_init_with_api_key(self) -> None:
        extractor = OpenAIVisionExtractor(model="gpt-4o-mini", api_key="test-key")
        assert extractor.api_key == "test-key"
        assert extractor.model == "gpt-4o-mini"

    def test_encode_image(self) -> None:
        extractor = OpenAIVisionExtractor(model="gpt-4o-mini", api_key="test-key")
        image_bytes = b"test image data"
        encoded = extractor._encode_image(image_bytes)
        assert isinstance(encoded, str)
        # Base64 encoding should produce valid output
        import base64

        decoded = base64.b64decode(encoded)
        assert decoded == image_bytes

    @patch("requests.post")
    def test_extract_text_success(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Extracted text from image"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        extractor = OpenAIVisionExtractor(model="gpt-4o-mini", api_key="test-key")
        result = extractor.extract_text(b"fake image data")

        assert result == "Extracted text from image"
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_extract_text_batch(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Page text"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        extractor = OpenAIVisionExtractor(model="gpt-4o-mini", api_key="test-key")
        results = extractor.extract_text_batch([b"img1", b"img2", b"img3"])

        assert len(results) == 3
        assert all(r == "Page text" for r in results)
        assert mock_post.call_count == 3


class TestOllamaVisionExtractor:
    def test_init_defaults(self) -> None:
        extractor = OllamaVisionExtractor(model="llava:latest")
        assert extractor.model == "llava:latest"
        assert "localhost:11434" in extractor.base_url

    def test_init_custom_base_url(self) -> None:
        extractor = OllamaVisionExtractor(
            model="llava:latest",
            base_url="http://custom:11434",
        )
        assert extractor.base_url == "http://custom:11434"

    @patch("requests.post")
    def test_extract_text_success(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Extracted via Ollama"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        extractor = OllamaVisionExtractor(model="llava:latest")
        result = extractor.extract_text(b"fake image data")

        assert result == "Extracted via Ollama"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/api/generate" in call_args[0][0]


class TestNIMVisionExtractor:
    def test_init_requires_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("NVIDIA_API_KEY", None)
            with pytest.raises(ValueError, match="NVIDIA API key required"):
                NIMVisionExtractor(model="test-model")

    def test_init_with_api_key(self) -> None:
        extractor = NIMVisionExtractor(model="test-model", api_key="test-key")
        assert extractor.api_key == "test-key"
        assert extractor.model == "test-model"

    @patch("requests.post")
    def test_extract_text_success(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "NIM extracted text"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        extractor = NIMVisionExtractor(model="test-model", api_key="test-key")
        result = extractor.extract_text(b"fake image data")

        assert result == "NIM extracted text"


class TestExtractionMode:
    def test_text_only_mode(self) -> None:
        mode = ExtractionMode.TEXT_ONLY
        assert mode.value == "text_only"

    def test_vision_assisted_mode(self) -> None:
        mode = ExtractionMode.VISION_ASSISTED
        assert mode.value == "vision_assisted"

    def test_mode_from_string(self) -> None:
        mode = ExtractionMode("text_only")
        assert mode == ExtractionMode.TEXT_ONLY

        mode = ExtractionMode("vision_assisted")
        assert mode == ExtractionMode.VISION_ASSISTED

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            ExtractionMode("invalid_mode")


class TestDefaultExtractionPrompt:
    def test_prompt_exists(self) -> None:
        assert DEFAULT_EXTRACTION_PROMPT is not None
        assert len(DEFAULT_EXTRACTION_PROMPT) > 100

    def test_prompt_mentions_key_elements(self) -> None:
        prompt = DEFAULT_EXTRACTION_PROMPT.lower()
        assert "text" in prompt
        assert "chart" in prompt or "graph" in prompt
        assert "table" in prompt
