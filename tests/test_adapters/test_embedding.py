import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

from adapters.embedding import OpenAIEmbedder, OllamaEmbedder


class TestOpenAIEmbedder:
    def test_embed_returns_embedding(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")
        embedder.client = mock_client

        result = embedder.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test text"
        )

    def test_embed_batch_returns_embeddings(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")
        embedder.client = mock_client

        result = embedder.embed_batch(["text 1", "text 2"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["text 1", "text 2"]
        )

    def test_dimension_returns_correct_value(self) -> None:
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")
        assert embedder.dimension == 1536

        embedder_large = OpenAIEmbedder(
            model="text-embedding-3-large", api_key="test-key"
        )
        assert embedder_large.dimension == 3072


class TestOllamaEmbedder:
    def test_embed_returns_embedding(self) -> None:
        with patch("requests.Session.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            embedder = OllamaEmbedder(model="nomic-embed-text")
            result = embedder.embed("test text")

            assert result == [0.1, 0.2, 0.3]
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"]["model"] == "nomic-embed-text"
            assert call_args[1]["json"]["prompt"] == "test text"

    def test_embed_batch_success(self) -> None:
        with patch("requests.Session.post") as mock_post:
            mock_response = MagicMock()
            # Ollama /api/embed returns "embeddings" key
            mock_response.json.return_value = {"embeddings": [[0.1] * 768, [0.1] * 768]}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            embedder = OllamaEmbedder(model="nomic-embed-text", dimension=768)
            result = embedder.embed_batch(["text 1", "text 2"])

            assert len(result) == 2
            assert all(len(emb) == 768 for emb in result)

    def test_embed_batch_raises_on_failure(self) -> None:
        with patch("requests.Session.post") as mock_post:
            # Use RequestException to trigger fallback to parallel which also fails
            mock_post.side_effect = requests.exceptions.RequestException("Connection failed")

            embedder = OllamaEmbedder(model="nomic-embed-text")

            with pytest.raises(RuntimeError) as exc_info:
                embedder.embed_batch(["text 1", "text 2"])

            assert "Embedding failed for 2/2 texts" in str(exc_info.value)

    def test_embed_batch_partial_failure_raises(self) -> None:
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # First call is /api/embed (batch) - we fail it to trigger parallel fallback
            if call_count[0] == 1:
                raise requests.exceptions.RequestException("Batch failed")
            # Next calls are individual /api/embeddings
            if call_count[0] == 3: # Fail the second individual call (idx 1)
                raise Exception("Failed on second call")
            
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": [0.1] * 768}
            mock_response.raise_for_status = MagicMock()
            return mock_response

        with patch("requests.Session.post", side_effect=side_effect):
            embedder = OllamaEmbedder(model="nomic-embed-text")

            with pytest.raises(RuntimeError) as exc_info:
                embedder.embed_batch(["text 1", "text 2", "text 3"])

            assert "Embedding failed for 1/3 texts" in str(exc_info.value)
            assert "indices [1]" in str(exc_info.value)

    def test_dimension_returns_configured_value(self) -> None:
        embedder = OllamaEmbedder(model="nomic-embed-text", dimension=1024)
        assert embedder.dimension == 1024
