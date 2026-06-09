from unittest.mock import patch

from adapters.gemini import GEMINI_EMBED_BATCH_SIZE, GeminiEmbedder


class TestGeminiEmbedder:
    @patch("adapters.gemini._get_api_key", return_value="test-key")
    @patch("adapters.gemini._call_gemini")
    def test_embed_batch_chunks_above_api_limit(
        self, mock_call: object, _mock_key: object
    ) -> None:
        embedder = GeminiEmbedder(model="gemini-embedding-001")
        texts = [f"text-{i}" for i in range(GEMINI_EMBED_BATCH_SIZE + 1)]

        def fake_call(_url: str, payload: dict, _key: str, _timeout: int) -> dict:
            count = len(payload["requests"])
            return {"embeddings": [{"values": [float(i)]} for i in range(count)]}

        mock_call.side_effect = fake_call

        result = embedder.embed_batch(texts)

        assert len(result) == len(texts)
        assert mock_call.call_count == 2
