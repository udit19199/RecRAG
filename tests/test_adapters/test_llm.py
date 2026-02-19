import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

from adapters.llm import OpenAILLM, OllamaLLM


class TestOpenAILLM:
    def test_generate_returns_response(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Generated response"))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")
        llm.client = mock_client

        result = llm.generate("test prompt")

        assert result == "Generated response"
        mock_client.chat.completions.create.assert_called_once()

    def test_chat_returns_response(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Chat response"))]
        mock_client.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")
        llm.client = mock_client

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = llm.chat(messages)

        assert result == "Chat response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=None,
        )

    def test_supports_streaming_true(self) -> None:
        llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")
        assert llm.supports_streaming is True


class TestOllamaLLM:
    def test_generate_returns_response(self) -> None:
        with patch("requests.Session.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Generated response"}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            llm = OllamaLLM(model="llama3")
            result = llm.generate("test prompt")

            assert result == "Generated response"
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"]["model"] == "llama3"
            assert call_args[1]["json"]["prompt"] == "test prompt"

    def test_generate_sets_stream_false(self) -> None:
        with patch("requests.Session.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "response"}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            llm = OllamaLLM(model="llama3")
            llm.generate("test")

            call_args = mock_post.call_args
            assert call_args[1]["json"]["stream"] is False

    def test_chat_returns_response(self) -> None:
        with patch("requests.Session.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"content": "Chat response"}}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            llm = OllamaLLM(model="llama3")
            messages = [{"role": "user", "content": "Hello"}]
            result = llm.chat(messages)

            assert result == "Chat response"
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"]["messages"] == messages
            assert call_args[1]["json"]["stream"] is False

    def test_chat_sets_stream_false(self) -> None:
        with patch("requests.Session.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"content": "response"}}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            llm = OllamaLLM(model="llama3")
            llm.chat([{"role": "user", "content": "test"}])

            call_args = mock_post.call_args
            assert call_args[1]["json"]["stream"] is False

    def test_supports_streaming_false(self) -> None:
        llm = OllamaLLM(model="llama3")
        assert llm.supports_streaming is False

    def test_generate_with_max_tokens(self) -> None:
        with patch("requests.Session.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "response"}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            llm = OllamaLLM(model="llama3", max_tokens=100)
            llm.generate("test")

            call_args = mock_post.call_args
            assert call_args[1]["json"]["options"] == {"num_predict": 100}
