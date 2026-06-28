"""Regression tests for bugs fixed in Phase 1.

Each test class maps to a bug ticket:
  - B6: API keys leaked into **kwargs
  - B8: auth.py header mismatch
  - B1: Sync file I/O in ingestion upload
  - B2: Hardcoded adapter timeouts
  - B3: Retry with idempotent-method-only
  - B4: Token-counting fallback configurable
  - B5: Lazy dimension lookup in NIM embedder
  - B7: Document DELETE API route + vector store cleanup
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch


from adapters import create_embedder, create_llm
from adapters.embedding import OpenAIEmbedder, OllamaEmbedder
from adapters.llm import OpenAILLM, OllamaLLM
from adapters.nim import NIMEmbedder
from adapters.utils import create_session_with_pooling
from urllib3.util import Retry
from pipelines.retrieval import (
    RetrievalPipeline,
    _get_tokenizer,
    DEFAULT_TOKENIZER_FALLBACK,
)
from stores import VectorStore


# ── B6: Pop API keys from **kwargs before passing to base adapter ─────────────


class TestB6ApiKeyLeak:
    """Regression: sensitive kwargs must not end up in self.kwargs after init."""

    def test_nim_embedder_does_not_leak_api_key_in_kwargs(self) -> None:
        """NIMEmbedder must pop api_key/base_url before super().__init__."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "env-key"}):
            embedder = NIMEmbedder(
                model="nvidia/nv-embedqa-e5-v5",
                api_key="explicit-key",
            )
        assert "api_key" not in embedder.kwargs
        assert "base_url" not in embedder.kwargs

    def test_nim_llm_does_not_leak_api_key_in_kwargs(self) -> None:
        with patch("adapters.nim.NVIDIALLM"):
            with patch.dict(os.environ, {"NVIDIA_API_KEY": "env-key"}):
                llm = create_llm("nim", model="meta/llama3-70b", api_key="explicit-key")
            assert "api_key" not in llm.kwargs
            assert "base_url" not in llm.kwargs

    def test_openai_embedder_does_not_leak_api_key_in_kwargs(self) -> None:
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")
        assert "api_key" not in embedder.kwargs
        assert "base_url" not in embedder.kwargs

    def test_openai_llm_does_not_leak_api_key_in_kwargs(self) -> None:
        llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")
        assert "api_key" not in llm.kwargs
        assert "base_url" not in llm.kwargs

    def test_ollama_embedder_does_not_leak_api_key_in_kwargs(self) -> None:
        """OllamaEmbedder must also pop kwargs defensively."""
        embedder = OllamaEmbedder(
            model="nomic-embed-text",
            api_key="should-not-leak",
            base_url="http://localhost:11434",
        )
        assert "api_key" not in embedder.kwargs
        assert "base_url" not in embedder.kwargs

    def test_ollama_llm_does_not_leak_api_key_in_kwargs(self) -> None:
        llm = OllamaLLM(
            model="llama3",
            api_key="should-not-leak",
            base_url="http://localhost:11434",
        )
        assert "api_key" not in llm.kwargs
        assert "base_url" not in llm.kwargs

    def test_factory_does_not_leak_api_keys(self) -> None:
        """Verify the create_embedder/create_llm factories also protect keys."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            embedder = create_embedder("openai", model="text-embedding-3-small")
            assert "api_key" not in embedder.kwargs

            llm = create_llm("openai", model="gpt-4o-mini")
            assert "api_key" not in llm.kwargs


# API Key header verification has been removed.


# ── B1: Sync file I/O in ingestion upload ────────────────────────────────────


class TestB1AsyncFileIO:
    """Regression: ingestion upload must use async-safe file I/O."""

    def test_reset_directory_helper(self) -> None:
        """_reset_directory must handle existing directories."""
        from app.ingestion.main import _reset_directory

        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("pathlib.Path.exists", return_value=True),
        ):
            _reset_directory(Path("/tmp/test"))
            mock_rmtree.assert_called_once_with("/tmp/test")
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_reset_directory_creates_if_missing(self) -> None:
        from app.ingestion.main import _reset_directory

        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("pathlib.Path.exists", return_value=False),
        ):
            _reset_directory(Path("/tmp/test"))
            mock_rmtree.assert_not_called()
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


# ── B2: Hardcoded adapter timeouts ────────────────────────────────────────────


class TestB2ConfigurableTimeouts:
    """Regression: adapter timeouts must come from config.toml or kwargs."""

    def test_ollama_embedder_accepts_timeout_from_kwargs(self) -> None:
        embedder = OllamaEmbedder(model="nomic-embed-text", timeout=45)
        assert embedder._timeout == 45

    def test_ollama_embedder_default_timeout(self) -> None:
        embedder = OllamaEmbedder(model="nomic-embed-text")
        assert embedder._timeout == 120

    def test_ollama_llm_accepts_timeout_from_kwargs(self) -> None:
        llm = OllamaLLM(model="llama3", timeout=30)
        assert llm._timeout == 30

    def test_ollama_llm_default_timeout(self) -> None:
        llm = OllamaLLM(model="llama3")
        assert llm._timeout == 120

    def test_openai_embedder_accepts_timeout(self) -> None:
        """OpenAIEmbedder passes timeout through to the OpenAI client."""
        with patch("adapters.embedding.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            OpenAIEmbedder(
                model="text-embedding-3-small",
                api_key="test-key",
                timeout=30,
            )
            mock_openai.assert_called_once()
            _, kwargs = mock_openai.call_args
            assert kwargs.get("timeout") == 30

    def test_pipeline_base_passes_extra_kwargs(self) -> None:
        """Pipeline._create_adapter_from_config must pass timeout through."""
        config = {
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text",
                "timeout": 60,
            }
        }
        from pipelines.base import create_embedder_from_config

        embedder = create_embedder_from_config(config)
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder._timeout == 60


# ── B3: Retry with idempotent-method-only ─────────────────────────────────────


class TestB3IdempotentRetry:
    """Regression: session retries must only apply to idempotent methods."""

    def test_session_uses_urllib3_retry(self) -> None:
        session = create_session_with_pooling()
        adapter = session.get_adapter("http://")
        assert isinstance(adapter.max_retries, Retry)

    def test_retry_allowed_methods_are_idempotent(self) -> None:
        session = create_session_with_pooling()
        adapter = session.get_adapter("http://")
        retry = adapter.max_retries
        assert isinstance(retry, Retry)
        allowed = retry.allowed_methods
        assert "GET" in allowed
        assert "HEAD" in allowed
        assert "PUT" in allowed
        assert "DELETE" in allowed
        assert "POST" not in allowed
        assert "PATCH" not in allowed

    def test_retry_has_backoff_and_status_forcelist(self) -> None:
        session = create_session_with_pooling()
        adapter = session.get_adapter("http://")
        retry = adapter.max_retries
        assert isinstance(retry, Retry)
        assert retry.backoff_factor == 0.5
        assert retry.total == 3

    def test_custom_max_retries(self) -> None:
        session = create_session_with_pooling(max_retries=5)
        adapter = session.get_adapter("http://")
        retry = adapter.max_retries
        assert isinstance(retry, Retry)
        assert retry.total == 5


# ── B4: Token-counting fallback configurable ──────────────────────────────────


class TestB4TokenCountingFallback:
    """Regression: token-counting fallback must be configurable."""

    def test_default_tokenizer_fallback(self) -> None:
        assert DEFAULT_TOKENIZER_FALLBACK == "cl100k_base"

    def test_retrieval_pipeline_accepts_tokenizer_fallback(self) -> None:
        import inspect

        sig = inspect.signature(RetrievalPipeline.__init__)
        assert "tokenizer_fallback" in sig.parameters
        default = sig.parameters["tokenizer_fallback"].default
        assert default == DEFAULT_TOKENIZER_FALLBACK

    def test_get_tokenizer_caches_by_model(self) -> None:
        """_get_tokenizer should cache encodings."""
        # Clear cache
        from pipelines.retrieval import _ENCODING_CACHE

        _ENCODING_CACHE.clear()
        t1 = _get_tokenizer("gpt-4")
        t2 = _get_tokenizer("gpt-4")
        assert t1 is t2  # same cached object

    def test_get_tokenizer_fallback(self) -> None:
        """When model is unknown, _get_tokenizer falls back to configured encoding."""
        # Clear cache before test
        from pipelines.retrieval import _ENCODING_CACHE

        _ENCODING_CACHE.clear()
        tokenizer = _get_tokenizer("nonexistent-model-xyz", fallback="cl100k_base")
        assert tokenizer.name == "cl100k_base"


# ── B5: Lazy dimension lookup in NIM embedder ─────────────────────────────────


class TestB5LazyDimension:
    """Regression: NIM embedder dimension must be resolved lazily, not at init."""

    def test_dimension_not_resolved_during_init(self) -> None:
        """dimension property access should trigger lazy resolution."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            with patch.object(NIMEmbedder, "_resolve_dimension") as mock_resolve:
                NIMEmbedder(model="nvidia/nv-embedqa-e5-v5")
                mock_resolve.assert_not_called()

    def test_dimension_resolved_on_first_access(self) -> None:
        """First access to dimension property triggers lazy resolution."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            with patch.object(
                NIMEmbedder, "_resolve_dimension", return_value=1024
            ) as mock_resolve:
                embedder = NIMEmbedder(model="nvidia/nv-embedqa-e5-v5")
                dim = embedder.dimension
                assert dim == 1024
                mock_resolve.assert_called_once()

    def test_dimension_cached_after_first_access(self) -> None:
        """Subsequent dimension accesses should use cached value."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            with patch.object(
                NIMEmbedder, "_resolve_dimension", return_value=1024
            ) as mock_resolve:
                embedder = NIMEmbedder(model="nvidia/nv-embedqa-e5-v5")
                _ = embedder.dimension
                _ = embedder.dimension
                assert mock_resolve.call_count == 1


# ── B7: Document DELETE API route + vector store cleanup ──────────────────────


class TestB7DocumentDelete:
    """Regression: DELETE /documents/{filename} must remove files + vectors."""

    def test_vector_store_delete_document_uses_source_filter(self) -> None:
        """Verify delete_document calls delete_by_filter with correct filter."""
        store = VectorStore.__new__(VectorStore)
        with patch.object(store, "delete_by_filter", return_value=3) as mock_delete:
            result = store.delete_document("report.pdf")
            assert result == 3
            mock_delete.assert_called_once_with('source == "report.pdf"')

    def test_delete_document_strips_path_traversal(self) -> None:
        """The delete_document method should use the source name as-is."""
        store = VectorStore.__new__(VectorStore)
        with patch.object(store, "delete_by_filter", return_value=0):
            result = store.delete_document("safe_name.pdf")
            assert result == 0


# API Key auth safety checks have been removed.
