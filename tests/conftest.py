from pathlib import Path
from typing import Any

import pytest

from adapters.base import BaseEmbedder, BaseLLM
from stores import VectorStore


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing."""

    def __init__(self, dimension: int = 1536, **kwargs: Any):
        super().__init__("mock-embedder", **kwargs)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return [0.1] * self._dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension for _ in texts]


class MockLLM(BaseLLM):
    """Mock LLM for testing."""

    def __init__(self, model: str = "mock-llm", **kwargs: Any):
        super().__init__(model, **kwargs)

    @property
    def supports_streaming(self) -> bool:
        return False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return "Mock response"

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "Mock chat response"


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dimension=128)


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def temp_vector_store(
    temp_storage_dir: Path, mock_embedder: MockEmbedder
) -> VectorStore:
    index_path = temp_storage_dir / "test.index"
    metadata_path = temp_storage_dir / "test.json"
    return VectorStore(
        dimension=mock_embedder.dimension,
        index_path=index_path,
        metadata_path=metadata_path,
    )


@pytest.fixture
def temp_pdf_dir(tmp_path: Path) -> Path:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    return pdf_dir


@pytest.fixture
def sample_pdf(temp_pdf_dir: Path) -> Path:
    pdf_path = temp_pdf_dir / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\ntest content\n%%EOF")
    return pdf_path


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    config_content = """
[embedding]
provider = "openai"
model = "text-embedding-3-small"

[llm]
provider = "openai"
model = "gpt-4o-mini"

[storage]
directory = "storage"

[ingestion]
directory = "data/pdfs"
chunk_size = 1024
chunk_overlap = 50

[retrieval]
top_k = 4
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)
    return config_path
