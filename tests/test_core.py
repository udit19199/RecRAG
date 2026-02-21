from pathlib import Path

import pytest

from loaders import DocumentLoader
from splitters import TextSplitter
from stores import VectorStore


class TestTextSplitter:
    def test_split_documents_empty(self) -> None:
        splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_documents([])
        assert chunks == []

    def test_split_text_returns_strings(self) -> None:
        splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
        text = "This is a test sentence. " * 20
        chunks = splitter.split_text(text)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)


class TestVectorStore:
    def test_add_single_document(self, temp_vector_store: VectorStore) -> None:
        embedding = [0.1] * 128
        document = "Test document content"
        metadata = {"source": "test.pdf"}

        temp_vector_store.add([embedding], [document], [metadata])

        assert temp_vector_store.count == 1
        assert len(temp_vector_store._metadata) == 1
        assert temp_vector_store._metadata[0]["text"] == document
        assert temp_vector_store._metadata[0]["source"] == "test.pdf"
        # Embeddings are stored in FAISS only, not in metadata (to save memory)
        assert "embedding" not in temp_vector_store._metadata[0]

    def test_add_multiple_documents(self, temp_vector_store: VectorStore) -> None:
        embeddings = [[0.1] * 128, [0.2] * 128, [0.3] * 128]
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        metadata_list = [{"source": "a.pdf"}, {"source": "b.pdf"}, {"source": "c.pdf"}]

        temp_vector_store.add(embeddings, documents, metadata_list)

        assert temp_vector_store.count == 3
        assert len(temp_vector_store._metadata) == 3

    def test_search_returns_results(self, temp_vector_store: VectorStore) -> None:
        embeddings = [[0.1] * 128, [0.5] * 128]
        documents = ["Document A", "Document B"]
        metadata_list = [{"source": "a.pdf"}, {"source": "b.pdf"}]

        temp_vector_store.add(embeddings, documents, metadata_list)

        query_embedding = [0.1] * 128
        distances, results = temp_vector_store.search(query_embedding, k=2)

        assert len(results) == 2
        assert results[0].text == "Document A"
        assert hasattr(results[0], "distance")

    def test_delete_all(self, temp_vector_store: VectorStore) -> None:
        embeddings = [[0.1] * 128, [0.2] * 128]
        documents = ["Doc 1", "Doc 2"]

        temp_vector_store.add(embeddings, documents)
        temp_vector_store.delete_all()

        assert temp_vector_store.count == 0
        assert temp_vector_store._metadata == []

    def test_remove_by_sources_removes_metadata(
        self, temp_vector_store: VectorStore
    ) -> None:
        embeddings = [[0.1] * 128, [0.2] * 128, [0.3] * 128]
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        metadata_list = [
            {"source": "a.pdf"},
            {"source": "b.pdf"},
            {"source": "a.pdf"},
        ]

        temp_vector_store.add(embeddings, documents, metadata_list)
        assert temp_vector_store.count == 3
        assert len(temp_vector_store._metadata) == 3

        temp_vector_store.add([[0.4] * 128], ["Doc 4"], [{"source": "a.pdf"}])

        assert len(temp_vector_store._metadata) == 2
        sources = {m["source"] for m in temp_vector_store._metadata}
        assert sources == {"a.pdf", "b.pdf"}
        assert temp_vector_store.count == 2

    def test_metadata_consistency_after_source_update(
        self, temp_vector_store: VectorStore
    ) -> None:
        """Test that metadata is correctly updated when re-adding same source.

        Note: FAISS index is not rebuilt (would require full re-indexing).
        Metadata tracks the latest documents per source correctly.
        """
        embeddings = [[0.1] * 128, [0.2] * 128]
        documents = ["Doc 1", "Doc 2"]
        metadata_list = [{"source": "a.pdf"}, {"source": "b.pdf"}]

        temp_vector_store.add(embeddings, documents, metadata_list)

        # Re-add with same source - metadata should be updated
        temp_vector_store.add([[0.3] * 128], ["Doc 3"], [{"source": "a.pdf"}])

        # Metadata should show the latest state
        assert len(temp_vector_store._metadata) == 2
        texts = {m["text"] for m in temp_vector_store._metadata}
        assert texts == {"Doc 2", "Doc 3"}

    def test_embeddings_not_stored_in_metadata(
        self, temp_vector_store: VectorStore
    ) -> None:
        """Test that embeddings are stored in FAISS only, not metadata (memory optimization)."""
        embedding = [0.5] * 128
        document = "Test document"
        metadata = {"source": "test.pdf"}

        temp_vector_store.add([embedding], [document], [metadata])

        # Embeddings stored in FAISS only to save ~50% memory
        assert "embedding" not in temp_vector_store._metadata[0]
        # But search still works via FAISS
        distances, results = temp_vector_store.search(embedding, k=1)
        assert len(results) == 1
        assert results[0].text == document


class TestDocumentLoader:
    def test_load_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        loader = DocumentLoader(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_empty_directory_raises_value_error(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        loader = DocumentLoader(empty_dir)
        with pytest.raises(ValueError, match="No files found"):
            loader.load()
