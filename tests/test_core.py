from pathlib import Path

import pytest

from loaders import DocumentLoader
from splitters import TextSplitter
from tests.mocks import InMemoryVectorStore


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
    def test_add_single_document(self, temp_vector_store: InMemoryVectorStore) -> None:
        embedding = [0.1] * 128
        document = "Test document content"
        metadata = {"source": "test.pdf"}

        temp_vector_store.add([embedding], [document], [metadata])

        assert temp_vector_store.count == 1
        _, results = temp_vector_store.search(embedding, k=1)
        assert results[0].text == document
        assert results[0].source == "test.pdf"

    def test_add_multiple_documents(
        self, temp_vector_store: InMemoryVectorStore
    ) -> None:
        embeddings = [[0.1] * 128, [0.2] * 128, [0.3] * 128]
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        metadata_list = [{"source": "a.pdf"}, {"source": "b.pdf"}, {"source": "c.pdf"}]

        temp_vector_store.add(embeddings, documents, metadata_list)

        assert temp_vector_store.count == 3

    def test_search_returns_results(
        self, temp_vector_store: InMemoryVectorStore
    ) -> None:
        embeddings = [[0.1] * 128, [0.5] * 128]
        documents = ["Document A", "Document B"]
        metadata_list = [{"source": "a.pdf"}, {"source": "b.pdf"}]

        temp_vector_store.add(embeddings, documents, metadata_list)

        query_embedding = [0.1] * 128
        _distances, results = temp_vector_store.search(query_embedding, k=2)

        assert len(results) == 2
        assert results[0].text == "Document A"
        assert hasattr(results[0], "distance")

    def test_delete_all(self, temp_vector_store: InMemoryVectorStore) -> None:
        embeddings = [[0.1] * 128, [0.2] * 128]
        documents = ["Doc 1", "Doc 2"]

        temp_vector_store.add(embeddings, documents)
        temp_vector_store.delete_all()

        assert temp_vector_store.count == 0
        _, results = temp_vector_store.search([0.1] * 128, k=4)
        assert results == []

    def test_source_deduplication_on_readd(
        self, temp_vector_store: InMemoryVectorStore
    ) -> None:
        """Re-adding documents with an existing source replaces the old ones."""
        embeddings = [[0.1] * 128, [0.9] * 128, [0.15] * 128]
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        metadata_list = [
            {"source": "a.pdf"},
            {"source": "b.pdf"},
            {"source": "a.pdf"},
        ]

        temp_vector_store.add(embeddings, documents, metadata_list)
        assert temp_vector_store.count == 3

        # Re-add under "a.pdf" — both previous a.pdf entries should be removed.
        temp_vector_store.add([[0.2] * 128], ["Doc 4"], [{"source": "a.pdf"}])

        assert temp_vector_store.count == 2

        _, r_b = temp_vector_store.search([0.9] * 128, k=1)
        assert r_b[0].source == "b.pdf"
        assert r_b[0].text == "Doc 2"

        _, r_a = temp_vector_store.search([0.2] * 128, k=1)
        assert r_a[0].source == "a.pdf"
        assert r_a[0].text == "Doc 4"

    def test_metadata_consistency_after_source_update(
        self, temp_vector_store: InMemoryVectorStore
    ) -> None:
        """When a source is updated its old content is no longer returned."""
        embeddings = [[0.1] * 128, [0.9] * 128]
        documents = ["Doc 1", "Doc 2"]
        metadata_list = [{"source": "a.pdf"}, {"source": "b.pdf"}]

        temp_vector_store.add(embeddings, documents, metadata_list)

        temp_vector_store.add([[0.3] * 128], ["Doc 3"], [{"source": "a.pdf"}])

        assert temp_vector_store.count == 2

        _, results = temp_vector_store.search([0.3] * 128, k=1)
        assert results[0].text == "Doc 3"
        assert results[0].source == "a.pdf"

    def test_search_returns_correct_metadata(
        self, temp_vector_store: InMemoryVectorStore
    ) -> None:
        """Extra metadata keys are preserved and returned in search results."""
        embedding = [0.5] * 128
        document = "Test document"
        metadata = {"source": "test.pdf", "page": 3}

        temp_vector_store.add([embedding], [document], [metadata])

        _distances, results = temp_vector_store.search(embedding, k=1)
        assert len(results) == 1
        assert results[0].text == document
        assert results[0].metadata.get("page") == 3


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


class TestDevSharedMilvusUri:
    def test_dev_shared_milvus_uri_reuse_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json
        import stores

        shared_dir = tmp_path / "recrag_dev"
        shared_dir.mkdir()
        port_file = shared_dir / "milvus_port.json"
        lock_file = shared_dir / "startup.lock"
        db_file = shared_dir / "milvus_lite.db"

        monkeypatch.setattr(stores, "_DEV_SHARED_DIR", str(shared_dir))
        monkeypatch.setattr(stores, "_DEV_SHARED_DB", str(db_file))
        monkeypatch.setattr(stores, "_DEV_PORT_FILE", str(port_file))
        monkeypatch.setattr(stores, "_DEV_LOCK_FILE", str(lock_file))

        dummy_port = 19530
        dummy_pid = 99999
        with open(port_file, "w") as fh:
            json.dump({"port": dummy_port, "pid": dummy_pid}, fh)

        monkeypatch.setattr(stores, "_is_port_open", lambda host, port: True)
        monkeypatch.setattr(stores, "_is_process_alive", lambda pid: True)

        uri = stores._dev_shared_milvus_uri()
        assert uri == f"http://127.0.0.1:{dummy_port}"

    def test_dev_shared_milvus_uri_start_new(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json
        import os
        import stores
        from unittest.mock import MagicMock

        shared_dir = tmp_path / "recrag_dev"
        shared_dir.mkdir()
        port_file = shared_dir / "milvus_port.json"
        lock_file = shared_dir / "startup.lock"
        db_file = shared_dir / "milvus_lite.db"

        monkeypatch.setattr(stores, "_DEV_SHARED_DIR", str(shared_dir))
        monkeypatch.setattr(stores, "_DEV_SHARED_DB", str(db_file))
        monkeypatch.setattr(stores, "_DEV_PORT_FILE", str(port_file))
        monkeypatch.setattr(stores, "_DEV_LOCK_FILE", str(lock_file))

        monkeypatch.setattr(stores, "_is_port_open", lambda host, port: False)
        monkeypatch.setattr(stores, "_is_process_alive", lambda pid: False)

        mock_server_manager = MagicMock()
        mock_server_manager.start_and_get_uri.return_value = "http://127.0.0.1:12345"

        monkeypatch.setattr(
            "milvus_lite.server_manager.server_manager_instance",
            mock_server_manager,
        )

        uri = stores._dev_shared_milvus_uri()
        assert uri == "http://127.0.0.1:12345"
        mock_server_manager.start_and_get_uri.assert_called_once_with(str(db_file))

        with open(port_file) as fh:
            info = json.load(fh)
        assert info["port"] == 12345
        assert info["pid"] == os.getpid()
