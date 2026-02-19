# RecRAG Refactoring Plan

This document outlines the refactoring steps required to evolve RecRAG into a multi-pipeline, multimodal RAG framework with evaluation and recommendation capabilities.

**Status:** Phase 1 and major portions of Phase 2-3 have been completed. See "Completed Work" section below.

---

## Executive Summary

The codebase has a solid foundation: clean abstract interfaces for embedders and LLMs, separate ingestion/retrieval containers, and a working config system. The following architectural constraints have been addressed:

✅ **Phase 1 - COMPLETED:**
1. ~~Hardcoded provider chains~~ — Provider registry implemented
2. ~~Tight coupling in pipelines~~ — Dependency injection added
3. ~~No vector store abstraction~~ — `BaseVectorStore` interface extracted
4. ~~Modality-coupled ingestion~~ — Extensible loader pattern in place

🔄 **Phase 2-3 - PARTIALLY COMPLETED:**
5. **No evaluation or recommendation infrastructure** — No metrics, benchmarks, or pipeline metadata (still pending)

---

## Completed Work

The following major refactoring has been completed:

### Code Quality Improvements
- ✅ Fixed critical indentation bug in `watch.py` (`run_ingestion_safe` was outside class)
- ✅ Added comprehensive test suite (28 tests)
- ✅ Implemented proper error handling in `OllamaEmbedder.embed_batch()`
- ✅ Added connection pooling via `adapters/utils.py`
- ✅ Extracted constants (DEFAULT_BATCH_SIZE, DEFAULT_CHUNK_SIZE, etc.)
- ✅ Removed dead code from `_remove_by_sources()`
- ✅ Consolidated duplicate logic in pipelines

### Architecture Improvements
- ✅ Extracted `BaseVectorStore`, `BaseDocumentLoader`, `BaseTextSplitter` interfaces
- ✅ Added dependency injection to `IngestionPipeline` and `RetrievalPipeline`
- ✅ Implemented provider registry pattern in `adapters/__init__.py`
- ✅ Created factory modules (`stores/`, `loaders/`)
- ✅ Added batch processing utilities in pipelines

---

## Phase 1: Core Interface Extraction

**Goal**: Extract abstractions for all pipeline dependencies.

### 1.1 Vector Store Interface

**New file**: `backend/src/stores/__init__.py`

```python
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, dimension: int, **kwargs: Any):
        self.dimension = dimension
    
    @abstractmethod
    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None
    ) -> None:
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 4
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        pass
    
    @abstractmethod
    def delete_all(self) -> None:
        pass
    
    @property
    def count(self) -> int:
        pass
```

**Refactor `core.py`**:
- Rename `VectorStore` → `FAISSVectorStore(BaseVectorStore)`
- Keep existing implementation unchanged

**New file**: `backend/src/stores/factory.py`

```python
from typing import Any

def create_vector_store(provider: str, dimension: int, **kwargs: Any) -> BaseVectorStore:
    """Create a vector store instance based on provider."""
    if provider == "faiss":
        from core import FAISSVectorStore
        return FAISSVectorStore(dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")
```

### 1.2 Document Loader Interface

**New file**: `backend/src/loaders/__init__.py`

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self) -> list[Any]:
        """Load all documents from the configured directory."""
        pass
    
    @abstractmethod
    def load_file(self, file_path: Path | str) -> list[Any]:
        """Load a single file."""
        pass
```

**Refactor `core.py`**:
- Rename `DocumentLoader` → `PDFLoader(BaseDocumentLoader)`
- Keep existing implementation (uses `SimpleDirectoryReader` internally)

**New file**: `backend/src/loaders/factory.py`

```python
from pathlib import Path
from typing import Any

def get_loader_for_file(file_path: Path | str) -> BaseDocumentLoader:
    """Get the appropriate loader for a file based on extension."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    if suffix == ".pdf":
        from core import PDFLoader
        return PDFLoader(file_path.parent)
    
    raise ValueError(f"No loader available for file type: {suffix}")


def create_loader(provider: str, directory: Path | str, **kwargs: Any) -> BaseDocumentLoader:
    """Create a document loader based on provider."""
    if provider == "pdf":
        from core import PDFLoader
        return PDFLoader(directory, **kwargs)
    elif provider == "auto":
        # Return a loader that auto-detects based on directory contents
        # For now, fallback to PDF
        from core import PDFLoader
        return PDFLoader(directory, **kwargs)
    else:
        raise ValueError(f"Unknown loader provider: {provider}")
```

### 1.3 Text Splitter Interface

**Refactor `core.py`**:
- Rename `TextSplitter` → `SentenceTextSplitter`
- Add `BaseTextSplitter` abstract class:

```python
class BaseTextSplitter(ABC):
    """Abstract base class for text splitters."""
    
    @abstractmethod
    def split_documents(self, documents: list[Any]) -> list[Chunk]:
        pass
    
    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        pass
```

---

## Phase 2: Pipeline Dependency Injection

**Goal**: Make pipelines composable with injected dependencies.

### 2.1 Refactor IngestionPipeline

**Current**:
```python
class IngestionPipeline:
    def __init__(self, config: dict[str, Any], config_path: Path):
        # Direct instantiation - hard to swap components
        self.embedder = create_embedder_from_config(config)
        self.splitter = TextSplitter(...)
        self.loader = DocumentLoader(...)
        self.vector_store = VectorStore(...)
```

**Target**:
```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class PipelineConfig:
    """Configuration for a pipeline instance."""
    pipeline_id: str
    embedder_provider: str
    embedder_model: str
    vector_store_provider: str = "faiss"
    chunk_size: int = 1024
    chunk_overlap: int = 50
    loader_provider: str = "pdf"
    ingestion_directory: str = "data/pdfs"


class IngestionPipeline:
    def __init__(
        self,
        embedder: BaseEmbedder,
        splitter: BaseTextSplitter,
        loader: BaseDocumentLoader,
        vector_store: BaseVectorStore,
    ):
        self.embedder = embedder
        self.splitter = splitter
        self.loader = loader
        self.vector_store = vector_store
    
    def run(self, force: bool = False) -> dict[str, Any]:
        # Implementation unchanged
        ...
    
    @classmethod
    def from_config(cls, config: dict[str, Any], config_path: Path) -> "IngestionPipeline":
        """Factory method to create pipeline from config."""
        embedder = create_embedder_from_config(config)
        splitter = SentenceTextSplitter(...)
        loader = create_loader("pdf", ...)
        vector_store = create_vector_store("faiss", dimension=embedder.dimension)
        
        return cls(embedder, splitter, loader, vector_store)
```

### 2.2 Refactor RetrievalPipeline

Same pattern: inject `embedder`, `llm`, and `vector_store`.

### 2.3 Add Pipeline Registry

**New file**: `backend/src/registry.py`

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class PipelineDefinition:
    """Definition of a pipeline for registration."""
    pipeline_id: str
    pipeline_type: str  # "ingestion" or "retrieval"
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineRegistry:
    """Registry for pipeline definitions."""
    
    def __init__(self):
        self._pipelines: dict[str, PipelineDefinition] = {}
    
    def register(self, definition: PipelineDefinition) -> None:
        self._pipelines[definition.pipeline_id] = definition
    
    def get(self, pipeline_id: str) -> PipelineDefinition | None:
        return self._pipelines.get(pipeline_id)
    
    def list_all(self) -> list[PipelineDefinition]:
        return list(self._pipelines.values())
    
    def clear(self) -> None:
        self._pipelines.clear()


# Global registry instance
_registry = PipelineRegistry()


def get_registry() -> PipelineRegistry:
    return _registry
```

---

## Phase 3: Provider Registry

**Goal**: Replace hardcoded if/elif chains with extensible registry.

### 3.1 Refactor Adapter Registry

**Current** (`adapters/__init__.py`):
```python
def create_embedder(provider: str, **kwargs):
    if provider == "openai":
        return OpenAIEmbedder(**kwargs)
    elif provider == "ollama":
        return OllamaEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")
```

**Target**:
```python
from typing import Any, Type

_EMBEDDER_REGISTRY: dict[str, Type[BaseEmbedder]] = {}
_LLM_REGISTRY: dict[str, Type[BaseLLM]] = {}


def register_embedder(provider: str, cls: Type[BaseEmbedder]) -> None:
    """Register an embedder provider."""
    _EMBEDDER_REGISTRY[provider] = cls


def register_llm(provider: str, cls: Type[BaseLLM]) -> None:
    """Register an LLM provider."""
    _LLM_REGISTRY[provider] = cls


def create_embedder(provider: str, **kwargs: Any) -> BaseEmbedder:
    """Create an embedder instance based on provider."""
    if provider not in _EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown embedder provider: {provider}. Available: {list(_EMBEDDER_REGISTRY.keys())}")
    return _EMBEDDER_REGISTRY[provider](**kwargs)


def create_llm(provider: str, **kwargs: Any) -> BaseLLM:
    """Create an LLM instance based on provider."""
    if provider not in _LLM_REGISTRY:
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {list(_LLM_REGISTRY.keys())}")
    return _LLM_REGISTRY[provider](**kwargs)


# Register built-in providers
from adapters.embedding import OpenAIEmbedder, OllamaEmbedder
from adapters.llm import OpenAILLM, OllamaLLM

register_embedder("openai", OpenAIEmbedder)
register_embedder("ollama", OllamaEmbedder)
register_llm("openai", OpenAILLM)
register_llm("ollama", OllamaLLM)
```

### 3.2 Future Provider Addition Example

To add NVIDIA NIM support:

```python
# In a new file: adapters/nvidia.py
from adapters.base import BaseEmbedder, BaseLLM

class NVIDIAEmbedder(BaseEmbedder):
    ...

class NVIDIALLM(BaseLLM):
    ...

# Registration (in adapters/__init__.py or separate registration file)
from adapters.nvidia import NVIDIAEmbedder, NVIDIALLM
register_embedder("nvidia", NVIDIAEmbedder)
register_llm("nvidia", NVIDIALLM)
```

---

## Phase 4: Modality-Aware Ingestion

**Goal**: Support new modalities without modifying existing code.

### 4.1 Refactor File Watcher

**Current** (`watch.py:183-184`):
```python
if event.src_path.endswith(".pdf"):
    self._schedule_ingestion(event.src_path)
```

**Target**:
```python
from loaders import get_loader_for_file
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf"}  # Expandable

class IngestionWatcher(FileSystemEventHandler):
    def __init__(self, ...):
        self.supported_extensions = SUPPORTED_EXTENSIONS.copy()
    
    def on_created(self, event):
        if event.is_directory:
            return
        ext = Path(event.src_path).suffix.lower()
        if ext in self.supported_extensions:
            self._schedule_ingestion(event.src_path)
    
    def register_modality(self, extension: str, loader_class: type) -> None:
        """Register a new modality."""
        self.supported_extensions.add(extension)
        # Store loader class for later use
```

### 4.2 Modality Routing

```python
# When adding video support later:
# 1. Create adapters/loaders/video.py with VideoLoader(BaseDocumentLoader)
# 2. Add to factory: create_loader("video", directory)
# 3. Register in watch.py: watcher.supported_extensions.add(".mp4")
# No changes to pipeline code required.
```

---

## Phase 5: Configuration Schema Evolution

**Goal**: Support multiple pipelines in config.

### 5.1 Extended Config Schema

**Current** (`config.toml`):
```toml
[embedding]
provider = "openai"
model = "text-embedding-3-small"

[llm]
provider = "openai"
model = "gpt-4o-mini"

[ingestion]
directory = "data/pdfs"

[retrieval]
top_k = 4
```

**Target**:
```toml
# Default/primary pipeline
[embedding]
provider = "openai"
model = "text-embedding-3-small"

[llm]
provider = "openai"
model = "gpt-4o-mini"

[ingestion]
directory = "data/pdfs"

[retrieval]
top_k = 4

# Pipeline definitions for multi-pipeline support
[[pipelines]]
id = "primary"
embedding.provider = "openai"
embedding.model = "text-embedding-3-small"
llm.provider = "openai"
llm.model = "gpt-4o-mini"
vector_store.provider = "faiss"

[[pipelines]]
id = "local-ollama"
embedding.provider = "ollama"
embedding.model = "nomic-embed-text"
llm.provider = "ollama"
llm.model = "llama3"
vector_store.provider = "faiss"
```

---

## Implementation Order

### ✅ Completed

| Step | Change | Files Affected | Status |
|------|--------|----------------|--------|
| 1 | Extract `BaseVectorStore` interface | `core.py`, `stores/__init__.py` | ✅ Complete |
| 2 | Extract `BaseDocumentLoader` interface | `core.py`, `loaders/__init__.py` | ✅ Complete |
| 3 | Add dependency injection to pipelines | `pipelines.py` | ✅ Complete |
| 4 | Convert provider factory to registry | `adapters/__init__.py` | ✅ Complete |
| 5 | Add modality routing in watcher | `watch.py` | ✅ Complete |
| 6 | Add pipeline registry | `registry.py` | ✅ Complete |
| 7 | Code quality improvements | Multiple files | ✅ Complete |

### 🔄 Remaining Work

| Step | Change | Files Affected | Unlocks |
|------|--------|----------------|---------|
| 8 | Extend config schema for multi-pipeline | `config.py`, `config.toml` | Multi-pipeline config |
| 9 | Add evaluation framework | New `evaluation/` module | Benchmarking capabilities |
| 10 | Add recommendation engine | New `recommendation/` module | Pipeline selection |
| 11 | Hybrid search support | `core.py` | Better retrieval quality |
| 12 | Async execution support | Multiple files | Better performance |

---

## What NOT to Build Yet

- **Evaluation framework** — Requires pipelines to exist first
- **Recommender logic** — Requires evaluation data and model metadata
- **Async execution** — Current sync API is sufficient for initial multi-pipeline
- **Streaming responses** — Interface exists; defer until UI needs it
- **Hybrid search** — Requires vector store interface (Phase 1)
- **GPU/resource awareness** — Requires model capability registry

---

## Backward Compatibility

- All refactors maintain existing API surface where possible
- Factory functions (`create_embedder_from_config`, etc.) remain unchanged
- CLI tools (`ingest.py`, `watch.py`) continue to work
- Config files remain valid (current config is a subset of new schema)

---

## File Change Summary

| Action | File |
|--------|------|
| Modify | `backend/src/core.py` — Rename classes, add interfaces |
| Create | `backend/src/stores/__init__.py` — Vector store interface + factory |
| Create | `backend/src/loaders/__init__.py` — Loader interface + factory |
| Modify | `backend/src/adapters/__init__.py` — Convert to registry |
| Modify | `backend/src/pipelines.py` — Add DI, PipelineConfig |
| Create | `backend/src/registry.py` — Pipeline registry |
| Modify | `backend/watch.py` — Modality routing |
| Modify | `backend/src/config.py` — Extended schema support (optional) |
| Modify | `config.toml` — Extended schema (optional) |
