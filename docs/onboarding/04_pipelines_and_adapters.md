# Module 4: Pipelines, Adapters, and Stores

## Layer 3: Pipelines (`src/pipelines/`)

This is where the actual RAG logic lives.

### RetrievalPipeline
Handles the question -> answer flow:
1. `retrieve(query)`: Embeds the query and searches the vector store.
2. `generate(query, contexts)`: Truncates contexts to fit token limits (`tiktoken`), formats the prompt, and generates the answer using the LLM.

### IngestionPipeline
Handles PDF -> Vector Store:
- **Text-only:** PDF → extract text → split → embed → store.
- **Vision-assisted:** PDF → render images → vision LLM describes images → embed descriptions → store.

## Layer 4: Adapters (`src/adapters/`)

We support 4 LLM providers (OpenAI, Ollama, NIM, Gemini). We use a **registry pattern** so the pipeline doesn't care which provider is in use.

Providers register themselves at import time:
```python
register_llm("openai", OpenAILLM)
```

Adding a new provider is as simple as creating a new adapter class and registering it. No pipeline changes needed!

## Layer 5: Vector Store (`src/stores.py`)

A `VectorStore` class wraps **Milvus**. 
Each embedder/vision combination gets its own collection
(e.g. `recrag_gpt_4o_mini_text_embedding_3_small`).
This means you can switch embedders without losing previously indexed data!

Now let's trace exactly how data flows through these layers in [Module 5: Tracing Flows](05_tracing_flows.md).
