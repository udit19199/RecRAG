# RecRAG Architecture & Onboarding Course

Welcome to the **RecRAG Architecture Guide**!

Instead of a massive monolithic document, this guide is structured as a crash course to get you onboarded as quickly as possible. Whether you're here to fix a bug in the Retrieval API or completely revamp the Frontend, this guide will give you the context you need.

## 📖 Course Syllabus

Work your way through these modules to develop a strong mental model of the codebase:

- **[Module 1: Getting Started](docs/onboarding/01_getting_started.md)** - Run the app locally and explore the repository structure.
- **[Module 2: Architecture Overview](docs/onboarding/02_architecture_overview.md)** - The 30-second summary and the 5-layer backend design.
- **[Module 3: APIs & Runtime](docs/onboarding/03_api_and_runtime.md)** - Understanding FastAPI routes, state machines, and concurrency.
- **[Module 4: Pipelines, Adapters & Vector Store](docs/onboarding/04_pipelines_and_adapters.md)** - The core RAG logic, the provider registry pattern, and Milvus integration.
- **[Module 5: Tracing Flows](docs/onboarding/05_tracing_flows.md)** - End-to-end Mermaid sequence diagrams showing exact data flow.

---

## 🎯 Quick Reference

Need to jump straight into the code? Here is where to look:

| "I want to..." | Look here |
|----------------|-----------|
| Add a new LLM provider | `src/adapters/` + register in `__init__.py` |
| Change how chunks are created | `src/splitters.py` |
| Change the RAG prompt template | `config.toml` → `[retrieval] context_template` |
| Add a new API endpoint | `app/retrieval/main.py` or `app/ingestion/main.py` |
| Change the query flow | `src/pipelines/retrieval.py` |
| Change how PDFs are parsed | `src/loaders.py` |
| Add a new eval metric | `src/evaluation/ragas_eval.py` |
| Change Milvus connection settings | `config.toml` → `[storage]` |

## 🛠 Useful Files

- **`AGENTS.md`**: Contains key repository commands, Python style guidelines, and testing expectations.
- **`config.toml`**: The main configuration file (secrets go in `.env`).
