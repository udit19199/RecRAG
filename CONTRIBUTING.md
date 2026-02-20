# Contributing to RecRAG

Thank you for your interest in contributing to RecRAG! This document provides guidelines for contributions.

## Getting Started

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- uv (package manager)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-org/RecRAG.git
cd RecRAG

# Install dependencies
uv sync

# Create .env file (see .env.example)
cp .env.example .env
```

## Development Workflow

### Running Locally
```bash
# Terminal 1: File watcher
uv run python backend/watch.py

# Terminal 2: Streamlit UI
uv run streamlit run backend/app.py
```

### Running with Docker
```bash
docker-compose up -d
```

### Running Tests
```bash
uv run pytest                                     # All tests
uv run pytest tests/test_ingest.py                # Single file
uv run pytest tests/test_ingest.py::test_name     # Single test
```

### Linting & Formatting
```bash
uv run ruff check .                               # Check
uv run ruff format .                              # Format
uv run ruff check --fix .                         # Auto-fix
```

---

## Commit Messages

We use **Conventional Commits** format. All commits must follow this structure:

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(adapters): add HuggingFace embedding provider` |
| `fix` | Bug fix | `fix(llm): set stream=False in Ollama API requests` |
| `docs` | Documentation changes | `docs(readme): update Docker commands` |
| `style` | Code style (formatting, whitespace) | `style(pipelines): fix indentation` |
| `refactor` | Code refactoring | `refactor(config): simplify path resolution` |
| `test` | Adding/updating tests | `test(pipelines): add ingestion unit tests` |
| `chore` | Maintenance tasks | `chore(deps): update dependencies` |
| `perf` | Performance improvements | `perf(embedding): batch embedding requests` |
| `ci` | CI/CD changes | `ci(github): add test workflow` |

### Scopes

Use the module or component name:

| Scope | Description |
|-------|-------------|
| `adapters` | LLM/embedding providers |
| `config` | Configuration loading |
| `pipelines` | Ingestion/retrieval pipelines |
| `core` | Document processing, vector store |
| `docker` | Docker configuration |
| `docs` | Documentation |
| `ci` | CI/CD configuration |

### Subject Line Rules

1. **Max 72 characters**
2. **Lowercase only** (no capitalization)
3. **No period** at the end
4. **Imperative mood** ("add" not "added" or "adds")
5. **Be specific** about what changed

### Body (Optional)

- Explain **what** and **why** (not how)
- Separate from subject with blank line
- Wrap at 72 characters

### Footer (Optional)

- Reference issues: `Closes #123`, `Fixes #456`
- Breaking changes: `BREAKING CHANGE: description`

### Examples

**Simple commit:**
```
feat(adapters): add HuggingFace embedding provider
```

**Commit with body:**
```
fix(llm): set stream=False in Ollama API requests

Ollama returns streaming responses by default which causes JSON
parsing errors. Setting stream=False ensures a single JSON response.
```

**Commit with issue reference:**
```
fix(config): correct regex pattern for env var substitution

The regex pattern was looking for ::- instead of :- which prevented
environment variables from being substituted correctly.

Fixes #42
```

**Breaking change:**
```
refactor(api)!: change embedder interface to support batches

BREAKING CHANGE: The embed() method now returns list[float] instead
of numpy array. Update all callers accordingly.
```

---

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following code style guidelines

3. **Write/update tests** for your changes

4. **Run linting and tests**:
   ```bash
   uv run ruff check .
   uv run pytest
   ```

5. **Commit with proper message format**

6. **Push and create PR**:
   ```bash
   git push origin feat/your-feature-name
   ```

7. **Fill out PR template** completely

### PR Title Format

Use the same format as commit messages:
```
<type>(<scope>): <description>
```

Example: `feat(adapters): add HuggingFace embedding provider`

---

## Code Style

### Python
- Follow PEP 8
- Use type hints for all functions
- Write docstrings for public functions/classes
- Max line length: 100 characters

### Imports
Order: standard library → third-party → local (separate with blank lines)

### Naming
| Element | Convention |
|---------|------------|
| Functions/variables | `snake_case` |
| Classes | `PascalCase` |
| Constants | `UPPER_SNAKE_CASE` |
| Private members | `_underscore_prefix` |

---

## Questions?

Open an issue for questions or discussions about contributions.
