# Contributing to RecRAG

Thank you for your interest in contributing to RecRAG! This document provides guidelines for contributions.

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js & pnpm
- Docker & Docker Compose
- [uv](https://docs.astral.sh/uv/) (package manager)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-org/RecRAG.git
cd RecRAG

# Full setup (install dependencies and create .env)
make setup
```

## Development Workflow

### Running Locally
To start all services concurrently:
```bash
make dev
```

To run individual services:
```bash
make retrieval  # Port 8000
make ingestion  # Port 8001
make frontend   # Port 3000
```

### Infrastructure (Docker)
To start infrastructure only (Milvus/Ollama):
```bash
make infra-up
```

To start the full stack in Docker:
```bash
make docker-up
```

### Quality Checks
```bash
make lint       # Ruff (backend) + Biome (frontend)
make format     # Format both backend and frontend
make test       # Run pytest
make typecheck  # Run mypy
make check      # Run lint, typecheck, and test
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
| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style (formatting, whitespace) |
| `refactor` | Code refactoring |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvements |
| `ci` | CI/CD changes |

---

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following code style guidelines.

3. **Run quality checks**:
   ```bash
   make check
   ```

4. **Commit with proper message format**.

5. **Push and create PR**.

---

## Code Style

### Python
- Follow PEP 8 (enforced by Ruff).
- Use type hints for all functions.
- Write docstrings for public functions/classes.

### Frontend
- Enforced by Biome (linting and formatting).

---

## Questions?

Open an issue for questions or discussions about contributions.
