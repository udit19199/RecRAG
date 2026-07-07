# Module 1: Getting Started

Welcome to RecRAG! This guide will get you up and running as quickly as possible.

## What is RecRAG?

RecRAG is a **RAG (Retrieval-Augmented Generation) pipeline** — you upload PDFs, they get chunked and stored in a vector
database, and an LLM answers your questions using the most relevant chunks as context.

## Local Development Setup

To get started coding, you can spin up the entire application stack locally using just a few commands.

1. **Install dependencies:**
   `make install`

2. **Setup the environment:**
   `make setup`

3. **Run the development servers:**
   `make dev`

This will spin up:
- **Retrieval API** on `localhost:8000`
- **Ingestion API** on `localhost:8001`
- **Frontend App** on `localhost:3000`

## Repository Structure

The codebase is split mainly into backend and frontend:

- `src/`, `app/`, `jobs/`: Python Backend
- `frontend/`: Next.js Frontend
- `tests/`: Pytest Suite

You're ready to start exploring! Move on to [Module 2: Architecture Overview](02_architecture_overview.md) to understand how the services communicate.
