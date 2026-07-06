# RecRAG Frontend

Next.js 16 (App Router) + React 19 + TypeScript + Tailwind CSS + Biome.

## Quick Start

From the repo root, the easiest path is:

```bash
make setup
make dev
```

Or run only the frontend:

```bash
pnpm install
pnpm dev
```

The frontend expects:

- retrieval API on `:8000`
- ingestion API on `:8001`
- orchestrator API on `:8002` (proxied through `/api/orchestrator`)

Override public API URLs with:

```bash
NEXT_PUBLIC_RETRIEVAL_API_URL=http://localhost:8000 \
NEXT_PUBLIC_INGESTION_API_URL=http://localhost:8001 \
pnpm dev
```

## Structure

```
app/
  (main)/             Chat, compare, generate, recommendations
  api/orchestrator/   Server-side proxy to orchestrator API
src/
  components/ui/      Shared UI primitives
  features/           Feature modules (chat, ingestion, model-compare, ...)
  lib/api/            Typed API client helpers
```

## Building

```bash
pnpm build
pnpm lint
pnpm format
```

See the root `README.md` for the full local setup flow.
