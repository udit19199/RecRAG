# RecRAG Frontend

Next.js 16 (App Router) + React 19 + TypeScript + Tailwind CSS + Biome.

## Quick Start

```bash
pnpm install
pnpm dev
```

The frontend expects the retrieval API on `:8000` and ingestion API on `:8001`.
Override with environment variables:

```bash
NEXT_PUBLIC_RETRIEVAL_API_URL=http://localhost:8000 \
NEXT_PUBLIC_INGESTION_API_URL=http://localhost:8001 \
pnpm dev
```

## Structure

```
app/
  page.tsx            Chat workbench (main page)
  compare/page.tsx    A/B model comparison
  layout.tsx          Root layout with sidebar + theme
  globals.css         Tailwind base styles

src/
  components/ui/      Shared UI primitives (shadcn/ui)
  features/
    chat/             Chat workbench, messages, file upload
    ingestion/        Ingestion status display, upload dialog
    model-compare/    Model picker + A/B comparison workbench
  lib/api/            API client with typed request/response helpers
  hooks/              Shared React hooks
```

## Building

```bash
pnpm build           # TypeScript check + production build
pnpm lint            # Biome lint
pnpm format          # Biome format
```

## API Endpoints

The API client in `src/lib/api/` calls these endpoints:

| Method | Route | Service |
|--------|-------|---------|
| POST | `/query` | Retrieval API |
| GET | `/health` | Retrieval API |
| GET | `/config` | Retrieval API |
| POST | `/config` | Retrieval API |
| GET | `/providers` | Retrieval API |
| GET | `/evaluate/{id}` | Retrieval API |
| POST | `/upload` | Ingestion API |
| GET | `/status` | Ingestion API |
| GET | `/files` | Ingestion API |
| POST | `/reindex` | Ingestion API |
| POST | `/ingest/target` | Ingestion API |
| POST | `/status/index` | Ingestion API |
| DELETE | `/documents/{name}` | Ingestion API |

See `src/lib/api/types.ts` for the full request/response type definitions.
