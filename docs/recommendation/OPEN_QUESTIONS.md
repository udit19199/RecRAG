# Recommendation System — Open Questions (TBD)

Items not yet decided. **Resolved decisions live in
[DESIGN_DECISIONS.md](./DESIGN_DECISIONS.md)** — consult that file first; update this
list when a question is closed (move the entry to DESIGN_DECISIONS as Dnn).

---

## Product & UX

| ID | Question | Notes |
|----|----------|-------|
| UX-1 | Wizard vs chat entry — exact screen flow and relationship to current 3-step onboarding | Dual intake agreed; wireframe TBD |
| UX-2 | Fast-path UI (no corpus) — fields, copy, CTA to upload | Preliminary recommendation only; not exportable (D12) |
| UX-3 | Re-benchmark prompt — exact copy and layout for yes / no / later + duration | Behavior decided (D22, D29); UI TBD |
| UX-4 | Human-readable report alongside JSON export? | JSON blueprint required (D13); HTML/markdown report optional |
| UX-5 | Customizer LLM default provider/model | User-selectable with sensible default (D17); default TBD |

---

## Benchmark & scoring

| ID | Question | Notes |
|----|----------|-------|
| BM-1 | Constraint → metric weight mapping | e.g. citations → faithfulness + context_precision |
| BM-2 | `FeasibilityScore` formula and warning thresholds | Pre-benchmark only; not export gate |
| BM-3 | Suite composition without industry templates | Domain-generated vs user query split (D24: no industry packs v1) |
| BM-4 | Architecture probes — count per `RagArchitecture`, authored vs generated | Two-dimensional suite agreed (D8); probes TBD |
| BM-5 | ~~Parallel vs sequential indexing~~ | **Resolved D35** — sequential index, parallel benchmark |
| BM-6 | ~~Candidate index failure mid-run~~ | **Resolved D35** — skip candidate, continue |

---

## Technical contracts

| ID | Question | Notes |
|----|----------|-------|
| TC-1 | ~~`PipelineBlueprint` JSON~~ | **Resolved D33** — `orchestration.models.PipelineBlueprint` |
| TC-2 | Collection naming — suffix/hash for chunk policy + architecture | **Implemented** `orchestration/collection_naming.py` |
| TC-3 | Corpus analysis — infer `document_modality` from uploaded files | Heuristics vs sampling TBD |
| TC-4 | Economy tier minimum capability gates | e.g. min dimensions, context window |
| TC-5 | Mid tier — precise price-ranking rule | D25: flagship = most expensive; mid rule fuzzy |
| TC-6 | Multi-provider blueprints allowed? | e.g. OpenAI embed + Gemini LLM in one export |
| TC-7 | ~~Orchestrator API routes~~ | **Resolved D34** — `app/orchestrator/main.py` |
| TC-8 | ~~Corpus handoff~~ | **Resolved D36** — shared ingestion `/files` + `/ingest/target` |

---

## Orchestrator & infrastructure

| ID | Question | Notes |
|----|----------|-------|
| INF-1 | ~~`make dev` orchestrator~~ | **Done** — `make orchestrator`, `make dev` includes `:8002` |
| INF-2 | Clerk — org required or personal-workspace fallback? | B2B orgs = workspace (D31) |
| INF-3 | `users` table — `clerk_user_id` only vs sync profile from JWT | |
| INF-4 | When to integrate Clerk vs stub `workspace_id` for early backend work | Sequencing TBD |
| INF-5 | Service auth — Orchestrator → ingestion/retrieval via `REC_RAG_API_KEY` | Agreed in spirit (D31); edge cases TBD |

---

## Deferred (decide when research picks direction)

| ID | Question | Notes |
|----|----------|-------|
| DEF-1 | Industry benchmark templates — which vertical first | No templates v1 (D24) |
| DEF-2 | `graph` / `agentic` — implement or remove | Commented out (D15) |
| DEF-3 | Prod behavior after research archived | Candidate min/max, backfill rules may differ from dev |
| DEF-4 | Clerk webhooks — org deletion / GDPR cleanup | Not required v1 (D32) |
| DEF-5 | Reranker in `citation` architecture | |
| DEF-6 | Video `document_modality` — loader support | Not in codebase today |
| DEF-7 | Clerk webhooks for membership roster / billing | |

---

## Suggested resolution order

1. TC-1, TC-7, TC-8 — blueprint JSON + Orchestrator API (unblocks implementation)
2. BM-5, BM-6 — benchmark runner behavior
3. TC-2, TC-3 — collection naming + corpus analysis
4. UX-1–UX-3 — intake and post-benchmark flows
5. INF-1–INF-4 — dev ergonomics and Clerk wiring
