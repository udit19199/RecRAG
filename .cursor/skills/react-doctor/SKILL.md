---
name: react-doctor
description: >-
  Scan frontend React/Next.js code with React Doctor before committing or when
  the user asks to run /doctor, triage diagnostics, or improve frontend health.
---

# React Doctor (RecRAG frontend)

Use for `frontend/` only. Manual full scans: `cd frontend && pnpm doctor`.

## Before committing frontend changes

Staged `frontend/` files are checked before commit:

- **Git** (Source Control UI or terminal): `.cursor/hooks/pre-commit`
- **Cursor agent shell**: `beforeShellExecution` on `git commit`

The check is skipped when no staged files are under `frontend/`.

If a commit is blocked, run:

```bash
cd frontend && npx react-doctor@latest --yes --staged --no-dead-code --blocking warning --verbose
```

Fix warnings and errors in staged files, then retry the commit.

## After frontend edits (agent stop hook)

When the agent finishes, the `stop` hook may request a follow-up if changed
frontend files still fail `react-doctor --diff --blocking warning`. Apply those
fixes before considering the task done.

## Full health check (manual)

```bash
cd frontend && pnpm doctor
```

Use for baseline score work or periodic audits. CI also runs diff-based checks on PRs.

## Useful flags

| Flag | Purpose |
| --- | --- |
| `--staged` | Scan git-staged files only |
| `--no-dead-code` | Skip unused-file/dependency analysis (use with `--staged`) |
| `--diff` | Scan files changed vs base branch |
| `--verbose` | Show file paths and line numbers |
| `--blocking warning` | Exit non-zero on warnings or errors |
| `--score` | Print only the 0–100 health score |
