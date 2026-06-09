#!/usr/bin/env bash
# Scan staged frontend files with React Doctor.
# Exit 0 when skipped or clean; exit 1 when diagnostics fail.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

if ! git diff --cached --name-only -- frontend/ | grep -q .; then
	exit 0
fi

output_file="$(mktemp "${TMPDIR:-/tmp}/react-doctor-staged.XXXXXX")"
trap 'rm -f "$output_file"' EXIT

set +e
(cd frontend && npx react-doctor@latest --yes --staged --no-dead-code --blocking warning >"$output_file" 2>&1)
status=$?
set -e

if [ "$status" -ne 0 ]; then
	cat "$output_file" >&2
	exit 1
fi
