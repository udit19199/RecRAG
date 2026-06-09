#!/usr/bin/env bash
# After agent work, nudge a fix loop when changed frontend files still fail React Doctor.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

if ! {
	git diff --name-only HEAD -- frontend/ 2>/dev/null
	git diff --cached --name-only -- frontend/ 2>/dev/null
} | grep -q .; then
	exit 0
fi

output_file="$(mktemp "${TMPDIR:-/tmp}/react-doctor-stop.XXXXXX")"
trap 'rm -f "$output_file"' EXIT

set +e
(cd frontend && npx react-doctor@latest --yes --diff --blocking warning --no-score >"$output_file" 2>&1)
status=$?
set -e

if [ "$status" -ne 0 ]; then
	node - "$output_file" <<'NODE'
const fs = require("node:fs");
const output = fs.readFileSync(process.argv[2], "utf8");
console.log(
	JSON.stringify({
		followup_message: `React Doctor found issues in changed frontend files. Fix these before finishing:\n\n${output}`,
	}),
);
NODE
fi
