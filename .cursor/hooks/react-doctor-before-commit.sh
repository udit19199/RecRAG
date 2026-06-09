#!/usr/bin/env bash
# Block git commit when staged frontend files fail React Doctor (--staged --blocking warning).
# Replaces a non-blocking git pre-commit hook with agent-visible feedback.

set -euo pipefail

input="$(cat)"
command="$(
	printf '%s' "$input" | node -e "
let data = '';
process.stdin.on('data', (chunk) => { data += chunk; });
process.stdin.on('end', () => {
	try {
		const payload = JSON.parse(data);
		process.stdout.write(payload.command || '');
	} catch {
		process.stdout.write('');
	}
});
"
)"

if ! printf '%s' "$command" | grep -qE 'git commit'; then
	printf '%s\n' '{"permission":"allow"}'
	exit 0
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

if ! git diff --cached --name-only -- frontend/ | grep -q .; then
	printf '%s\n' '{"permission":"allow"}'
	exit 0
fi

output_file="$(mktemp "${TMPDIR:-/tmp}/react-doctor-commit.XXXXXX")"
trap 'rm -f "$output_file"' EXIT

set +e
(cd frontend && npx react-doctor@latest --yes --staged --blocking warning >"$output_file" 2>&1)
status=$?
set -e

if [ "$status" -ne 0 ]; then
	node - "$output_file" <<'NODE'
const fs = require("node:fs");
const output = fs.readFileSync(process.argv[2], "utf8");
console.log(
	JSON.stringify({
		permission: "deny",
		user_message:
			"React Doctor found issues in staged frontend files. The agent can fix them before committing.",
		agent_message: `React Doctor blocked this git commit. Fix these staged frontend regressions, then retry the commit:\n\n${output}`,
	}),
);
NODE
	exit 0
fi

printf '%s\n' '{"permission":"allow"}'
