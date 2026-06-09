#!/usr/bin/env bash
# Cursor beforeShellExecution: block git commit when staged frontend files fail React Doctor.

set -euo pipefail

hook_dir="$(CDPATH= cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--git-hook" ]; then
	exec "$hook_dir/react-doctor-check-staged.sh"
fi

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

output_file="$(mktemp "${TMPDIR:-/tmp}/react-doctor-commit.XXXXXX")"
trap 'rm -f "$output_file"' EXIT

set +e
"$hook_dir/react-doctor-check-staged.sh" >"$output_file" 2>&1
status=$?
set -e

if [ "$status" -ne 0 ]; then
	output="$(cat "$output_file")"
	node - "$output" <<'NODE'
const fs = require("node:fs");
const output = fs.readFileSync(0, "utf8");
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
