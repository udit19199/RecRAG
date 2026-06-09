#!/usr/bin/env bash
#
# bump-version.sh — bump semver across the monorepo and create a git tag.
#
# Usage:
#   bash scripts/bump-version.sh patch   # 0.1.0 → 0.1.1
#   bash scripts/bump-version.sh minor   # 0.1.0 → 0.2.0
#   bash scripts/bump-version.sh major   # 0.1.0 → 1.0.0
#   bash scripts/bump-version.sh 2.3.4   # set explicit version
#
# What it does:
#   1. Reads current version from VERSION (single source of truth).
#   2. Computes the next version.
#   3. Updates VERSION, pyproject.toml, and frontend/package.json.
#   4. Stages the changed files.
#   5. Creates an annotated git tag (v<version>).
#
# Requirements:
#   - Must be run from the repo root.
#   - Working tree must be clean (no uncommitted changes).

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────────

die() { echo -e "${RED}error:${NC} $*" >&2; exit 1; }
info() { echo -e "${CYAN}→${NC} $*"; }
ok() { echo -e "${GREEN}✓${NC} $*"; }

bump_component() {
  local current="$1" component="$2"
  local major minor patch
  IFS='.' read -r major minor patch <<< "$current"
  case "$component" in
    major) echo "$((major + 1)).0.0" ;;
    minor) echo "${major}.$((minor + 1)).0" ;;
    patch) echo "${major}.${minor}.$((patch + 1))" ;;
    *) die "unknown component: $component" ;;
  esac
}

validate_semver() {
  local v="$1"
  if ! [[ "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    die "'$v' is not valid semver (expected MAJOR.MINOR.PATCH)"
  fi
}

# ── Preflight ─────────────────────────────────────────────────────────────────

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repo"
cd "$REPO_ROOT"

[[ -f VERSION ]]          || die "VERSION file not found at repo root"
[[ -f pyproject.toml ]]   || die "pyproject.toml not found"
[[ -f frontend/package.json ]] || die "frontend/package.json not found"

if ! git diff --quiet || ! git diff --cached --quiet; then
  die "working tree is not clean — commit or stash changes first"
fi

# ── Compute next version ─────────────────────────────────────────────────────

CURRENT=$(cat VERSION | tr -d '[:space:]')
validate_semver "$CURRENT"

ARG="${1:-}"
if [[ -z "$ARG" ]]; then
  die "usage: bash scripts/bump-version.sh {major|minor|patch|X.Y.Z}"
elif [[ "$ARG" =~ ^(major|minor|patch)$ ]]; then
  NEXT=$(bump_component "$CURRENT" "$ARG")
else
  NEXT="$ARG"
  validate_semver "$NEXT"
fi

if [[ "$NEXT" == "$CURRENT" ]]; then
  die "next version ($NEXT) is the same as current ($CURRENT)"
fi

info "Bumping $CURRENT → $NEXT"

# ── Update files ──────────────────────────────────────────────────────────────

# VERSION
echo "$NEXT" > VERSION
ok "VERSION"

# pyproject.toml — replace the first `version = "..."` under [project]
sed -i.bak "s/^version = \"${CURRENT}\"/version = \"${NEXT}\"/" pyproject.toml
rm -f pyproject.toml.bak
ok "pyproject.toml"

# frontend/package.json — replace the first `"version": "..."`
sed -i.bak "s/\"version\": \"${CURRENT}\"/\"version\": \"${NEXT}\"/" frontend/package.json
rm -f frontend/package.json.bak
ok "frontend/package.json"

# ── Git tag ───────────────────────────────────────────────────────────────────

TAG="v${NEXT}"
git add VERSION pyproject.toml frontend/package.json
git commit -m "chore(release): ${TAG}" --no-verify
git tag -a "$TAG" -m "Release ${TAG}"

ok "Created tag ${TAG}"
echo ""
echo "Next steps:"
echo "  git push origin main --tags"
