#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# RecRAG smoke test — validates deployed pods and health endpoints.
#
# Usage:
#   bash scripts/smoke_test.sh                    # default namespace "recrag"
#   NAMESPACE=recrag bash scripts/smoke_test.sh   # explicit namespace
#
# Exit code: 0 if all checks pass, 1 otherwise.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NAMESPACE="${NAMESPACE:-recrag}"
PASS=0
FAIL=0

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[info]${NC} $1"; }
ok()    { echo -e "${GREEN}[ ok ]${NC} $1"; }
err()   { echo -e "${RED}[fail]${NC} $1"; }

check_pod() {
    local deploy=$1
    local pod
    pod=$(kubectl get pod -n "$NAMESPACE" -l "app=$deploy" -o name 2>/dev/null | head -1 || echo "")
    if [ -z "$pod" ]; then
        err "No pod found for deployment '$deploy'"
        FAIL=$((FAIL + 1))
        return 1
    fi
    local status
    status=$(kubectl get "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
    if [ "$status" = "Running" ]; then
        ok "Pod '$deploy' is Running"
        PASS=$((PASS + 1))
    else
        err "Pod '$deploy' status: $status"
        FAIL=$((FAIL + 1))
    fi
}

check_health() {
    local deploy=$1
    local port=$2
    local url="http://localhost:${port}/health"
    info "Checking health for $deploy at $url"
    # Try port-forward first; if we can reach directly, even better
    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" 2>/dev/null || echo "000")
    if [ "$status_code" = "200" ]; then
        ok "Health check passed for $deploy (HTTP $status_code)"
        PASS=$((PASS + 1))
    else
        err "Health check failed for $deploy (HTTP $status_code)"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "═══════════════════════════════════════════════"
echo "  RecRAG Smoke Tests"
echo "═══════════════════════════════════════════════"
echo ""

# ── 1. Pod health checks ─────────────────────────────────────────────────────
info "Checking pod status..."
check_pod "api-ingestion"
check_pod "api-retrieval"
check_pod "frontend"

echo ""

# ── 2. Health endpoint checks ────────────────────────────────────────────────
info "Checking health endpoints (requires port-forwards or direct access)..."
check_health "ingestion-api" 8001
check_health "retrieval-api" 8000

echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════"
echo "  Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "═══════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    info "Troubleshooting tips:"
    echo "  kubectl get pods -n $NAMESPACE"
    echo "  kubectl describe pod -n $NAMESPACE -l app=api-ingestion"
    echo "  kubectl logs -n $NAMESPACE -l app=api-ingestion"
    echo "  kubectl logs -n $NAMESPACE -l app=api-retrieval"
    echo ""
    exit 1
fi

exit 0
