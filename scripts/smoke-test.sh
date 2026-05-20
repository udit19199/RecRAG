#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# RecRAG Smoke Test — verify the deployment is working end-to-end.
#
# Usage:
#   bash scripts/smoke-test.sh                    # localhost defaults
#   bash scripts/smoke-test.sh http://my-vm:8000  # custom host
#
# Tests:
#   1. Ingestion API health
#   2. Retrieval API health
#   3. Frontend responds
#   4. Upload a test PDF → ingestion
#   5. Query the retrieval API
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'
ok()    { echo -e "${GREEN}[PASS]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()   { echo -e "${RED}[FAIL]${NC} $1"; }

# ── Config ────────────────────────────────────────────────────────────────────
INGESTION_BASE="${1:-http://localhost:8001}"
RETRIEVAL_BASE="${2:-http://localhost:8000}"
FRONTEND_BASE="${3:-http://localhost:3000}"

# Strip trailing slashes
INGESTION_BASE="${INGESTION_BASE%/}"
RETRIEVAL_BASE="${RETRIEVAL_BASE%/}"
FRONTEND_BASE="${FRONTEND_BASE%/}"

PASSED=0
FAILED=0
TOTAL=0

run_test() {
    TOTAL=$((TOTAL + 1))
    local name="$1"
    shift
    if "$@" &>/dev/null; then
        ok "$name"
        PASSED=$((PASSED + 1))
    else
        err "$name"
        FAILED=$((FAILED + 1))
    fi
}

# ── Create a minimal test PDF ─────────────────────────────────────────────────
create_test_pdf() {
    # Creates a valid minimal PDF with visible text
    cat <<'PDFEOF'
%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 24 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000360 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
437
%%EOF
PDFEOF
}

TEST_PDF=$(mktemp)
trap 'rm -f "$TEST_PDF"' EXIT
create_test_pdf > "$TEST_PDF"

echo -e "${CYAN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}  RecRAG Smoke Test${NC}"
echo -e "${CYAN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Ingestion API: ${BOLD}$INGESTION_BASE${NC}"
echo -e "  Retrieval API: ${BOLD}$RETRIEVAL_BASE${NC}"
echo -e "  Frontend:      ${BOLD}$FRONTEND_BASE${NC}"
echo ""

# ── Test 1: Ingestion API health ─────────────────────────────────────────────
run_test "Ingestion API /health" \
    curl -sf "$INGESTION_BASE/health"

# ── Test 2: Retrieval API health ─────────────────────────────────────────────
run_test "Retrieval API /health" \
    curl -sf "$RETRIEVAL_BASE/health"

# ── Test 3: Frontend responds ────────────────────────────────────────────────
run_test "Frontend responds" \
    curl -sf -o /dev/null "$FRONTEND_BASE"

# ── Test 4: Upload test PDF to ingestion ─────────────────────────────────────
run_test "Upload PDF → /upload" \
    curl -sf -X POST "$INGESTION_BASE/upload" \
        -F "files=@$TEST_PDF" \
        -F "extraction_mode=text_only" \
        -o /dev/null

# Wait for ingestion to finish (poll /status)
echo -e "  ${CYAN}[wait]${NC} Waiting for ingestion to complete..."
for i in $(seq 1 30); do
    status=$(curl -sf "$INGESTION_BASE/status" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "processing")
    if [ "$status" = "complete" ]; then
        ok "Ingestion completed"
        PASSED=$((PASSED + 1))
        TOTAL=$((TOTAL + 1))
        break
    fi
    if [ "$status" = "error" ]; then
        err "Ingestion failed"
        FAILED=$((FAILED + 1))
        TOTAL=$((TOTAL + 1))
        break
    fi
    sleep 2
done

# ── Test 5: Query the retrieval API ──────────────────────────────────────────
run_test "Query /query returns response" \
    curl -sf -X POST "$RETRIEVAL_BASE/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "Hello"}' \
        -o /dev/null

# ── Test 6: Query returns JSON with expected fields ──────────────────────────
QUERY_RESULT=$(curl -sf -X POST "$RETRIEVAL_BASE/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Hello"}' 2>/dev/null || echo "")
if echo "$QUERY_RESULT" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    assert 'response' in d, 'missing response'
    assert 'context' in d, 'missing context'
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    ok "/query returns valid JSON with response + context"
    PASSED=$((PASSED + 1))
else
    err "/query response is missing expected fields"
    FAILED=$((FAILED + 1))
fi
TOTAL=$((TOTAL + 1))

# ── Test 7: Ingestion /files lists the uploaded PDF ──────────────────────────
run_test "Ingestion /files lists uploaded PDF" \
    curl -sf "$INGESTION_BASE/files" 2>/dev/null | python3 -c "
import sys, json
files = json.load(sys.stdin).get('files', [])
assert any('test' in f.lower() for f in files), 'test PDF not found in file list'
" 2>/dev/null

# ── Test 8: Retrieval API /providers returns structure ───────────────────────
run_test "Retrieval /providers returns provider info" \
    curl -sf "$RETRIEVAL_BASE/providers" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'embedders' in d
assert 'llms' in d
" 2>/dev/null

# ── Test 9: Retrieval API /config returns current config ─────────────────────
run_test "Retrieval /config returns embedding + llm" \
    curl -sf "$RETRIEVAL_BASE/config" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'embedding' in d
assert 'llm' in d
" 2>/dev/null

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}Passed: ${PASSED}${NC}  ${RED}Failed: ${FAILED}${NC}  Total: ${TOTAL}"
if [ "$FAILED" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}All smoke tests passed!${NC}"
else
    echo -e "  ${RED}${BOLD}Some smoke tests failed.${NC}"
    echo -e "  Run manual checks:"
    echo -e "    ${COMPOSE_CMD:-docker compose} logs api-ingestion  # Check ingestion logs"
    echo -e "    ${COMPOSE_CMD:-docker compose} logs api-retrieval # Check retrieval logs"
fi
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""

exit "$FAILED"
