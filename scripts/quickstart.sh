#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# RecRAG Quickstart — deploy on a fresh VM with one command.
#
# Usage:
#   bash scripts/quickstart.sh
#
# What it does:
#   1. Installs Docker if missing
#   2. Creates .env from .env.example if missing (you edit the keys)
#   3. Builds images and starts everything via docker compose
#   4. Shows you where to access it
#
# Requirements:
#   - Git (to clone the repo)
#   - API keys (set them in .env when prompted)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${CYAN}[info]${NC} $1"; }
ok()    { echo -e "${GREEN}[ ok ]${NC} $1"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $1"; }
err()   { echo -e "${RED}[error]${NC} $1"; }

cd "$(dirname "$0")/.."

# ── Step 1: Check / install Docker ────────────────────────────────────────────
info "Checking Docker..."
if ! command -v docker &>/dev/null; then
    warn "Docker not found. Installing..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    ok "Docker installed. You may need to log out and back in for group changes."
    DOCKER_CMD="sudo docker"
else
    DOCKER_CMD="docker"
    if ! docker info &>/dev/null; then
        if sudo docker info &>/dev/null; then
            DOCKER_CMD="sudo docker"
        else
            err "Docker daemon not running. Start it: sudo systemctl start docker"
            exit 1
        fi
    fi
fi
ok "Docker ready ($DOCKER_CMD)"

# Detect compose command
if ${DOCKER_CMD} compose version &>/dev/null; then
    COMPOSE_CMD="${DOCKER_CMD} compose"
elif command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="${DOCKER_CMD}-compose"
else
    err "docker compose not found."
    err "Install: sudo apt-get install docker-compose-plugin"
    exit 1
fi
ok "docker compose ready"

# ── Step 2: .env file ─────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo ""
        warn "================================================"
        warn "  .env created from .env.example"
        warn "  EDIT IT with your API keys:"
        warn "    nano .env"
        warn ""
        warn "  At minimum, set OPENAI_API_KEY"
        warn "================================================"
        echo ""
        if [ -t 0 ]; then
            read -rp "Press Enter after editing .env (or Ctrl+C to abort)... "
        else
            err "Edit .env with your API keys, then re-run."
            exit 1
        fi
    else
        err "No .env or .env.example found."
        err "Create .env with at least: OPENAI_API_KEY=sk-..."
        exit 1
    fi
fi
ok ".env found"

# ── Step 3: Build and start everything ────────────────────────────────────────
info "Building images and starting services (first build takes a few minutes)..."
${COMPOSE_CMD} up -d --build

# ── Step 4: Wait for health checks ────────────────────────────────────────────
info "Waiting for services to become healthy..."
for svc in api-ingestion api-retrieval frontend; do
    container="recrag-${svc}"
    for i in $(seq 1 18); do
        status=$(${DOCKER_CMD} inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "starting")
        if [ "$status" = "healthy" ]; then
            ok "$container healthy"
            break
        fi
        if [ "$status" = "unhealthy" ]; then
            warn "$container unhealthy — check logs: ${COMPOSE_CMD} logs $svc"
            break
        fi
        sleep 5
    done
done

# ── Step 5: Summary ───────────────────────────────────────────────────────────
HOST_IP=$(curl -s ifconfig.me 2>/dev/null || echo "localhost")

echo ""
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  RecRAG is running!${NC}"
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Frontend:${NC}     http://${HOST_IP}:3000"
echo -e "  ${CYAN}Retrieval API:${NC} http://localhost:8000/docs"
echo -e "  ${CYAN}Ingestion API:${NC} http://localhost:8001/docs"
echo ""
echo -e "  ${YELLOW}Next steps:${NC}"
echo -e "    1. Open the frontend and upload a PDF"
echo -e "    2. Or upload via API:"
echo -e "       curl -X POST http://localhost:8001/upload -F \"files=@doc.pdf\""
echo ""
echo -e "  ${YELLOW}Useful commands:${NC}"
echo -e "    ${COMPOSE_CMD} logs -f              # All logs"
echo -e "    ${COMPOSE_CMD} logs -f api-retrieval  # Retrieval logs"
echo -e "    ${COMPOSE_CMD} down                 # Stop everything"
echo -e "    ${COMPOSE_CMD} up -d                # Start again"
echo ""
