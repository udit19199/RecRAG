#!/usr/bin/env bash
# Pull published images and restart the EC2 Docker Compose stack in place.

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/recrag}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.ec2.yml}"
AWS_REGION="${AWS_REGION:-}"
API_IMAGE="${API_IMAGE:-}"
FRONTEND_IMAGE="${FRONTEND_IMAGE:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info() {
    echo -e "${CYAN}[info]${NC} $1"
}

ok() {
    echo -e "${GREEN}[ ok ]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[warn]${NC} $1"
}

err() {
    echo -e "${RED}[error]${NC} $1"
}

require_cmd() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        err "Missing required command: $command_name"
        exit 1
    fi
}

require_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        err "Required file not found: $path"
        exit 1
    fi
}

maybe_login_ecr() {
    if [[ -z "$API_IMAGE$FRONTEND_IMAGE" ]]; then
        return
    fi

    if [[ "$API_IMAGE$FRONTEND_IMAGE" != *".amazonaws.com/"* ]]; then
        return
    fi

    require_cmd aws
    if [ -z "$AWS_REGION" ]; then
        err "AWS_REGION must be set when deploying images from ECR"
        exit 1
    fi

    local registry
    registry="${API_IMAGE%%/*}"
    if [ -z "$registry" ] && [ -n "$FRONTEND_IMAGE" ]; then
        registry="${FRONTEND_IMAGE%%/*}"
    fi
    if [ -n "$registry" ]; then
        info "Logging in to ECR registry $registry"
        aws ecr get-login-password --region "$AWS_REGION" | \
            docker login --username AWS --password-stdin "$registry"
        ok "ECR login succeeded"
    fi
}

main() {
    require_cmd docker

    if ! docker compose version >/dev/null 2>&1; then
        err "Docker Compose plugin is not installed"
        exit 1
    fi

    require_file "$APP_DIR/$COMPOSE_FILE"
    require_file "$APP_DIR/.env"
    require_file "$APP_DIR/config.toml"

    cd "$APP_DIR"
    maybe_login_ecr

    info "Pulling latest images"
    docker compose -f "$COMPOSE_FILE" pull

    info "Restarting RecRAG stack"
    docker compose -f "$COMPOSE_FILE" up -d --remove-orphans

    info "Waiting for container health checks"
    sleep 10
    docker compose -f "$COMPOSE_FILE" ps

    info "Pruning dangling Docker images"
    docker image prune -f >/dev/null 2>&1 || warn "Docker image prune failed"

    ok "Deployment finished"
}

main "$@"
