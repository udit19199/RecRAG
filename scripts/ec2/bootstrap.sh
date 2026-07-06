#!/usr/bin/env bash
# Bootstrap a single EC2 host for RecRAG Docker Compose deploys.
#
# This script is intended for the one-time server setup before GitHub Actions
# starts shipping fresh container images to the instance.

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/recrag}"
REPO_URL="${REPO_URL:-https://github.com/udit19199/RecRAG.git}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-main}"

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

ensure_package() {
    local package="$1"
    if ! dpkg -s "$package" >/dev/null 2>&1; then
        info "Installing package: $package"
        sudo apt-get install -y "$package"
    fi
}

install_docker() {
    if command -v docker >/dev/null 2>&1; then
        ok "Docker already installed"
        return
    fi

    info "Installing Docker"
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER" || true
    sudo systemctl enable --now docker
    ok "Docker installed"
}

install_compose_plugin() {
    if docker compose version >/dev/null 2>&1; then
        ok "Docker Compose plugin already installed"
        return
    fi

    info "Installing Docker Compose plugin"
    sudo apt-get update
    ensure_package docker-compose-plugin
    ok "Docker Compose plugin installed"
}

install_aws_cli() {
    if command -v aws >/dev/null 2>&1; then
        ok "AWS CLI already installed"
        return
    fi

    info "Installing AWS CLI"
    sudo apt-get update
    ensure_package unzip
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$tmpdir/awscliv2.zip"
    unzip -q "$tmpdir/awscliv2.zip" -d "$tmpdir"
    sudo "$tmpdir/aws/install" --update
    ok "AWS CLI installed"
}

prepare_app_dir() {
    info "Preparing application directory at $APP_DIR"
    sudo mkdir -p "$APP_DIR"
    sudo chown "$USER":"$USER" "$APP_DIR"

    if [ ! -d "$APP_DIR/.git" ]; then
        info "Cloning repository"
        git clone --branch "$DEPLOY_BRANCH" "$REPO_URL" "$APP_DIR"
    else
        info "Refreshing repository checkout"
        git -C "$APP_DIR" fetch origin "$DEPLOY_BRANCH"
        git -C "$APP_DIR" checkout "$DEPLOY_BRANCH"
        git -C "$APP_DIR" pull origin "$DEPLOY_BRANCH"
    fi
    ok "Repository is ready"
}

prepare_env() {
    if [ ! -f "$APP_DIR/.env" ]; then
        cp "$APP_DIR/.env.example" "$APP_DIR/.env"
        warn "Created $APP_DIR/.env from .env.example"
        warn "Edit it before the first deployment."
    else
        ok "Found existing $APP_DIR/.env"
    fi
}

main() {
    sudo apt-get update
    ensure_package git
    ensure_package curl

    install_docker
    install_compose_plugin
    install_aws_cli
    prepare_app_dir
    prepare_env

    cat <<EOF

${GREEN}Bootstrap complete.${NC}

Next steps:
  1. Edit ${APP_DIR}/.env with your provider keys and any Milvus settings.
  2. Attach an IAM role with ECR read access to this EC2 instance.
  3. Point GitHub Actions deploy secrets at this host.
  4. Use docker-compose.ec2.yml for production deploys.
EOF
}

main "$@"
