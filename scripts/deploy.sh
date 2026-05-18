#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# RecRAG deploy — build images, import into k3s, apply Kustomize, wait.
#
# Usage:
#   bash scripts/deploy.sh
#   OVERLAY=dev bash scripts/deploy.sh
#   DOMAIN=recrag.mycompany.com bash scripts/deploy.sh
#
# Requirements:
#   - .env file present with required keys
#   - Docker installed (auto-installed if missing)
#   - k3s installed (auto-installed if missing)
#   - kubectl, kustomize
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

OVERLAY="${OVERLAY:-prod}"
NAMESPACE="${NAMESPACE:-recrag}"
DOMAIN="${DOMAIN:-example.com}"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${CYAN}[info]${NC} $1"; }
ok()    { echo -e "${GREEN}[ ok ]${NC} $1"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $1"; }
err()   { echo -e "${RED}[error]${NC} $1"; }

# ── Cleanup handler ──────────────────────────────────────────────────────────
CLEANUP_FILES=()
cleanup() {
    if [ ${#CLEANUP_FILES[@]} -gt 0 ]; then
        rm -rf "${CLEANUP_FILES[@]}"
    fi
}
trap cleanup EXIT

# ── Step 0: Validate .env (read-only, not sourced) ───────────────────────────
validate_env() {
    if [ ! -f .env ]; then
        err ".env file not found. Create one from .env.example:"
        err "  cp .env.example .env"
        exit 1
    fi

    local missing=0
    # Read keys from .env file without sourcing
    local env_content
    env_content=$(grep -v '^\s*#' .env | grep -v '^\s*$' || true)

    # Check at least one LLM provider key exists (check the keys without exposing values)
    local has_openai=0 has_nvidia=0 has_ollama=0 has_milvus=0
    while IFS='=' read -r key value; do
        key="$(echo "$key" | tr -d '[:space:]')"
        value="$(echo "$value" | tr -d '[:space:]')"
        case "$key" in
            OPENAI_API_KEY)   [ -n "$value" ] && has_openai=1 ;;
            NVIDIA_API_KEY)   [ -n "$value" ] && has_nvidia=1 ;;
            OLLAMA_API_KEY)   [ -n "$value" ] && has_ollama=1 ;;
            MILVUS_HOST)      [ -n "$value" ] && has_milvus=1 ;;
        esac
    done <<< "$env_content"

    if [ "$has_openai" -eq 0 ] && [ "$has_nvidia" -eq 0 ] && [ "$has_ollama" -eq 0 ]; then
        warn "No LLM provider API key found in .env."
        warn "  Set at least one of: OPENAI_API_KEY, NVIDIA_API_KEY, or OLLAMA_API_KEY"
        missing=1
    fi

    if [ "$has_milvus" -eq 0 ]; then
        warn "MILVUS_HOST not set in .env — Milvus connection will fail."
        missing=1
    fi

    return "$missing"
}

info "Validating .env..."
validate_env || true

# ── Step 1: Install Docker if missing ─────────────────────────────────────────
install_docker() {
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo systemctl enable --now docker
    ok "Docker installed"
}

if ! command -v docker &>/dev/null; then
    install_docker
else
    ok "Docker already installed"
fi

# ── Step 2: Install k3s if missing ────────────────────────────────────────────
install_k3s() {
    info "Installing k3s (lightweight Kubernetes)..."
    curl -sfL https://get.k3s.io | sh -s - \
        --disable=traefik \
        --write-kubeconfig-mode=644
    mkdir -p "$HOME/.kube"
    sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
    sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
    export KUBECONFIG="$HOME/.kube/config"
    ok "k3s installed. Waiting for cluster readiness..."
    sleep 10
    kubectl wait --for=condition=Ready node --all --timeout=60s 2>/dev/null || true
}

if ! command -v kubectl &>/dev/null; then
    install_k3s
else
    if ! kubectl cluster-info --request-timeout=5s &>/dev/null; then
        warn "kubectl found but cannot reach a cluster. Installing k3s..."
        if command -v k3s &>/dev/null; then
            info "k3s binary found — starting service..."
            sudo systemctl start k3s 2>/dev/null || sudo service k3s start 2>/dev/null || true
            sleep 5
        else
            install_k3s
        fi
    fi
fi
ok "Kubernetes cluster is reachable"

# ── Step 3: Install kustomize if missing ──────────────────────────────────────
if ! command -v kustomize &>/dev/null && ! kubectl kustomize --help &>/dev/null 2>&1; then
    info "Installing kustomize..."
    curl -sSLO https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.4.3/kustomize_v5.4.3_linux_amd64.tar.gz
    tar -xzf kustomize_v5.4.3_linux_amd64.tar.gz
    chmod +x kustomize && sudo mv kustomize /usr/local/bin/
    rm -f kustomize_v5.4.3_linux_amd64.tar.gz
    ok "kustomize installed"
fi

# ── Step 4: Build Docker images ───────────────────────────────────────────────
info "Building RecRAG API image..."
docker build -t recrag-api:latest -f Dockerfile --target api .

info "Building RecRAG frontend image..."
docker build -t recrag-frontend:latest -f frontend/Dockerfile \
    --build-arg NEXT_PUBLIC_RETRIEVAL_API_URL=http://retrieval.$DOMAIN \
    --build-arg NEXT_PUBLIC_INGESTION_API_URL=http://ingestion.$DOMAIN \
    frontend/

ok "Docker images built successfully"

# ── Step 5: Import images into k3s containerd ─────────────────────────────────
info "Importing images into k3s containerd..."
# k3s uses its own containerd instance; import via ctr or save+load
docker save recrag-api:latest | sudo k3s ctr images import - 2>/dev/null || \
    docker save recrag-api:latest | sudo ctr -n k8s.io images import - 2>/dev/null || \
    warn "Could not import API image into k3s containerd (will use imagePullPolicy: IfNotPresent)"

docker save recrag-frontend:latest | sudo k3s ctr images import - 2>/dev/null || \
    docker save recrag-frontend:latest | sudo ctr -n k8s.io images import - 2>/dev/null || \
    warn "Could not import frontend image into k3s containerd (will use imagePullPolicy: IfNotPresent)"

ok "Images imported into k3s containerd"

# ── Step 6: Create namespace + secrets ────────────────────────────────────────
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
ok "Namespace '$NAMESPACE' ready"

info "Creating recrag-secrets from .env..."
kubectl delete secret recrag-secrets -n "$NAMESPACE" --ignore-not-found

# Use --from-env-file to avoid leaking secrets into process listings
if [ -f .env ]; then
    kubectl create secret generic recrag-secrets \
        -n "$NAMESPACE" \
        --from-env-file=.env
    ok "Secret 'recrag-secrets' created from .env"
else
    # Create placeholder if .env doesn't exist
    kubectl create secret generic recrag-secrets \
        -n "$NAMESPACE" \
        --from-literal=placeholder=unset
    warn "Secret created with placeholder values (no .env found)"
fi

# ── Step 7: Create configmap ──────────────────────────────────────────────────
if [ -f config.toml ]; then
    kubectl create configmap recrag-config \
        -n "$NAMESPACE" \
        --from-file=config.toml=config.toml \
        --dry-run=client -o yaml | kubectl apply -f -
    ok "ConfigMap 'recrag-config' created"
else
    warn "config.toml not found — skipping ConfigMap creation"
fi

# ── Step 8: Prepare Kustomize manifests with imagePullPolicy: IfNotPresent ────
TMPDIR=$(mktemp -d)
CLEANUP_FILES+=("$TMPDIR")

# Copy overlay into temp dir and patch
if [ -d "k8s/overlays/$OVERLAY" ]; then
    cp -r "k8s/overlays/$OVERLAY"/* "$TMPDIR/" 2>/dev/null || true
fi
if [ -d "k8s/base" ]; then
    cp -r k8s/base/* "$TMPDIR/" 2>/dev/null || true
fi

# Write a kustomization that patches imagePullPolicy on all deployments
cat > "$TMPDIR/kustomization.yaml" <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
$(for f in "$TMPDIR"/*.yaml; do
    if [ "$(basename "$f")" != "kustomization.yaml" ]; then
        echo "  - $(basename "$f")"
    fi
done)

patches:
  - target:
      kind: Deployment
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/imagePullPolicy
        value: IfNotPresent
      - op: replace
        path: /spec/template/spec/containers/0/image
        value: recrag-api:latest
  - target:
      kind: Deployment
      name: frontend
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/imagePullPolicy
        value: IfNotPresent
      - op: replace
        path: /spec/template/spec/containers/0/image
        value: recrag-frontend:latest
EOF

# ── Step 9: Apply manifests ───────────────────────────────────────────────────
info "Applying Kustomize manifests..."
kubectl kustomize "$TMPDIR" | kubectl apply -n "$NAMESPACE" -f -

# ── Step 10: Create recrag-tunnel.service for port-forwards ───────────────────
info "Setting up recrag-tunnel.service for port-forwards..."
if command -v systemctl &>/dev/null; then
    sudo tee /etc/systemd/system/recrag-tunnel.service > /dev/null <<'SERVICEEOF'
[Unit]
Description=RecRAG port-forward tunnel
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=5
ExecStartPre=/bin/sleep 5
ExecStart=/usr/local/bin/kubectl port-forward -n recrag deployment/frontend 3000:3000
ExecStartPost=/usr/local/bin/kubectl port-forward -n recrag deployment/api-retrieval 8000:8000
ExecStartPost=/usr/local/bin/kubectl port-forward -n recrag deployment/api-ingestion 8001:8001

[Install]
WantedBy=multi-user.target
SERVICEEOF
    sudo systemctl daemon-reload
    sudo systemctl enable recrag-tunnel.service --now 2>/dev/null || true
    ok "recrag-tunnel.service created and started"
else
    warn "systemctl not available — skipping systemd service"
    info "Starting port-forwards in background..."
    nohup kubectl port-forward -n "$NAMESPACE" deployment/frontend 3000:3000 &
    nohup kubectl port-forward -n "$NAMESPACE" deployment/api-retrieval 8000:8000 &
    nohup kubectl port-forward -n "$NAMESPACE" deployment/api-ingestion 8001:8001 &
fi

# ── Step 11: Wait for rollout ─────────────────────────────────────────────────
info "Waiting for deployments to roll out..."
for deploy in api-ingestion api-retrieval frontend; do
    if kubectl get deployment "$deploy" -n "$NAMESPACE" &>/dev/null; then
        kubectl rollout status "deployment/$deploy" -n "$NAMESPACE" --timeout=5m || \
            warn "Rollout for $deploy timed out — check with: kubectl get pods -n $NAMESPACE"
    fi
done

# ── Step 12: Summary ──────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  RecRAG is deployed!${NC}"
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""

# Get pod status
echo -e "${CYAN}Pods:${NC}"
kubectl get pods -n "$NAMESPACE" -o wide
echo ""

echo -e "  ${CYAN}Frontend:${NC}     http://recrag.$DOMAIN"
echo -e "  ${CYAN}Retrieval API:${NC} http://retrieval.$DOMAIN"
echo -e "  ${CYAN}Ingestion API:${NC} http://ingestion.$DOMAIN"
echo ""

INGRESS_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
if [ -n "$INGRESS_IP" ]; then
    echo -e "  Ingress IP: ${BOLD}$INGRESS_IP${NC}"
    echo -e "  Add these to your DNS or /etc/hosts:"
    echo -e "    $INGRESS_IP  recrag.$DOMAIN retrieval.$DOMAIN ingestion.$DOMAIN"
fi

echo ""
echo -e "  ${YELLOW}Useful commands:${NC}"
echo -e "    kubectl get pods -n $NAMESPACE"
echo -e "    kubectl logs -n $NAMESPACE deployment/api-retrieval -f"
echo -e "    kubectl logs -n $NAMESPACE deployment/api-ingestion -f"
echo -e "    kubectl logs -n $NAMESPACE deployment/frontend -f"
echo ""

# Port-forward fallback if no ingress
if [ -z "$INGRESS_IP" ]; then
    echo -e "  ${CYAN}Local access (port-forwards running):${NC}"
    echo -e "    Frontend:     http://localhost:3000"
    echo -e "    Retrieval API: http://localhost:8000/docs"
    echo -e "    Ingestion API: http://localhost:8001/docs"
fi
echo ""

# ── Smoke test ────────────────────────────────────────────────────────────────
info "Running smoke tests..."
sleep 3
smoke_passed=0
smoke_failed=0

for deploy in api-ingestion api-retrieval frontend; do
    pod_name=$(kubectl get pod -n "$NAMESPACE" -l "app=$deploy" -o name 2>/dev/null | head -1 || echo "")
    if [ -n "$pod_name" ]; then
        pod_status=$(kubectl get "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
        if [ "$pod_status" = "Running" ]; then
            ok "Pod $deploy is Running"
            smoke_passed=$((smoke_passed + 1))
        else
            warn "Pod $deploy status: $pod_status"
            smoke_failed=$((smoke_failed + 1))
        fi
    else
        warn "No pod found for $deploy"
        smoke_failed=$((smoke_failed + 1))
    fi
done

echo ""
echo -e "${GREEN}Smoke tests: ${smoke_passed} passed, ${RED}${smoke_failed} failed${NC}"
echo -e "${BOLD}Deployment complete!${NC}"
