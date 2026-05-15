#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# RecRAG bootstrap — clone-to-running in one command.
#
# Usage (on any Ubuntu/Debian cloud instance with Docker/Podman):
#
#   git clone https://github.com/udit19199/RecRAG.git && cd RecRAG
#   cp .env.example .env          # edit with your keys
#   bash scripts/bootstrap.sh     # one command
#
# What it does:
#   1. Installs k3s (lightweight Kubernetes) if no kubeconfig is found
#   2. Installs kustomize if missing
#   3. Reads .env and creates the recrag-secrets Secret
#   4. Deploys RecRAG via Kustomize (k8s/overlays/prod)
#   5. Waits for all deployments to roll out
#   6. Prints access URLs and next steps
#
# Environment variables:
#   OVERLAY    — kustomize overlay to use (default: prod)
#   NAMESPACE  — Kubernetes namespace (default: recrag)
#   DOMAIN     — ingress hostname suffix (default: example.com)
#                e.g. DOMAIN=recrag.mycompany.com → services at
#                recrag.recrag.mycompany.com, retrieval.recrag.mycompany.com, ...
#                NOTE: Changing DOMAIN requires rebuilding the frontend with
#                correct NEXT_PUBLIC_* URLs — see docs/KUBERNETES_DEPLOYMENT.md
#
# Examples:
#   bash scripts/bootstrap.sh                          # prod overlay
#   OVERLAY=dev bash scripts/bootstrap.sh              # dev overlay (lightweight)
#   DOMAIN=recrag.mydevbox.com bash scripts/bootstrap.sh  # custom domain
# ──────────────────────────────────────────────────────────────────────────────

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

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  warn ".env file not found. Creating from .env.example..."
  cp .env.example .env
  echo ""
  err "Edit .env with your API keys, then re-run this script."
  exit 1
fi

# Source .env so we can read keys (source safely, ignoring comments/blank lines)
set -a; source .env; set +a

if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${NVIDIA_API_KEY:-}" ] && [ -z "${OLLAMA_API_KEY:-}" ]; then
  warn "No LLM provider API key found in .env. Set at least one of:"
  warn "  OPENAI_API_KEY, NVIDIA_API_KEY, or OLLAMA_API_KEY"
fi

if [ -z "${MILVUS_HOST:-}" ]; then
  warn "MILVUS_HOST not set in .env — Milvus connection will fail."
fi

# ── Step 1: Install k3s (if no Kubernetes cluster) ────────────────────────────
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
  # kubectl exists — check if it can actually talk to a cluster
  if ! kubectl cluster-info --request-timeout=5s &>/dev/null; then
    warn "kubectl found but cannot reach a cluster. Installing k3s..."
    # Check if k3s binary exists but service isn't running
    if command -v k3s &>/dev/null; then
      info "k3s binary found — starting service..."
      sudo systemctl start k3s 2>/dev/null || sudo service k3s start 2>/dev/null || true
      sleep 5
    else
      install_k3s
    fi
  fi
fi

ok "Kubernetes cluster is reachable: $(kubectl cluster-info 2>/dev/null | head -1)"

# ── Step 2: Install kustomize (if missing) ────────────────────────────────────
if ! command -v kustomize &>/dev/null && ! kubectl kustomize --help &>/dev/null 2>&1; then
  info "Installing kustomize..."
  curl -sSLO https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.4.3/kustomize_v5.4.3_linux_amd64.tar.gz
  tar -xzf kustomize_v5.4.3_linux_amd64.tar.gz
  chmod +x kustomize && sudo mv kustomize /usr/local/bin/
  rm -f kustomize_v5.4.3_linux_amd64.tar.gz
  ok "kustomize installed"
fi

# ── Step 3: Install ingress-nginx (if not already running) ────────────────────
if ! kubectl get pods -n ingress-nginx --request-timeout=5s &>/dev/null 2>&1; then
  info "Installing nginx-ingress controller..."
  kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.0/deploy/static/provider/baremetal/deploy.yaml
  sleep 10
  kubectl wait --namespace ingress-nginx \
    --for=condition=ready pod \
    --selector=app.kubernetes.io/component=controller \
    --timeout=120s
  ok "nginx-ingress controller running"
fi

# ── Step 4: Create namespace ──────────────────────────────────────────────────
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
ok "Namespace '$NAMESPACE' ready"

# ── Step 5: Create secrets from .env ─────────────────────────────────────────
info "Creating recrag-secrets from .env..."
kubectl delete secret recrag-secrets -n "$NAMESPACE" --ignore-not-found

# Build --from-literal args from .env (only known keys)
SECRET_ARGS=()
for key in REC_RAG_API_KEY OPENAI_API_KEY NVIDIA_API_KEY OLLAMA_API_KEY \
           MILVUS_HOST MILVUS_USERNAME MILVUS_PASSWORD; do
  val="${!key:-}"
  if [ -n "$val" ]; then
    SECRET_ARGS+=(--from-literal="$(echo "$key" | tr '[:upper:]' '[:lower:]' | tr '_' '-')=$val")
  fi
done

# Map env var names to k8s secret key names (lowercased with hyphens)
kubectl create secret generic recrag-secrets \
  -n "$NAMESPACE" \
  "${SECRET_ARGS[@]}" 2>/dev/null || true

# If --from-literal didn't work (empty), create empty secret as fallback
if ! kubectl get secret recrag-secrets -n "$NAMESPACE" &>/dev/null; then
  kubectl create secret generic recrag-secrets -n "$NAMESPACE" \
    --from-literal=placeholder=unset
fi
ok "Secret 'recrag-secrets' created"

# ── Step 6: Override ingress hostnames in the overlay ─────────────────────────
# We apply the overlay as-is, but patch the ingress hostnames and frontend URLs.
# Create a temp overlay that includes the prod overlay + hostname patches.

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

cat > "$TMPDIR/kustomization.yaml" <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../overlays/$OVERLAY

patches:
  - target:
      kind: Ingress
      name: recrag-ingress
    patch: |-
      - op: replace
        path: /spec/rules
        value:
          - host: recrag.$DOMAIN
            http:
              paths:
                - path: /
                  pathType: Prefix
                  backend:
                    service:
                      name: svc-frontend
                      port:
                        number: 3000
          - host: retrieval.$DOMAIN
            http:
              paths:
                - path: /
                  pathType: Prefix
                  backend:
                    service:
                      name: svc-retrieval
                      port:
                        number: 8000
          - host: ingestion.$DOMAIN
            http:
              paths:
                - path: /
                  pathType: Prefix
                  backend:
                    service:
                      name: svc-ingestion
                      port:
                        number: 8001
      - op: add
        path: /spec/tls
        value: []
  - target:
      kind: Deployment
      name: frontend
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/env
        value:
          - name: NEXT_PUBLIC_RETRIEVAL_API_URL
            value: http://retrieval.$DOMAIN
          - name: NEXT_PUBLIC_INGESTION_API_URL
            value: http://ingestion.$DOMAIN
EOF

# Copy overlay files referenced by the temp kustomization
cp -r "k8s/overlays/$OVERLAY"/* "$TMPDIR/" 2>/dev/null || true

# ── Step 7: Apply manifests ──────────────────────────────────────────────────
info "Deploying RecRAG (overlay: $OVERLAY)..."

cd "$TMPDIR"
kubectl kustomize . | kubectl apply -f -
cd - >/dev/null

# ── Step 8: Wait for rollout ─────────────────────────────────────────────────
info "Waiting for deployments to roll out..."
for deploy in api-ingestion api-retrieval frontend; do
  if kubectl get deployment "$deploy" -n "$NAMESPACE" &>/dev/null; then
    kubectl rollout status "deployment/$deploy" -n "$NAMESPACE" --timeout=5m || \
      warn "Rollout for $deploy timed out — check with: kubectl get pods -n $NAMESPACE"
  fi
done

# ── Step 9: Print summary ────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  RecRAG is deployed!${NC}"
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Frontend:${NC}     http://recrag.$DOMAIN"
echo -e "  ${CYAN}Retrieval API:${NC} http://retrieval.$DOMAIN"
echo -e "  ${CYAN}Ingestion API:${NC} http://ingestion.$DOMAIN"
echo ""
echo -e "  ${YELLOW}Note: DNS must point these hostnames to your cluster's Ingress IP.${NC}"
echo ""

# Show the Ingress IP
INGRESS_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
INGRESS_HOST=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
if [ -n "$INGRESS_IP" ]; then
  echo -e "  Ingress IP: ${BOLD}$INGRESS_IP${NC}"
  echo -e "  Add these to your DNS or /etc/hosts:"
  echo -e "    $INGRESS_IP  recrag.$DOMAIN retrieval.$DOMAIN ingestion.$DOMAIN"
elif [ -n "$INGRESS_HOST" ]; then
  echo -e "  Ingress hostname: ${BOLD}$INGRESS_HOST${NC}"
fi

echo ""
echo -e "  ${CYAN}Useful commands:${NC}"
echo -e "    kubectl get pods -n $NAMESPACE"
echo -e "    kubectl logs -n $NAMESPACE deployment/api-retrieval -f"
echo -e "    kubectl logs -n $NAMESPACE deployment/api-ingestion -f"
echo -e "    kubectl logs -n $NAMESPACE deployment/frontend -f"
echo -e "    kubectl kustomize k8s/overlays/$OVERLAY | kubectl diff -f -"
echo ""
echo -e "  ${YELLOW}To remove everything:${NC}  kubectl delete namespace $NAMESPACE"
echo ""

# ── Step 10 (optional): port-forward for quick access if no Ingress IP ───────
if [ -z "$INGRESS_IP" ] && [ -z "$INGRESS_HOST" ]; then
  echo -e "${YELLOW}No Ingress LoadBalancer IP detected. Starting port-forwards...${NC}"
  kubectl port-forward -n "$NAMESPACE" deployment/frontend 3000:3000 &
  kubectl port-forward -n "$NAMESPACE" deployment/api-retrieval 8000:8000 &
  kubectl port-forward -n "$NAMESPACE" deployment/api-ingestion 8001:8001 &
  echo ""
  echo -e "  ${CYAN}Local access:${NC}"
  echo -e "    Frontend:     http://localhost:3000"
  echo -e "    Retrieval API: http://localhost:8000/docs"
  echo -e "    Ingestion API: http://localhost:8001/docs"
  echo ""
  echo -e "  ${YELLOW}Press Ctrl+C to stop port-forwards.${NC}"
  wait
fi
