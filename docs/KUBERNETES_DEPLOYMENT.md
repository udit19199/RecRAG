# RecRAG Kubernetes Deployment Guide

Deploy RecRAG on any Kubernetes cluster — minikube, k3s, GKE, EKS, AKS, or
self-managed.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Architecture](#kubernetes-architecture)
4. [Manifest Overview](#manifest-overview)
5. [Configuration](#configuration)
6. [Ingress & TLS](#ingress--tls)
7. [CI/CD Integration](#cicd-integration)
8. [Monitoring](#monitoring)
9. [Scaling](#scaling)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Kubernetes cluster** — any CNCF-certified distribution
- **kubectl** ≥ 1.28
- **kustomize** (built into kubectl ≥ 1.21 via `kubectl kustomize`)
- **GitHub Container Registry access** — images are in `ghcr.io/udit19199/`
- **External Milvus** — managed (Zilliz Cloud) or self-hosted Milvus instance
- **Ingress controller** — nginx-ingress recommended; see [Ingress & TLS](#ingress--tls)

---

## Quick Start

The fastest way to go from clone to running:

```bash
# ONE COMMAND — installs K8s + deploys everything
git clone https://github.com/udit19199/RecRAG.git && cd RecRAG
cp .env.example .env                    # Edit with your API keys
make deploy-k8s
```

That's it. The `scripts/bootstrap.sh` script handles everything:

1. Installs **k3s** (lightweight K8s) if no cluster is present
2. Installs **kustomize** if missing
3. Installs **nginx-ingress** controller
4. Reads `.env` and creates the `recrag-secrets` Secret
5. Patches ingress hostnames with your domain (default: `example.com`)
6. Applies the Kustomize manifests
7. Waits for rollout and prints access URLs

### Manual Deploy

If you already have a cluster, you can skip the bootstrap:

---

## Kubernetes Architecture

```
                         ┌──────────────┐
                         │   Ingress    │   ── recrag.example.com
                         │  (nginx/gce) │      retrieval.recrag.example.com
                         └──────┬───────┘      ingestion.recrag.example.com
                                │
                     ┌──────────┼──────────┐
                     │          │          │
                ┌────▼────┐ ┌──▼───┐ ┌───▼─────┐
                │ Frontend │ │ Retr │ │ Ingest  │
                │ (:3000)  │ │(:8000)│ │ (:8001) │
                │ ClusterIP│ │ClusterIP││ClusterIP│
                └────┬─────┘ └──┬───┘ └───┬─────┘
                     │          │          │
                     └──────────┼──────────┘
                                │
                      ┌─────────▼────────┐
                      │   Milvus (ext.)  │
                      │ (Zilliz / self)   │
                      └──────────────────┘
```

### Services

| Component    | Container Image                        | Port | Kind       | Replicas (prod) |
|-------------|----------------------------------------|------|------------|-----------------|
| Ingestion   | `ghcr.io/udit19199/recrag-api`        | 8001 | Deployment | 2               |
| Retrieval   | `ghcr.io/udit19199/recrag-api`        | 8000 | Deployment | 2               |
| Frontend    | `ghcr.io/udit19199/recrag-frontend`   | 3000 | Deployment | 2               |
| Evaluation  | `ghcr.io/udit19199/recrag-api`        | —    | Job        | on-demand       |

### Stateful Dependencies

| Resource       | Type                | Size (dev) | Size (prod) | Mounted By               |
|---------------|---------------------|-----------|-------------|--------------------------|
| `recrag-data` | PersistentVolumeClaim | 1 Gi      | 50 Gi       | Ingestion, Evaluation    |
| `recrag-state`| PersistentVolumeClaim | 256 Mi    | 5 Gi        | Ingestion, Retrieval, Evaluation |

### External Dependencies (not in-cluster)

- **Milvus** — vector database. Use [Zilliz Cloud](https://cloud.zilliz.com/) (recommended)
  or deploy separately. Configured via `config.toml` and env vars.
- **LLM / Embedding providers** — OpenAI, Ollama, NVIDIA NIM, etc.
  Set API keys in the `recrag-secrets` Secret.

---

## Manifest Overview

```
k8s/
├── base/
│   ├── kustomization.yaml              # Resource list, common labels, image defaults
│   ├── namespace.yaml                  # recrag namespace
│   ├── serviceaccount.yaml             # Service account for pods
│   ├── configmap.yaml                  # config.toml (shared across all services)
│   ├── secret.yaml                     # TEMPLATE — populate via kubectl, not git
│   ├── services.yaml                   # ClusterIP services for all 3 components
│   ├── pvc.yaml                        # PersistentVolumeClaims for data + state
│   ├── ingress.yaml                    # nginx-ingress with subdomain routing
│   ├── api-ingestion-deployment.yaml   # Ingestion FastAPI
│   ├── api-retrieval-deployment.yaml   # Retrieval FastAPI
│   ├── frontend-deployment.yaml        # Next.js frontend
│   ├── hpa.yaml                        # HorizontalPodAutoscalers (scaling)
│   └── evaluation-job.yaml             # On-demand evaluation batch job
└── overlays/
    ├── dev/
    │   └── kustomization.yaml          # Smaller resources, no HPA, local dev hosts
    └── prod/
        └── kustomization.yaml          # Multi-replica, TLS, larger PVCs
```

### Using Kustomize

```bash
# Preview dev manifests
kubectl kustomize k8s/overlays/dev

# Apply dev to a cluster
kubectl kustomize k8s/overlays/dev | kubectl apply -f -

# Apply production
kubectl kustomize k8s/overlays/prod | kubectl apply -f -
```

Kustomize is layered — `overlays/prod` inherits everything from `base/` and
patches only what differs. You can create additional overlays for staging,
review apps, etc.

---

## Configuration

### 1. Secrets (required — do before deploying)

Create the `recrag-secrets` Secret with your API keys:

```bash
kubectl create secret generic recrag-secrets \
  --namespace recrag \
  --from-literal=rec-rag-api-key='...' \
  --from-literal=openai-api-key='sk-...' \
  --from-literal=nvidia-api-key='nvapi-...' \
  --from-literal=ollama-api-key='...' \
  --from-literal=milvus-host='in03-xxxx.serverless.cloud.zilliz.com' \
  --from-literal=milvus-username='' \
  --from-literal=milvus-password=''
```

All secret keys are optional — omit providers you don't use.

### 2. ConfigMap

The `recrag-config` ConfigMap holds `config.toml`. To override for your environment:

```bash
# Edit the base ConfigMap or create a patch overlay
kubectl edit configmap recrag-config -n recrag
```

Changes take effect on pod restart (config is not hot-reloaded).

---

## Ingress & TLS

### nginx-ingress (recommended for most clusters)

```bash
# Install nginx-ingress if not present
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx --create-namespace
```

The base Ingress manifest uses the `nginx` ingress class. For production, add
TLS with cert-manager:

```bash
# Install cert-manager
helm repo add jetstack https://charts.jetstack.io
helm upgrade --install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true

# Create a ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
EOF
```

Then add the `cert-manager.io/cluster-issuer` annotation to the Ingress (already
documented in `k8s/base/ingress.yaml`).

### GKE Ingress

```yaml
# In k8s/overlays/prod/kustomization.yaml, change:
kubernetes.io/ingress.class: "gce"
# And remove cert-manager annotation — GKE handles TLS via ManagedCertificate.
```

### Alternative: LoadBalancer Services

If you don't want an Ingress, switch services to `LoadBalancer` type:

```yaml
metadata:
  name: svc-frontend
spec:
  type: LoadBalancer  # instead of ClusterIP
```

Each service gets an external IP. Add DNS A/AAAA records pointing to them.

---

## Frontend Public API URLs

The Next.js frontend has `NEXT_PUBLIC_RETRIEVAL_API_URL` and
`NEXT_PUBLIC_INGESTION_API_URL` baked into the browser bundle at **build time**.
These must match your ingress hostnames:

| Env Var                        | Value (production)                  |
|-------------------------------|-------------------------------------|
| `NEXT_PUBLIC_RETRIEVAL_API_URL` | `https://retrieval.recrag.example.com` |
| `NEXT_PUBLIC_INGESTION_API_URL` | `https://ingestion.recrag.example.com` |

Set these:
1. **In Docker builds** — via `--build-arg` in CI/CD (see `.github/workflows/ci-cd.yml`)
2. **In the frontend Deployment** — via `env` section (see `k8s/base/frontend-deployment.yaml`)
   - Note: runtime env vars only affect server-side rendering, not client JS.
     For client-side, the values must be baked in at build time.

---

## CI/CD Integration

The CI/CD pipeline already:
1. Builds backend (`recrag-api`) and frontend (`recrag-frontend`) images
2. Pushes them to GHCR with SHA tags
3. Deploys to EC2 via SSH

To add K8s deployment, extend the workflow with a `deploy-k8s` job:

```yaml
deploy-k8s:
  name: Deploy to Kubernetes
  runs-on: ubuntu-latest
  needs: [build-and-push]
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  steps:
    - uses: actions/checkout@v4

    - name: Set up kubectl
      uses: azure/setup-kubectl@v4
      with:
        version: "v1.30.0"

    - name: Configure kubeconfig
      run: |
        mkdir -p $HOME/.kube
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > $HOME/.kube/config

    - name: Update image tags
      working-directory: k8s/overlays/prod
      run: |
        kustomize edit set image \
          ghcr.io/udit19199/recrag-api=ghcr.io/udit19199/recrag-api:sha-${{ github.sha }} \
          ghcr.io/udit19199/recrag-frontend=ghcr.io/udit19199/recrag-frontend:sha-${{ github.sha }}

    - name: Deploy
      run: |
        kubectl kustomize k8s/overlays/prod | kubectl apply -f -
        kubectl rollout status deployment/api-retrieval -n recrag --timeout=5m
        kubectl rollout status deployment/api-ingestion -n recrag --timeout=5m
        kubectl rollout status deployment/frontend -n recrag --timeout=5m
```

**Required GitHub Secrets** for the K8s deploy job:

| Secret          | Description                                              |
|----------------|----------------------------------------------------------|
| `KUBECONFIG`   | Base64-encoded kubeconfig YAML (context with deploy rights) |

---

## Monitoring

### Prometheus Metrics

All API pods expose Prometheus metrics at `/metrics` (port 8000/8001).
Annotations are pre-configured:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8001"  # or "8000" for retrieval
```

If using kube-prometheus-stack (Prometheus Operator), it auto-discovers pods
with these annotations.

### Dashboards

Import the sample Grafana dashboard (coming soon — PRs welcome). Key metrics:

| Metric                              | Description             |
|-------------------------------------|-------------------------|
| `http_requests_total`               | Request count by route  |
| `http_request_duration_seconds`     | Latency histogram       |
| `http_request_exceptions_total`     | Error rate by route     |

### Logs

```bash
# Pod logs
kubectl logs -n recrag deployment/api-ingestion
kubectl logs -n recrag deployment/api-retrieval
kubectl logs -n recrag deployment/frontend

# Tail recent logs
kubectl logs -n recrag deployment/api-ingestion --tail=100 -f

# JSON-structured logs — filter by request ID
kubectl logs -n recrag deployment/api-ingestion | grep '"request_id":"req_...'
```

For log aggregation, deploy a DaemonSet like fluent-bit or use your cloud
provider's logging agent.

---

## Scaling

### Horizontal Pod Autoscaling

HPAs are pre-configured in `k8s/base/hpa.yaml`:

| Deployment     | Min → Max | CPU Target | Memory Target |
|---------------|-----------|-----------|---------------|
| api-ingestion  | 1 → 5     | 70%       | 80%           |
| api-retrieval  | 1 → 10    | 70%       | 80%           |
| frontend       | 1 → 5     | 70%       | —             |

HPAs require the **metrics-server** in your cluster:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Manual Scaling

```bash
kubectl scale deployment/api-retrieval -n recrag --replicas=3
kubectl scale deployment/api-ingestion -n recrag --replicas=3
kubectl scale deployment/frontend -n recrag --replicas=3
```

---

## Running Evaluation as a Job

```bash
# Trigger a one-shot evaluation
kubectl delete job recrag-evaluation -n recrag --ignore-not-found
kubectl apply -f k8s/base/evaluation-job.yaml

# Monitor
kubectl logs -n recrag job/recrag-evaluation -f

# Check results (mounted in state/ PVC)
# Attach to a debug pod or read directly from the PV
```

For scheduled evaluations, use a CronJob:

```bash
kubectl create cronjob recrag-evaluation-cron \
  --namespace recrag \
  --image=ghcr.io/udit19199/recrag-api \
  --schedule="0 3 * * 0" \
  -- uv run python jobs/evaluate.py
```

---

## Troubleshooting

| Symptom                              | Likely Cause                        | Fix                                    |
|--------------------------------------|-------------------------------------|----------------------------------------|
| Pods stuck in `CrashLoopBackOff`     | Missing secrets or wrong config     | `kubectl describe pod -n recrag <pod>` |
| Ingress returns 404                  | Ingress class mismatch              | Check `kubernetes.io/ingress.class`    |
| Frontend can't reach APIs            | Wrong NEXT_PUBLIC_* URLs            | Rebuild frontend with correct URLs     |
| HPA not scaling                      | metrics-server not deployed         | `kubectl top pods -n recrag`           |
| Milvus connection error              | Wrong host/port or credentials      | Verify `recrag-secrets`                |
| PVCs stuck in Pending                | No StorageClass available           | Set `storageClassName` or install one  |
| ImagePullBackOff                     | Image tag doesn't exist             | Check GHCR for available tags          |

### Pod Debugging

```bash
# Full pod details
kubectl describe pod -n recrag -l app.kubernetes.io/component=api-retrieval

# Shell into a running pod
kubectl exec -it -n recrag deployment/api-retrieval -- /bin/bash

# Check config mounted correctly
kubectl exec -n recrag deployment/api-ingestion -- cat /app/config.toml

# Verify env vars
kubectl exec -n recrag deployment/api-ingestion -- env | grep -E '^(OPENAI|MILVUS|REC_RAG)'
```

---

## Reference

| Path                                  | Description                          |
|---------------------------------------|--------------------------------------|
| `k8s/base/`                           | Shared Kubernetes manifests          |
| `k8s/overlays/dev/`                   | Dev/minikube overrides               |
| `k8s/overlays/prod/`                  | Production overrides                 |
| `docs/DEPLOYMENT_GUIDE_DEV.md`        | VM-based dev deployment (EC2/GCE/etc.) |
| `docs/CICD_SETUP.md`                  | CI/CD pipeline docs                  |
| `.github/workflows/ci-cd.yml`        | GitHub Actions workflow              |
