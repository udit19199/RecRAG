# RecRAG — Retrieval-Augmented Generation

Lightweight RAG pipeline: upload a PDF batch, index the full corpus, and query via LLM-backed RAG.

## Quick Start (Kubernetes — one command)

On any cloud VM (Ubuntu/Debian) with Docker installed:

```bash
git clone https://github.com/udit19199/RecRAG.git && cd RecRAG
cp .env.example .env                    # Edit with your API keys
make deploy-k8s                          # One command: installs K8s + deploys everything
```

The script installs k3s, kustomize, and nginx-ingress if needed, creates
secrets from `.env`, and applies the Kustomize manifests. After ~3-5 minutes
you'll see access URLs printed.

Advanced variants:

```bash
make deploy-k8s-dev                      # Dev overlay (smaller resources, no HPA)
OVERLAY=dev DOMAIN=my.vm.ip.nip.io bash scripts/bootstrap.sh  # Custom domain
```

See [Kubernetes Deployment Guide](docs/KUBERNETES_DEPLOYMENT.md) for full details.

## Quick Start (Docker)

```bash
cp .env.example .env  # Update with your API keys and Milvus credentials
make docker-up
```

Note: the frontend browser bundle uses build-time API URLs in Docker, so the
compose build needs the API endpoints baked in (handled by `make docker-up`).

The default and recommended setup is managed/serverless Milvus configured via
`config.toml` and `.env`.

## Local Development

```bash
make setup      # One-time setup (deps + .env)
make dev        # Start all services (Next.js + APIs)
```

Optional local Milvus for development:

```bash
make infra-up
```

- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **APIs**: [8000](http://localhost:8000) (Retrieval) / [8001](http://localhost:8001) (Ingestion)
- **Checks**: `make check` (Lint + Test)
- **Tools**: `make ingest` / `make evaluate`

---

---

## Deployment

### Kubernetes (Recommended)

📖 **[Kubernetes Deployment Guide](docs/KUBERNETES_DEPLOYMENT.md)**

Deploy on any CNCF Kubernetes cluster (minikube, k3s, GKE, EKS, AKS) using
Kustomize manifests:

```bash
# Apply secrets first, then deploy
kubectl create namespace recrag
kubectl create secret generic recrag-secrets -n recrag \
  --from-literal=openai-api-key='sk-...' \
  --from-literal=milvus-host='in03-xxxx.serverless.cloud.zilliz.com'

kubectl kustomize k8s/overlays/dev | kubectl apply -f -
```

Three ingress subdomains route to the correct services (edit hostnames in
the overlay before deploying):

| Subdomain                       | Service      |
|--------------------------------|--------------|
| `recrag.example.com`          | Frontend     |
| `retrieval.recrag.example.com`| Retrieval API|
| `ingestion.recrag.example.com`| Ingestion API|

### VM / Docker Compose

For detailed deployment guidance across AWS, GCP, and Azure for development and testing phases:

📖 **[Development & Testing Deployment Guide](docs/DEPLOYMENT_GUIDE_DEV.md)**

Includes:
- Architecture overview and service breakdown
- Hardware requirements for each service
- Stress testing specifications
- Cost comparison across AWS, GCP, and Azure
- Step-by-step deployment instructions
- Monitoring and troubleshooting tips

---

## Configuration
Configuration is managed in `config.toml` (supports `${VAR:-default}` env var substitution).
