# RecRAG — Deploy in One Command

This is the fastest way to get RecRAG running on any cloud VM.

## What you need

- A cloud VM (Ubuntu/Debian preferred) with **Docker** installed
- API keys for at least one LLM provider (OpenAI, Ollama, or NVIDIA NIM)
- A **Milvus** instance (recommended: [Zilliz Cloud free tier](https://cloud.zilliz.com/))

> **No Kubernetes experience required.** The bootstrap script installs k3s
> (lightweight Kubernetes, ~80MB binary) automatically. If you already have
> a cluster, it skips that step and deploys straight into yours.

---

## One-command deploy

```bash
# 1. Clone the repo
git clone https://github.com/udit19199/RecRAG.git
cd RecRAG

# 2. Set up your API keys
cp .env.example .env
nano .env     # paste in your keys (see tips below)

# 3. Run it
make deploy-k8s
```

That's it. After 3-5 minutes you'll see:

```
══════════════════════════════════════════════════════
  RecRAG is deployed!
══════════════════════════════════════════════════════

  Frontend:       http://recrag.example.com
  Retrieval API:  http://retrieval.example.com
  Ingestion API:  http://ingestion.example.com
```

---

## .env tips

Minimal working `.env`:

```env
# At least one LLM provider
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxx

# Your Milvus connection
MILVUS_HOST=in03-xxxx.serverless.cloud.zilliz.com
MILVUS_USERNAME=your-email
MILVUS_PASSWORD=your-password

# API key for RecRAG itself (optional in dev)
REC_RAG_API_KEY=my-secret-key
```

Leave any provider keys blank if you're not using that provider.

---

## Custom domain

By default the script uses `example.com` as the hostname suffix. Pass your
real domain (or a nip.io address for quick testing):

```bash
# Quick test without DNS — nip.io resolves <IP>.nip.io to the IP
DOMAIN=203.0.113.42.nip.io make deploy-k8s

# Production domain
DOMAIN=recrag.mycompany.com make deploy-k8s
```

This sets up three hostnames:

| Address                         | Service        |
|---------------------------------|----------------|
| `recrag.your-domain`           | Frontend       |
| `retrieval.your-domain`        | Retrieval API  |
| `ingestion.your-domain`        | Ingestion API  |

---

## Dev overlay (smaller footprint)

On a low-resource VM (1-2 GB RAM), use the dev overlay:

```bash
make deploy-k8s-dev
```

This reduces resource requests (256Mi per pod instead of 2Gi), disables
autoscaling, and uses 1Gi PVCs. Good for testing.

---

## What happens behind the scenes

The `scripts/bootstrap.sh` script runs these steps automatically:

| # | Step |
|---|------|
| 1 | Detects if a Kubernetes cluster exists; if not, installs **k3s** |
| 2 | Installs **kustomize** for manifest management |
| 3 | Installs **nginx-ingress** controller for traffic routing |
| 4 | Creates the `recrag` namespace |
| 5 | Reads your `.env` file and creates the `recrag-secrets` Secret |
| 6 | Patches ingress hostnames to your domain |
| 7 | Applies all K8s manifests (Deployments, Services, PVCs, HPAs, Ingress) |
| 8 | Waits for every deployment to roll out successfully |
| 9 | Prints access URLs and next steps |

The three services deployed:

```
                        ┌──────────────────┐
                        │    Ingress       │
                        │  (nginx, :80)    │
                        └────┬──────┬─────┘
                             │      │
                    ┌────────▼┐  ┌──▼──────────┐
                    │ Frontend│  │  APIs        │
                    │ (:3000) │  │              │
                    │ Next.js │  │ Retrieval    │
                    │         │  │ (:8000)      │
                    │         │  │ Ingestion    │
                    │         │  │ (:8001)      │
                    └─────────┘  └──────┬───────┘
                                        │
                                 ┌──────▼───────┐
                                 │   Milvus     │
                                 │  (external)  │
                                 └──────────────┘
```

---

## Useful commands after deploy

```bash
# Check pod status
kubectl get pods -n recrag

# Follow retrieval API logs
kubectl logs -n recrag deployment/api-retrieval -f

# Follow ingestion API logs
kubectl logs -n recrag deployment/api-ingestion -f

# Follow frontend logs
kubectl logs -n recrag deployment/frontend -f

# Manually trigger an evaluation job
kubectl delete job recrag-evaluation -n recrag --ignore-not-found
kubectl apply -f k8s/base/evaluation-job.yaml

# Scale up (e.g. retrieval to handle more queries)
kubectl scale deployment/api-retrieval -n recrag --replicas=3

# Tear everything down
kubectl delete namespace recrag
```

---

## Accessing the services

### Via DNS

Point these DNS records to your cluster's Ingress IP:

```bash
# Get the Ingress IP
kubectl get svc -n ingress-nginx ingress-nginx-controller \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

Then add A records:
```
recrag.<your-domain>       → <ingress-ip>
retrieval.<your-domain>    → <ingress-ip>
ingestion.<your-domain>    → <ingress-ip>
```

### Via /etc/hosts (quick test)

```bash
echo "<ingress-ip> recrag.example.com retrieval.example.com ingestion.example.com" \
  | sudo tee -a /etc/hosts
```

### Via port-forward (no DNS needed)

```bash
kubectl port-forward -n recrag deployment/frontend 3000:3000 &
kubectl port-forward -n recrag deployment/api-retrieval 8000:8000 &
kubectl port-forward -n recrag deployment/api-ingestion 8001:8001 &
```

Then access: http://localhost:3000

---

## Known limitations

- **Frontend API URLs are baked at build time.** Changing the ingress domain
  after deployment requires rebuilding the frontend Docker image with the new
  `NEXT_PUBLIC_*` build args. See `docs/KUBERNETES_DEPLOYMENT.md` for details.
- **Milvus is external.** This script does not deploy Milvus inside the
  cluster. Use Zilliz Cloud or a self-hosted Milvus separately.
- **Single-node cluster for now.** k3s runs on one VM. For HA, deploy to
  a proper multi-node cluster (EKS, GKE, AKS) and use the Kustomize manifests
  directly.

---

## Reference

| File | Description |
|------|-------------|
| `scripts/bootstrap.sh` | The one-command bootstrap script |
| `Makefile` | `make deploy-k8s` and `make deploy-k8s-dev` targets |
| `k8s/base/` | Shared Kubernetes manifests |
| `k8s/overlays/dev/` | Dev overlay (smaller resources) |
| `k8s/overlays/prod/` | Production overlay (multi-replica, TLS) |
| `docs/KUBERNETES_DEPLOYMENT.md` | Full K8s deployment guide |
| `docs/DEPLOYMENT_GUIDE_DEV.md` | VM-based (non-K8s) deployment guide |
