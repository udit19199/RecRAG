# CI/CD Setup Guide

## Overview

RecRAG uses GitHub Actions for continuous integration and deployment. The pipeline addresses production readiness issues identified in [PRODUCTION_ISSUES.md](./PRODUCTION_ISSUES.md).

## Pipeline Stages

```
PR Push / Main Push
        │
        ├── Backend Checks (lint, format, typecheck, test)
        ├── Frontend Checks (lint, format, build)
        └── Security Scan (pip-audit, Trivy)
                │
        [On push to main only]
                │
        ├── Build & Push Docker Images (GHCR)
        ├── Deploy to EC2 (SSH)
        └── Smoke Tests (health checks)
```

## Required GitHub Secrets

| Secret | Description | Example |
|--------|-------------|---------|
| `EC2_SSH_KEY` | SSH private key for EC2 deployment | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `EC2_HOST` | EC2 instance hostname or IP | `ec2-xx-xx-xx-xx.compute-1.amazonaws.com` |
| `EC2_USER` | SSH username (default: `ec2-user`) | `ec2-user` or `ubuntu` |
| `EC2_HEALTH_URL` | Health check base URL | `http://ec2-host:8000` |
| `FRONTEND_URL` | Frontend URL for smoke tests | `http://ec2-host:3000` |

## Required GitHub Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_RETRIEVAL_API_URL` | Retrieval API URL for frontend build | `http://localhost:8000` |
| `NEXT_PUBLIC_INGESTION_API_URL` | Ingestion API URL for frontend build | `http://localhost:8001` |

## EC2 Instance Setup

### Prerequisites

1. EC2 instance with Docker installed
2. Security group allowing ports 3000, 8000, 8001
3. SSH key pair configured for GitHub Actions access
4. Persistent EBS volume mounted at `/opt/recrag`

### Initial Setup Script

Run these commands on the EC2 instance before first deployment:

```bash
# Create directories
sudo mkdir -p /opt/recrag/{data,state}

# Copy config and env files
sudo cp config.toml /opt/recrag/config.toml
sudo cp .env /opt/recrag/.env

# Create Docker network
docker network create recrag-network

# Set permissions
sudo chown -R ec2-user:ec2-user /opt/recrag
```

### Environment File (/opt/recrag/.env)

```env
OPENAI_API_KEY=sk-...
NVIDIA_API_KEY=nvapi-...
MILVUS_URI=http://milvus-host:19530
```

## Smoke Tests

The pipeline runs automated smoke tests after deployment. You can also run them manually:

```bash
# From the repository root
uv run python jobs/smoke_test.py --base-url http://your-ec2-host

# With a specific test PDF
uv run python jobs/smoke_test.py --base-url http://your-ec2-host --test-pdf data/sample.pdf
```

## Docker Images

Images are pushed to GitHub Container Registry (GHCR):

- `ghcr.io/<owner>/recrag-api:<tag>` - Backend APIs
- `ghcr.io/<owner>/recrag-frontend:<tag>` - Next.js frontend

Tags are based on:
- Git SHA: `sha-abc123`
- Branch name: `main`
- Semantic version: `v1.0.0` (when tagged)

## Troubleshooting

### Deployment fails with SSH error

1. Verify `EC2_SSH_KEY` secret is correct
2. Check EC2 security group allows SSH (port 22) from GitHub Actions IPs
3. Test SSH manually: `ssh -i key.pem ec2-user@ec2-host`

### Smoke tests fail

1. Check service logs: `docker logs recrag-api-ingestion`
2. Verify health endpoint: `curl http://ec2-host:8000/health`
3. Check Docker network: `docker network inspect recrag-network`

### Build fails

1. Check workflow logs in GitHub Actions tab
2. Verify `make check` passes locally
3. Ensure Dockerfiles build locally: `docker build --target api .`

## Security Notes

- API keys are stored in GitHub Secrets, not in the repository
- Docker images are scanned with Trivy on every push
- Python dependencies are audited with pip-audit
- CORS is restricted in production (see S-02 in PRODUCTION_ISSUES.md)
