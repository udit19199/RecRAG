# RecRAG Deployment Guide - Development & Testing Phase

This guide covers deploying RecRAG across **AWS**, **GCP**, and **Azure** for development and testing purposes. Production deployments will be addressed separately.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Services Breakdown](#services-breakdown)
3. [Hardware Requirements](#hardware-requirements)
4. [Deployment Options](#deployment-options)
5. [Cloud Provider Comparison](#cloud-provider-comparison)
6. [Local Development Deployment](#local-development-deployment)
7. [Troubleshooting & Monitoring](#troubleshooting--monitoring)

---

## Architecture Overview

RecRAG consists of 4 primary components:

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                      │
│              Port 3000 (HTTP/HTTPS via proxy)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼────┐  ┌─────▼──────┐  ┌───▼────┐
    │ Ingestion  │ Retrieval   │  │ Eval   │
    │ API       │ API         │  │ Job    │
    │ :8001    │ :8000       │  │(async) │
    └──┬──────┘  └─────┬──────┘  └────────┘
       │              │
       └──────┬───────┘
              │
        ┌─────▼────────────┐
        │  Milvus Vector   │
        │  (Serverless or  │
        │   Local)         │
        └──────────────────┘
```

---

## Services Breakdown

### 1. **Ingestion API (FastAPI)**
- **Port**: 8001
- **Purpose**: Upload PDFs, check ingestion status, reindex collections
- **Dependencies**: Python 3.12, FastAPI, Milvus connection, file storage
- **Key Routes**:
  - `POST /upload` - Accept PDF uploads
  - `GET /status` - Ingestion progress
  - `POST /reindex` - Rebuild vector index
  - `GET /health` - Service health

### 2. **Retrieval API (FastAPI)**
- **Port**: 8000
- **Purpose**: Query documents, run evaluations, manage configuration
- **Dependencies**: Python 3.12, FastAPI, Milvus connection, LLM/embedding providers
- **Key Routes**:
  - `POST /query` - RAG query endpoint
  - `POST /evaluate` - Start async evaluation job
  - `GET /eval-jobs/{job_id}` - Poll evaluation status
  - `GET /config` - View current configuration
  - `PATCH /config` - Update runtime settings
  - `GET /health` - Service health

### 3. **Frontend (Next.js)**
- **Port**: 3000
- **Purpose**: User interface for uploading PDFs and querying
- **Dependencies**: Node.js 18+, React 19, TypeScript
- **Features**:
  - PDF upload interface
  - Query builder
  - Status monitoring
  - Configuration management

### 4. **Vector Database (Milvus)**
- **Port**: 19530 (gRPC)
- **Purpose**: Store and search document embeddings
- **Deployment Models**:
  - **Serverless (Recommended for Dev)**: Zilliz Cloud (managed)
  - **Local (Optional)**: Docker Compose with etcd + MinIO + Milvus

### 5. **Evaluation Job Runner (Async)**
- **Invoked by**: Retrieval API background task
- **Purpose**: Run RAGAS evaluation on query results
- **Output**: JSON evaluation metrics stored in `state/`

---

## Hardware Requirements

### Local Development (macOS/Linux/Windows)

| Service | CPU | RAM | Disk | Notes |
|---------|-----|-----|------|-------|
| **Ingestion API** | 1-2 cores | 1-2 GB | N/A | Minimal I/O |
| **Retrieval API** | 2-4 cores | 2-4 GB | N/A | LLM inference can be local or cloud |
| **Frontend** | 1 core | 512 MB | N/A | Dev server with hot reload |
| **Local Milvus** | 4-8 cores | 8-16 GB | 20-50 GB | Only if running locally; optional |
| **Total (with local Milvus)** | 8-16 cores | 12-26 GB | 20-50 GB | For stress testing |

### AWS Deployment (Dev/Testing)

| Service | Instance Type | vCPU | Memory | Cost/month (approx) | Use Case |
|---------|---------------|------|--------|---------------------|----------|
| **Ingestion API** | t3.small | 2 | 2 GB | $17 | Baseline |
| **Retrieval API** | t3.medium | 2 | 4 GB | $35 | LLM calls, embedding |
| **Frontend** | t3.small | 2 | 2 GB | $17 | Static + API client |
| **Milvus (Standalone)** | m5.xlarge | 4 | 16 GB | $185 | Vector storage |
| **RDS PostgreSQL** | db.t3.micro | - | - | $15/month | Optional: metadata |
| **S3 Storage** | - | - | - | $0.023/GB | PDF storage |
| **Load Balancer** | ALB | - | - | $25 | Optional: multi-AZ |
| **Total (minimal)** | - | - | - | ~$90 | Single-instance dev |
| **Total (resilient)** | - | - | - | ~$180 | Multi-AZ with ALB |

### GCP Deployment (Dev/Testing)

| Service | Machine Type | vCPU | Memory | Cost/month (approx) | Use Case |
|---------|--------------|------|--------|---------------------|----------|
| **Ingestion API** | e2-small | 2 | 2 GB | $15 | Baseline |
| **Retrieval API** | e2-medium | 2 | 4 GB | $33 | LLM calls, embedding |
| **Frontend** | e2-small | 2 | 2 GB | $15 | Static + API client |
| **Milvus (Compute Engine)** | e2-standard-4 | 4 | 16 GB | $95 | Vector storage |
| **Cloud Firestore** | - | - | - | $0.06/100K reads | Optional: metadata |
| **Cloud Storage** | - | - | - | $0.020/GB | PDF storage |
| **Cloud Load Balancer** | - | - | - | $18 | Optional: multi-region |
| **Total (minimal)** | - | - | - | ~$76 | Single-instance dev |
| **Total (resilient)** | - | - | - | ~$176 | Multi-region with LB |

### Azure Deployment (Dev/Testing)

| Service | VM Size | vCPU | Memory | Cost/month (approx) | Use Case |
|---------|---------|------|--------|---------------------|----------|
| **Ingestion API** | B2s | 2 | 4 GB | $30 | Baseline (B-series burstable) |
| **Retrieval API** | B2ms | 2 | 8 GB | $55 | LLM calls, embedding |
| **Frontend** | B1s | 1 | 1 GB | $10 | Static + API client |
| **Milvus (Standard_E4s_v3)** | E4s_v3 | 4 | 32 GB | $220 | Vector storage + cache |
| **Azure SQL Database** | B_Gen5_1 | - | - | $5 | Optional: metadata |
| **Blob Storage** | - | - | - | $0.0184/GB | PDF storage |
| **Application Gateway** | - | - | - | $20 | Optional: WAF + routing |
| **Total (minimal)** | - | - | - | ~$120 | Single-instance dev |
| **Total (resilient)** | - | - | - | ~$240 | Multi-AZ with AG |

### Stress Testing Requirements

For load testing (concurrent users, throughput benchmarking):

| Scenario | CPU | RAM | Disk | Duration | Notes |
|----------|-----|-----|------|----------|-------|
| **10 concurrent users** | 2-4 cores | 4 GB | 5 GB | 1 hour | Baseline test |
| **50 concurrent users** | 4-8 cores | 8 GB | 10 GB | 2 hours | Moderate load |
| **100 concurrent users** | 8-16 cores | 16 GB | 20 GB | 4 hours | High load (DB heavy) |
| **10K document ingestion** | 4 cores | 8 GB | 30 GB | variable | Full corpus benchmark |
| **Sustained 1K eval/day** | 2 cores | 4 GB | 10 GB | continuous | Background worker |

**Testing Tools**:
- `locust` for load generation (Python)
- `wrk` for HTTP benchmarking
- `k6` for continuous load testing
- Prometheus + Grafana for metrics collection

---

## Deployment Options

### Option 1: Local Development (Recommended for Initial Dev)

**Setup Time**: 10-15 minutes  
**Cost**: $0 (uses local machine)  
**Best For**: Individual developers, quick prototyping

```bash
# Clone and install
git clone <repo>
cd RecRAG
make install
make setup

# Start all services locally
make dev              # Starts ingestion, retrieval, frontend on localhost
make infra-up         # (Optional) Start local Milvus

# Access:
# - Frontend: http://localhost:3000
# - Ingestion API: http://localhost:8001/docs
# - Retrieval API: http://localhost:8000/docs
```

**Pros**:
- Zero cost, fastest iteration
- Full control, easy debugging
- No network latency

**Cons**:
- Limited to your machine's resources
- No multi-user testing
- Not representative of production

---

### Option 2: Docker Compose (Recommended for Team Dev)

**Setup Time**: 20-30 minutes  
**Cost**: $0-5/month (if using managed Milvus)  
**Best For**: Small teams, CI/CD integration

```bash
# Build and start all services
make docker-up

# Scale individual services
docker-compose up -d --scale api-retrieval=2

# View logs
make infra-logs

# Stop everything
make docker-down
```

**Pros**:
- Reproducible across machines
- Easy to scale services
- Good for CI/CD pipelines

**Cons**:
- Still runs on local resources
- Milvus recommended external (Zilliz Cloud)

---

### Option 3: AWS Deployment

#### Minimal Setup (Single EC2 instance)

```bash
# 1. Launch EC2 instance (t3.xlarge, 16GB RAM, 50GB storage)
# - AMI: Ubuntu 22.04 LTS
# - Security Group: Allow 3000, 8000, 8001, 22

# 2. SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# 3. Install dependencies
sudo apt update && sudo apt install -y python3.12 nodejs npm docker.io docker-compose git
sudo usermod -aG docker ubuntu

# 4. Clone and deploy
git clone <repo>
cd RecRAG
cp .env.example .env
# Edit .env with AWS/Milvus credentials
make docker-up

# 5. Set up reverse proxy (optional but recommended)
# - Install nginx: sudo apt install -y nginx
# - Configure upstream to :3000, :8000, :8001
# - Enable SSL with Let's Encrypt
```

#### Resilient Multi-AZ Setup

```bash
# Use AWS CloudFormation or Terraform
# Services:
# - ALB with auto-scaling group for APIs
# - RDS PostgreSQL for metadata (optional)
# - S3 for PDF storage
# - Zilliz Cloud for managed Milvus
# - CloudWatch for monitoring
```

**AWS Networking**:
- **VPC**: Create private subnets for services
- **Security Groups**: Restrict inter-service communication
- **Route 53**: DNS with health checks
- **RDS Proxy**: Connection pooling for databases

---

### Option 4: GCP Deployment

#### Minimal Setup (Single Compute Engine VM)

```bash
# 1. Create VM (e2-standard-4, 16GB RAM)
gcloud compute instances create recrag-dev \
  --machine-type=e2-standard-4 \
  --zone=us-central1-a \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

# 2. SSH and deploy
gcloud compute ssh recrag-dev --zone=us-central1-a

# 3. Install and run
sudo apt update && sudo apt install -y python3.12 nodejs docker.io docker-compose
git clone <repo> && cd RecRAG
make docker-up
```

#### Resilient Setup (Multi-region)

```bash
# Use GCP App Engine or Cloud Run for serverless
# - Ingestion/Retrieval APIs on Cloud Run (auto-scaling)
# - Frontend on Cloud Storage + Cloud CDN
# - Firestore for state management
# - Vertex AI for LLM calls (optional)
```

**GCP Specifics**:
- **Cloud Load Balancing**: Multi-region routing
- **Cloud Armor**: DDoS protection
- **Cloud Monitoring**: Built-in observability
- **Secret Manager**: Secure credential storage

---

### Option 5: Azure Deployment

#### Minimal Setup (Single VM in Availability Set)

```bash
# 1. Create resource group and VM
az group create --name recrag-rg --location eastus
az vm create \
  --resource-group recrag-rg \
  --name recrag-vm \
  --image UbuntuLTS \
  --size Standard_E4s_v3 \
  --admin-username azureuser

# 2. SSH and deploy
ssh azureuser@<vm-ip>

# 3. Install and run
sudo apt update && sudo apt install -y python3.12 nodejs docker.io docker-compose
git clone <repo> && cd RecRAG
make docker-up
```

#### Resilient Setup (Multi-AZ with App Service)

```bash
# Use Azure App Service for stateless services
# - API containers on App Service
# - Frontend on Static Web Apps (CDN included)
# - Azure SQL Database for metadata
# - Blob Storage for PDFs
# - Application Gateway with WAF
```

**Azure Specifics**:
- **Virtual Machine Scale Sets**: Auto-scaling
- **Azure DevOps**: Built-in CI/CD
- **Application Insights**: APM and monitoring
- **Key Vault**: Secret management

---

## Cloud Provider Comparison

### Feature Matrix

| Feature | AWS | GCP | Azure |
|---------|-----|-----|-------|
| **VM Start Time** | 2-3 min | 1-2 min | 2-3 min |
| **Load Balancer Cost** | $25/month | $18/month | $20/month |
| **Container Support** | ECS, EKS | GKE, Cloud Run | AKS, Container Instances |
| **Serverless Option** | AWS Lambda, Fargate | Cloud Run | Functions, Container Instances |
| **Managed Vector DB** | Amazon OpenSearch (not Milvus) | Vertex AI Vector Search | Not native |
| **Free Tier** | Yes (1 year) | Yes (3 months) | Yes (1 month) |
| **Community Adoption** | Largest | Growing | Moderate |
| **Documentation Quality** | Excellent | Excellent | Very Good |
| **Pricing Transparency** | Clear | Clear | Clear |
| **Support Quality** | Enterprise | Enterprise | Enterprise |

### Cost Comparison (Monthly, Dev Tier)

Assuming:
- 2 API servers
- 1 frontend
- 1 managed Milvus (Zilliz Cloud): ~$50
- 10GB storage
- 1 TB bandwidth

| Component | AWS | GCP | Azure |
|-----------|-----|-----|-------|
| **Compute** | $140 | $120 | $150 |
| **Load Balancer** | $25 | $18 | $20 |
| **Storage** | $10 | $8 | $10 |
| **Managed Milvus** | $50 | $50 | $50 |
| **Database (optional)** | $15 | $10 | $5 |
| **Monitoring** | $5 | $5 | $5 |
| **Total** | **$245** | **$211** | **$240** |

### Recommendation by Use Case

| Scenario | Recommendation | Reason |
|----------|---|---|
| **Solo developer** | AWS Free Tier or local | Start free, easy to scale |
| **Small team (2-5 people)** | GCP | Better pricing, excellent docs |
| **Enterprise adoption** | AWS | Largest market share, most integrations |
| **Microsoft stack shop** | Azure | Tight integration with O365, AD |
| **Cost-conscious startup** | GCP | Lower per-hour rates |

---

## Local Development Deployment

### Prerequisites

```bash
# macOS
brew install python@3.12 node docker
# or
conda create -n recrag python=3.12 nodejs

# Linux (Ubuntu/Debian)
sudo apt install -y python3.12 python3.12-venv nodejs npm docker.io

# Windows
# - Install WSL2
# - Install Ubuntu 22.04 from Microsoft Store
# - Follow Linux instructions
```

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/hpe/recrag.git
cd RecRAG

# 2. Install dependencies (uses uv for faster resolution)
make install

# 3. Set up environment
make setup
# - Creates .env from .env.example
# - Prompts for API keys (OpenAI, Ollama, etc.)

# 4. Start development services
make dev
# Launches:
# - Ingestion API on :8001
# - Retrieval API on :8000
# - Frontend on :3000

# 5. (Optional) Start local Milvus
make infra-up
# - Starts etcd, MinIO, Milvus in Docker
```

### Accessing Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Web UI |
| **Ingestion API** | http://localhost:8001/docs | Swagger UI for uploads |
| **Retrieval API** | http://localhost:8000/docs | Swagger UI for queries |
| **Milvus (if running)** | http://localhost:19530 | gRPC endpoint |

### Environment Configuration

Create `.env` from template:

```bash
cp .env.example .env
```

Edit `.env` for your setup:

```env
# LLM Configuration
OPENAI_API_KEY=sk-...
OLLAMA_BASE_URL=http://localhost:11434

# Milvus (if running locally)
MILVUS_HOST=localhost
MILVUS_PORT=19530

# (Optional) Cloud Milvus credentials
MILVUS_USERNAME=user@zilliz
MILVUS_PASSWORD=password123
```

### Development Workflow

```bash
# Watch for code changes and test
make lint          # Run Ruff linting
make format        # Auto-format with Ruff
make typecheck     # Run mypy
make test          # Run pytest suite

# Ingest test documents
make ingest        # Ingests PDFs from data/pdfs/

# Run evaluation
make evaluate      # RAGAS evaluation on eval_dataset.json

# Full validation
make check         # lint + test + build
```

---

## Troubleshooting & Monitoring

### Common Issues

#### 1. Milvus Connection Failed

```bash
# Check if local Milvus is running
docker ps | grep milvus

# If not, start it
make infra-up

# Verify connection
python3 -c "from pymilvus import connections; connections.connect(host='localhost', port=19530); print('OK')"
```

#### 2. Port Already in Use

```bash
# Find process using port (e.g., 8000)
lsof -i :8000
kill -9 <PID>

# Or change port in docker-compose.yml
```

#### 3. Out of Memory

```bash
# Check memory usage
docker stats

# Increase Docker memory limit:
# Docker Desktop → Preferences → Resources → Memory: 8GB+
```

#### 4. PDF Upload Fails

```bash
# Check permissions
ls -la data/pdfs/
chmod 755 data/pdfs/

# Verify file size (max: typically 100MB, check config)
du -sh data/pdfs/*
```

### Monitoring in Development

#### Local Metrics

```bash
# Install monitoring tools
pip install prometheus-client grafana

# View API logs
docker logs -f recrag-api-ingestion
docker logs -f recrag-api-retrieval

# Check database status
curl http://localhost:8001/health
curl http://localhost:8000/health
```

#### Cloud Monitoring

**AWS CloudWatch**:
```bash
# View logs
aws logs tail /aws/ecs/recrag --follow
```

**GCP Cloud Monitoring**:
```bash
# View metrics
gcloud monitoring metrics-descriptors list
gcloud logging read "resource.type=gce_instance" --limit 50
```

**Azure Monitor**:
```bash
# View logs
az monitor log-analytics query \
  --workspace "recrag-logs" \
  --analytics-query "print 'Hello, Analytics!'"
```

### Performance Tuning

| Bottleneck | Solution | Expected Improvement |
|-----------|----------|----------------------|
| **PDF parsing slow** | Enable vision extraction for scanned PDFs | +50% throughput |
| **Query latency high** | Increase Milvus search_threads in config | +30% responsiveness |
| **API response slow** | Add caching layer (Redis) | +40% for repeat queries |
| **Database queries slow** | Add indexes on frequently filtered columns | +60% for status queries |
| **Vector search slow** | Optimize embedding model or batch size | +25% throughput |

---

## Summary

| Deployment | Setup Time | Cost | Best For |
|-----------|-----------|------|----------|
| **Local Dev** | 15 min | $0 | Single developer |
| **Docker Compose** | 20 min | $0-5 | Small team |
| **AWS Single VM** | 30 min | ~$90/mo | Fast scaling |
| **GCP Single VM** | 30 min | ~$76/mo | Best pricing |
| **Azure Single VM** | 30 min | ~$120/mo | Microsoft stack |
| **Multi-AZ (any)** | 60 min | ~$200+/mo | Production-ready |

**Recommended Path**:
1. Start with **local development** for initial work
2. Move to **Docker Compose** for team collaboration
3. Deploy to **GCP** for cost-effective staging
4. Use **AWS** for production with multi-AZ failover

---

## Next Steps

- [ ] Choose deployment target (local, cloud, or both)
- [ ] Set up monitoring and alerting
- [ ] Create CI/CD pipeline for automated deployments
- [ ] Configure auto-scaling policies
- [ ] Document runbooks for common operations
- [ ] Plan backup and disaster recovery strategy

For production deployment guidance, see `DEPLOYMENT_GUIDE_PROD.md` (coming soon).
