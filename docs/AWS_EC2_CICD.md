# AWS EC2 CI/CD Deployment

This repository now includes a lightweight production path for a small team:

- **CI** runs on GitHub Actions for backend and frontend checks.
- **CD** builds Docker images, pushes them to **Amazon ECR**, then restarts the
  stack on a single **EC2** instance with Docker Compose.

For a workload of roughly 10 users, this is usually a better operational fit
than Kubernetes. It keeps the moving parts small while preserving automated,
repeatable deploys.

## Architecture

The production EC2 stack uses `docker-compose.ec2.yml` and runs:

- `frontend` on port `3000`
- `api-retrieval` on port `8000`
- `api-ingestion` on port `8001`
- `orchestrator` on port `8002`
- `milvus` on ports `19530` and `9091`

The frontend talks to the public retrieval and ingestion endpoints using the
build-time `NEXT_PUBLIC_*` variables, while server-side Next.js routes proxy to
the orchestrator over the internal Docker network.

## 1. AWS prerequisites

Create or prepare:

1. An **EC2 instance** (Ubuntu works well)
2. Two **ECR repositories**
   - one for the API image
   - one for the frontend image
3. An **IAM role for GitHub Actions** with permission to push to ECR
4. An **instance profile for EC2** with permission to pull from ECR

Suggested EC2 security group rules:

- `3000/tcp` from trusted client IPs
- `8000/tcp` from trusted client IPs
- `8001/tcp` from trusted client IPs
- `8002/tcp` only if you want direct orchestrator access
- `22/tcp` from your admin IPs or bastion host

If you plan to put Nginx, Caddy, or an ALB in front later, you can tighten
these rules and only expose `80/443`.

## 2. One-time EC2 bootstrap

SSH into the instance and run:

```bash
git clone https://github.com/udit19199/RecRAG.git
cd RecRAG
bash scripts/ec2/bootstrap.sh
```

The bootstrap script:

- installs Docker
- installs the Docker Compose plugin
- installs the AWS CLI
- prepares `/opt/recrag`
- clones or refreshes the repo there
- creates `/opt/recrag/.env` from `.env.example` if needed

Then edit:

```bash
nano /opt/recrag/.env
```

At minimum, set the provider keys you use in production. If you use managed
Milvus instead of the local Compose service, set:

- `MILVUS_HOST`
- `MILVUS_USERNAME`
- `MILVUS_PASSWORD`

## 3. GitHub repository variables and secrets

### Secrets

Set these GitHub Actions **secrets**:

- `AWS_ROLE_TO_ASSUME`
- `EC2_HOST`
- `EC2_USERNAME`
- `EC2_SSH_KEY`

### Variables

Set these GitHub Actions **variables**:

- `AWS_REGION`
- `ECR_API_REPOSITORY`
- `ECR_FRONTEND_REPOSITORY`
- `EC2_APP_DIR` (recommended: `/opt/recrag`)
- `PRODUCTION_RETRIEVAL_API_URL` (example: `http://YOUR_EC2_IP:8000`)
- `PRODUCTION_INGESTION_API_URL` (example: `http://YOUR_EC2_IP:8001`)

The frontend image is built with the public API URLs baked in, so these
variables must match the URLs your users will actually reach in production.

## 4. How the pipeline works

### Continuous integration

`.github/workflows/frontend-ci.yml` now runs:

- backend Ruff
- backend mypy
- backend pytest
- frontend lint
- React Doctor
- frontend build

### Continuous deployment

`.github/workflows/deploy-ec2.yml` runs on pushes to `main` and:

1. Assumes the AWS role with OIDC
2. Logs in to ECR
3. Builds the API image from `Dockerfile --target api`
4. Builds the frontend image from `frontend/Dockerfile`
5. Pushes both images with `latest` and commit-SHA tags
6. Uploads the production compose/config/deploy scripts to EC2
7. SSHes into EC2 and runs `scripts/ec2/deploy.sh`

The remote deploy script performs:

```bash
docker compose -f docker-compose.ec2.yml pull
docker compose -f docker-compose.ec2.yml up -d --remove-orphans
```

## 5. Manual deploy and rollback

You can also deploy manually on the server:

```bash
cd /opt/recrag
API_IMAGE=<account>.dkr.ecr.<region>.amazonaws.com/recrag-api:<sha> \
FRONTEND_IMAGE=<account>.dkr.ecr.<region>.amazonaws.com/recrag-frontend:<sha> \
AWS_REGION=<region> \
bash scripts/ec2/deploy.sh
```

To roll back, redeploy an older image tag by setting the older SHA-based image
references in the same command.

## 6. Operational notes

- This setup is intentionally optimized for **small-team simplicity**, not for
  high-availability orchestration.
- The EC2 host keeps runtime state on its attached volume via the Compose mounts.
- If you later outgrow a single instance, the clean next step is usually:
  **ALB + ECS** or **EKS**, not a more complicated single-VM script.
