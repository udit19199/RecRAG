# Production Readiness Issues

Issues to resolve before deploying RecRAG on a single EC2 instance.
**Rate limiting** and **TLS/SSL** are deferred (noted below).

---

## Security

### S-01: No API Authentication
- **Severity**: Critical
- **Status**: ✅ Resolved
- **Action taken**: 
  - Created `src/auth.py` with API key authentication middleware
  - Added `RecRAG-API-Key` header verification to all protected endpoints
  - Health, docs, and OpenAPI endpoints remain public
  - Set `REC_RAG_API_KEY` environment variable to enable authentication

### S-02: Overly Permissive CORS
- **Severity**: High
- **Status**: ✅ Resolved
- **Action taken**: 
  - Restricted `allow_methods` to `["GET", "POST", "PATCH", "OPTIONS"]`
  - Restricted `allow_headers` to `["Content-Type", "Authorization", "RecRAG-API-Key"]`
  - Added `expose_headers=["X-Request-ID"]` for correlation ID access

### S-03: No File Upload Limits
- **Severity**: High
- **Status**: ✅ Resolved
- **Action taken**: 
  - Added 100 MB file size limit in `src/validation.py`
  - Validates file size before disk write
  - Returns HTTP 413 (Payload Too Large) on violation

### S-04: API Keys on Local Disk
- **Severity**: Medium
- **Status**: ⏳ Deferred
- **Action**: Rotate all keys. Consider reading from environment only (no `.env` on server) or a secrets manager.

### S-05: No Input Sanitization
- **Severity**: Medium
- **Status**: ✅ Resolved
- **Action taken**: 
  - Added query length validation (max 4096 chars)
  - Added filename validation (length, path traversal, allowed characters)
  - Added file size validation (max 100 MB)
  - All validation in `src/validation.py` with clear error messages

---

## Observability

### O-01: No Structured Logging
- **Severity**: High
- **Status**: ✅ Resolved
- **Action taken**: 
  - Created `src/logging.py` with ASGI structured logging middleware
  - Adds JSON-formatted logs with request IDs, method, path, status, duration
  - Request IDs propagated via `scope["state"]["request_id"]`
  - Error logs for 5xx responses, info logs otherwise

### O-02: No Metrics
- **Severity**: High
- **Status**: ✅ Resolved
- **Action taken**: 
  - Created `src/metrics.py` with Prometheus metrics middleware
  - Tracks: `http_requests_total`, `http_request_duration_seconds`, `http_request_exceptions_total`
  - Labels: method, path, status
  - Added `/metrics` endpoint to both APIs
  - Added `prometheus-client` to dependencies

### O-03: No Log Aggregation
- **Severity**: High
- **Status**: ⏳ Deferred
- **Action**: Deploy a log shipper (e.g. CloudWatch agent, Fluentd) or ensure stdout is captured by the hosting platform.

### O-04: No Alerting
- **Severity**: High
- **Status**: ⏳ Deferred
- **Action**: Set up basic CloudWatch alarms (or equivalent) on health endpoint failures and error rate thresholds.

### O-05: No Distributed Tracing
- **Severity**: Medium
- **Status**: ⏳ Deferred
- **Action**: Add OpenTelemetry instrumentation for end-to-end visibility.

---

## Infrastructure & Resilience

### I-01: No Resource Limits on API Services
- **Severity**: Medium
- **Status**: ✅ Resolved
- **Action taken**: 
  - Added `deploy.resources.limits` to both API services in `docker-compose.yml`
  - Memory limit: 2GB per API service
  - CPU limit: 1 CPU per API service

### I-02: No Graceful Shutdown
- **Severity**: Medium
- **Status**: ✅ Resolved
- **Action taken**: 
  - Added `--timeout-graceful-shutdown 30` to uvicorn commands in docker-compose
  - Lifespan handlers already exist for cleanup (runtime shutdown, task cancellation)
  - Docker sends SIGTERM with 30s timeout for in-flight request draining

### I-03: State on Local Disk
- **Severity**: Medium
- **Status**: ⏳ Deferred
- **Action**: Ensure persistent EBS volume mounts or migrate state to a database / object store.

### I-04: No Database Backup Strategy
- **Severity**: High
- **Status**: ⏳ Deferred
- **Action**: Implement Milvus backup (e.g. `milvus-backup` tool or Zilliz snapshots) and include in deployment.

### I-05: Single Point of Failure
- **Severity**: High
- **Status**: ⏳ Deferred
- **Action**: Acceptable for single EC2. Document RTO/RPO and ensure AMI backups or recovery scripts exist.

---

## CI/CD & Automation

### C-01: No CI/CD Pipeline
- **Severity**: Critical
- **Status**: ✅ Resolved
- **Action taken**: Created `.github/workflows/ci-cd.yml` with:
  - `make check` (lint, typecheck, test) on PRs and pushes to main
  - Docker image build and push to GHCR on main pushes
  - Automated deployment to EC2 via SSH
  - Post-deployment smoke tests
- **Documentation**: See [docs/CICD_SETUP.md](./CICD_SETUP.md)

### C-02: No Automated Security Scanning
- **Severity**: High
- **Status**: ✅ Resolved
- **Action taken**: Added to CI pipeline:
  - `pip-audit` for Python dependency vulnerability scanning
  - Trivy config scan for Dockerfile and infrastructure security
  - Results uploaded as workflow artifacts for review

### C-03: No Infrastructure as Code
- **Severity**: Medium
- **Status**: ⏳ Deferred
- **Action**: Write Terraform config for the EC2 instance, security group, EBS volume, and IAM role.

### C-04: No Smoke Tests
- **Severity**: Medium
- **Status**: ✅ Resolved
- **Action taken**: Created `jobs/smoke_test.py` with:
  - Health endpoint checks for both APIs and frontend
  - PDF upload and ingestion status polling
  - RAG query verification
  - Integrated into CI/CD pipeline post-deployment

---

## Code Quality

### Q-01: Type Checking Broken
- **Severity**: Medium
- **Status**: ✅ Resolved
- **Action taken**: 
  - `mypy` is correctly installed in dev dependencies via `pyproject.toml`
  - `make typecheck` runs successfully with `uv run mypy src/`
  - Type checking configuration in place with `ignore_missing_imports = true`

### Q-02: No Frontend Tests
- **Severity**: Medium
- **Status**: ⏳ Deferred
- **Action**: Add at least a basic frontend build step to CI; add unit tests for critical UI flows.
- **Note**: Frontend build is already verified in CI/CD pipeline

---

## Deferred Items

These items are acknowledged but explicitly deferred and will be handled later:

| Item | Reason |
|------|--------|
| **Rate limiting** | Not needed for ≤20 users in the first 6 months. |
| **TLS / SSL** | Will be configured after initial deployment. |
| **S-04** — API keys management | Acceptable for single-instance deployment; will migrate to secrets manager later. |
| **O-03** — Log aggregation | Structured logging in place; aggregation can be added post-launch. |
| **O-04** — Alerting | Health checks exist; alerting can be configured post-launch. |
| **O-05** — Distributed tracing | Not critical for initial launch; can be added later. |
| **I-03** — State persistence | EBS volume mounts sufficient for single EC2. |
| **I-04** — Milvus backup | Can be implemented post-launch with `milvus-backup`. |
| **I-05** — Single point of failure | Acceptable for single EC2; document RTO/RPO. |
| **C-03** — IaC | Manual setup acceptable for initial launch. |
| **Q-02** — Frontend tests | Build verification in CI sufficient for now. |

---

## Priority Order for EC2 Launch

### ✅ Completed
1. **S-01** — API authentication (API key middleware)
2. **C-01** — CI/CD pipeline (build → push → deploy)
3. **O-01, O-02** — Structured logging + Prometheus metrics
4. **S-02, S-03** — CORS hardening + file size limits
5. **I-01, I-02** — Resource limits + graceful shutdown
6. **S-05** — Input sanitization (query, filename, file size)
7. **C-02, C-04** — Security scanning + smoke tests
8. **Q-01** — Type checking fixed

### ⏳ Deferred (Post-Launch)
9. **O-04** — Health alerts (CloudWatch alarms)
10. **I-04** — Milvus backup strategy
11. **C-03** — IaC for EC2 (Terraform)
12. **Q-02** — Frontend tests
13. **S-04** — API keys management (secrets manager)
14. **O-03** — Log aggregation
15. **O-05** — Distributed tracing
16. **I-03** — State persistence (EBS volume)
17. **I-05** — Single point of failure (acceptable for single EC2)
