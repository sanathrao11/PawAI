# Deployment Guide

## AWS Infrastructure

### Services Used

| Service | Purpose |
|---|---|
| ECS Fargate | Serverless container compute |
| ECR | Docker image registry |
| RDS PostgreSQL | Managed database |
| ElastiCache Redis | Managed Redis queue |
| S3 | Model artifact storage |
| ALB | Load balancer for API |
| IAM | Roles and permissions |
| CloudWatch | Container logs |

---

## Setup Order

### 1. ECR Repositories

Create two private repositories:
- `pawai-api`
- `pawai-frontend`

### 2. RDS PostgreSQL

- Engine: PostgreSQL 16
- Instance: `db.t3.micro`
- DB name: `pawai`, user: `pawai`
- After creation, connect and run: `CREATE DATABASE pawai;`

### 3. ElastiCache Redis

- Engine: Redis OSS 7
- Mode: Cluster mode **disabled**
- Node: `cache.t3.micro`, 0 replicas
- Use primary endpoint in `REDIS_URL`

### 4. S3 Bucket

- Bucket: `pawai-bucket` (or any name)
- Upload: `best_model.pt` and `model_metadata.json`

### 5. IAM Roles

**ecsTaskExecutionRole** — attached policies:
- `AmazonECSTaskExecutionRolePolicy`
- `CloudWatchLogsFullAccess`
- `AmazonEC2ContainerRegistryReadOnly`

**ecsTaskRole** — attached policies:
- `AmazonS3ReadOnlyAccess`

### 6. ECS Cluster

- Name: `pawai-cluster`
- Infrastructure: AWS Fargate only

### 7. Task Definitions

**pawai-api** (0.5 vCPU, 1GB):
- Image: `<account>.dkr.ecr.<region>.amazonaws.com/pawai-api:latest`
- Port: 8000
- Task role: `ecsTaskRole`
- Env: `DATABASE_URL`, `REDIS_URL`

**pawai-worker** (1 vCPU, 2GB):
- Image: same as API
- Command: `/bin/sh,-c,aws s3 cp s3://<bucket>/best_model.pt /app/best_model.pt && aws s3 cp s3://<bucket>/model_metadata.json /app/model_metadata.json && celery -A app.celery_app.celery worker --loglevel=info --concurrency=2`
- Task role: `ecsTaskRole`
- Env: `DATABASE_URL`, `REDIS_URL`, `PET_BEHAVIOR_MODEL_PATH=/app/best_model.pt`, `PET_BEHAVIOR_METADATA_PATH=/app/model_metadata.json`

**pawai-frontend** (0.25 vCPU, 0.5GB):
- Image: `<account>.dkr.ecr.<region>.amazonaws.com/pawai-frontend:latest`
- Port: 80

### 8. ALB

- Name: `pawai-alb`, internet-facing
- Subnets: all 3 AZs
- Target group: `pawai-api-tg`, type IP, port 8000, health check `/health`

### 9. ECS Services

Create three services in `pawai-cluster`:

| Service | Task def | Desired | Load balancer |
|---|---|---|---|
| `pawai-api-service` | `pawai-api` | 1 | `pawai-alb` → `pawai-api-tg` |
| `pawai-worker-service` | `pawai-worker` | 1 | none |
| `pawai-frontend-service` | `pawai-frontend` | 1 | none |

Enable Service Connect on all services (namespace: `pawai`).

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to `main`, `develop`, and `feature/**` branches.

### Jobs

1. **test-backend** — ruff lint + pytest with 70% coverage gate
2. **test-frontend** — npm install + vite build
3. **build-and-push** (push events only) — builds API and frontend Docker images, pushes to ECR with SHA tag and `latest`
4. **deploy** (main branch only) — calls `aws ecs update-service --force-new-deployment` for all 3 services

### Required GitHub Secrets

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REPO_API
ECR_REPO_FRONTEND
ECS_CLUSTER
ECS_SERVICE_API
ECS_SERVICE_WORKER
ECS_SERVICE_FRONTEND
VITE_API_URL        (e.g. http://pawai-alb-xxx.eu-north-1.elb.amazonaws.com)
```

---

## Updating the Model

To deploy a new model checkpoint without a code change:

1. Export `best_model.pt` and `model_metadata.json` from notebook
2. Upload both to S3: `aws s3 cp best_model.pt s3://pawai-bucket/`
3. Force redeploy worker: `aws ecs update-service --cluster pawai-cluster --service pawai-worker-service --force-new-deployment`

Worker downloads fresh files from S3 on every restart.
