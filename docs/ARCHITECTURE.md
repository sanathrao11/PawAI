# Architecture

## System Design

PawAI is designed around the principle that ML inference is slow and unpredictable in duration. Rather than blocking the HTTP request while the model runs, every prediction is processed asynchronously via a task queue. This decouples the API from the ML layer and makes both independently scalable.

## Component Breakdown

### API (FastAPI)

- Accepts sensor window submissions
- Creates job records in PostgreSQL
- Enqueues Celery tasks to Redis
- Returns `job_id` immediately (HTTP 202)
- Serves job status and results on poll

The API has zero ML dependencies at runtime. It never loads the model.

### Queue (Redis)

- Acts as Celery's broker (task dispatch) and result backend
- AWS ElastiCache Redis 7, cluster mode disabled (single-node)
- Tasks serialized as JSON

### Worker (Celery)

- Downloads `best_model.pt` and `model_metadata.json` from S3 on startup
- Loads the CNNBiLSTM model into memory once (cached in process)
- Processes prediction tasks: preprocessing → inference → save result
- Updates job status: `pending` → `processing` → `done` / `failed`

### Database (PostgreSQL)

Single table `prediction_jobs`:

```
id              UUID        primary key
status          ENUM        pending | processing | done | failed
top_k_requested INTEGER
result          TEXT        JSON-encoded prediction result
error           TEXT        error message if failed
created_at      DATETIME
updated_at      DATETIME
```

### Frontend (React + Nginx)

- Vite-built React SPA served by Nginx
- `VITE_API_URL` baked in at Docker build time
- Polls `GET /jobs/{id}` every 1.5 seconds until terminal status
- CSV parser validates 12 or 14 column sensor files
- Demo data generator creates synthetic 200-timestep windows

## ML Model Architecture

```
Input: (batch, 14 channels, 200 timesteps)
         |
    1D CNN Encoder
    Conv1d(14→32, k=7) → BN → ReLU → MaxPool
    Conv1d(32→64, k=5) → BN → ReLU → MaxPool
         |
    BiLSTM (hidden=128, layers=2, bidirectional)
    Output: (batch, timesteps, 256)
         |
    Attention Layer
    Linear(256→1) → Softmax → weighted sum
         |
    Context Vector (batch, 256)
         |
    Classifier
    Linear(256→128) → ReLU → Dropout → Linear(128→15)
         |
Output: logits (batch, 15 classes)
```

## Infrastructure Diagram

```
Internet
    |
    v
AWS ALB (pawai-alb)
Port 80, eu-north-1, all 3 AZs
    |
    v
Target Group (pawai-api-tg)
Port 8000, health check /health
    |
    v
ECS Fargate Service: pawai-api-service
Task: pawai-api (0.5 vCPU, 1GB RAM)
    |         |
    |         v
    |    AWS ElastiCache Redis
    |    (pawai-redis, cache.t3.micro)
    |         |
    |         v
    |    ECS Fargate Service: pawai-worker-service
    |    Task: pawai-worker (1 vCPU, 2GB RAM)
    |         |
    |         v
    |    AWS S3 (pawai-bucket)
    |    best_model.pt, model_metadata.json
    |
    v
AWS RDS PostgreSQL (pawai, db.t3.micro)


ECS Fargate Service: pawai-frontend-service
Task: pawai-frontend (0.25 vCPU, 0.5GB RAM)
Nginx serving React SPA
```

## Scaling Strategy

| Component | How to scale |
|---|---|
| API | Increase `desired_count` on `pawai-api-service` |
| Worker | Increase `desired_count` or `--concurrency` flag |
| Redis | Upgrade ElastiCache node tier |
| Database | Upgrade RDS instance, add read replicas |
| Frontend | Increase `desired_count`, or migrate to CloudFront + S3 |

## Security Notes

- All services run in the same VPC and security group — no public DB exposure
- IAM roles follow least-privilege: `ecsTaskExecutionRole` (ECR + CloudWatch), `ecsTaskRole` (S3 read-only)
- Model artifacts stored in private S3 bucket
- CORS configured on API to allow browser requests
- **Not yet implemented:** HTTPS (ACM certificate + ALB HTTPS listener), secrets in AWS Secrets Manager
