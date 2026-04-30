# PawAI — Pet Behavior Classification System

A production-grade AI system that classifies pet behavior from IMU (accelerometer + gyroscope) sensor data using a CNN+BiLSTM deep learning model. Requests are processed asynchronously via a Redis-backed Celery worker queue, deployed on AWS ECS Fargate with a full CI/CD pipeline.

---

## Architecture Overview

```
Browser (React)
      |
      | HTTP
      v
AWS ALB (Load Balancer)
      |
      v
FastAPI (ECS Fargate)
      |
      |-- POST /predict --> Redis Queue --> Celery Worker (ECS Fargate)
      |                                          |
      |                                          |--> Load model from S3
      |                                          |--> Run CNN+BiLSTM inference
      |                                          |--> Save result to PostgreSQL
      |
      |-- GET /jobs/{id} --> PostgreSQL --> Return result
```

### Request Flow

1. User uploads a CSV of sensor readings (or uses demo data) via the React frontend
2. Frontend POSTs the sensor window to `POST /predict` via the ALB
3. API creates a job record in PostgreSQL (status: `pending`) and enqueues a Celery task to Redis
4. API immediately returns `{ job_id, status: "pending" }` — HTTP 202
5. Frontend polls `GET /jobs/{job_id}` every 1.5 seconds
6. Celery worker picks up the task, downloads model files from S3, runs inference
7. Worker saves the prediction result to PostgreSQL (status: `done`)
8. Frontend displays the predicted behavior class and confidence scores

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Nginx |
| Backend API | FastAPI, Python 3.11, SQLAlchemy |
| Queue Broker | Redis 7 (AWS ElastiCache) |
| Worker | Celery 5 |
| ML Model | PyTorch — CNN + BiLSTM with attention |
| Database | PostgreSQL 16 (AWS RDS) |
| Model Storage | AWS S3 |
| Container Registry | AWS ECR |
| Compute | AWS ECS Fargate (3 services) |
| Load Balancer | AWS ALB |
| CI/CD | GitHub Actions |

---

## ML Model

**Architecture:** CNNBiLSTM — 1D CNN feature extractor → BiLSTM sequence model → attention-weighted pooling → classifier

**Input:** Sensor window of shape `(200 timesteps, 14 channels)`
- 12 raw IMU channels (accelerometer + gyroscope, back + neck sensors)
- 2 engineered ODBA (Overall Dynamic Body Acceleration) channels

**Output:** 15 behavior classes:
`Carrying object`, `Drinking`, `Eating`, `Galloping`, `Lying chest`, `Pacing`, `Panting`, `Playing`, `Shaking`, `Sitting`, `Sniffing`, `Standing`, `Synchronization`, `Trotting`, `Walking`

---

## Project Structure

```
PawAI/
├── app/                        # FastAPI backend
│   ├── main.py                 # API endpoints
│   ├── tasks.py                # Celery task definitions
│   ├── celery_app.py           # Celery configuration
│   ├── database.py             # SQLAlchemy models + session
│   ├── predictor.py            # Model inference wrapper
│   ├── model.py                # CNNBiLSTM architecture
│   ├── preprocessing.py        # Sensor data preprocessing
│   ├── model_status.py         # Model file validation
│   ├── schemas.py              # Pydantic request/response schemas
│   └── config.py               # Environment configuration
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js              # API client + CSV parser
│   │   └── components/
│   │       ├── ModelStatus.jsx
│   │       ├── PredictForm.jsx
│   │       └── ResultDisplay.jsx
│   ├── Dockerfile
│   └── nginx.conf
├── tests/                      # Test suite
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_predictor.py
│   ├── test_tasks.py
│   └── test_model_loading.py
├── docs/                       # Documentation
├── .github/workflows/ci.yml    # CI/CD pipeline
├── Dockerfile                  # API + worker image
├── docker-compose.yml          # Local development stack
└── requirements.txt
```

---

## Local Development

### Prerequisites

- Python 3.11
- Node.js 20
- Docker + Docker Compose

### Run the full stack

```bash
docker-compose up
```

Services:
- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- Redis: `localhost:6379`
- PostgreSQL: `localhost:5432`

### Run API only (SQLite, no Redis)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Run frontend

```bash
cd frontend
npm install
npm run dev
# http://localhost:5173
```

### Run worker

```bash
celery -A app.celery_app.celery worker --loglevel=info
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker + result backend |
| `DATABASE_URL` | `sqlite:///./pawai.db` | SQLAlchemy connection string |
| `PET_BEHAVIOR_MODEL_PATH` | `best_model.pt` | Path to model checkpoint |
| `PET_BEHAVIOR_METADATA_PATH` | `model_metadata.json` | Path to model metadata |
| `PET_BEHAVIOR_DEVICE` | `cpu` | Inference device (`cpu` or `cuda`) |
| `VITE_API_URL` | `` | Frontend API base URL (build-time) |

---

## API Reference

### `GET /health`

Returns service liveness and model status.

```json
{ "status": "ok", "model_ready": false }
```

### `GET /model/status`

Returns detailed model file validation.

```json
{
  "ready": true,
  "checkpoint_exists": true,
  "metadata_exists": true,
  "metadata_valid": true,
  "checkpoint_path": "/app/best_model.pt",
  "metadata_path": "/app/model_metadata.json",
  "class_names": ["Galloping", "Sitting", "Walking", ...],
  "window_size": 200,
  "error": null
}
```

### `POST /predict`

Submit a sensor window for async inference.

**Request:**
```json
{
  "window": [[0.1, 0.2, ..., 1.4], ...],
  "top_k": 3
}
```

**Response (HTTP 202):**
```json
{ "job_id": "uuid", "status": "pending" }
```

### `GET /jobs/{job_id}`

Poll job status and result.

```json
{
  "job_id": "uuid",
  "status": "done",
  "result": {
    "predicted_class": "Walking",
    "confidence": 0.70,
    "top_k": [
      { "class_name": "Walking", "probability": 0.70 },
      { "class_name": "Trotting", "probability": 0.18 },
      { "class_name": "Pacing", "probability": 0.07 }
    ],
    "window_size": 200,
    "model_ready": true
  },
  "error": null
}
```

Job status values: `pending` → `processing` → `done` / `failed`

---

## CI/CD Pipeline

```
push / PR to main or develop
         |
         ├── test-backend (parallel)
         │     lint (ruff) + pytest --cov ≥70%
         │
         └── test-frontend (parallel)
               npm install + vite build
                     |
               build-and-push (on push only)
                     Build API image → push to ECR
                     Build frontend image → push to ECR
                           |
                     deploy (main only)
                           Force new ECS deployment
                           api + worker + frontend services
```

### GitHub Secrets Required

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user for CI/CD |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `AWS_REGION` | AWS region |
| `ECR_REPO_API` | ECR repository name for API |
| `ECR_REPO_FRONTEND` | ECR repository name for frontend |
| `ECS_CLUSTER` | ECS cluster name |
| `ECS_SERVICE_API` | ECS API service name |
| `ECS_SERVICE_WORKER` | ECS worker service name |
| `ECS_SERVICE_FRONTEND` | ECS frontend service name |
| `VITE_API_URL` | Public API URL for frontend build |

---

## Testing

```bash
# Run all tests
pytest tests/ --cov=app --cov-report=term-missing

# Coverage target: 70%+
```

Test categories:
- **Unit tests** (`test_predictor.py`) — preprocessing, model loading, predictor
- **API tests** (`test_api.py`) — endpoints, request validation, job lifecycle
- **Queue tests** (`test_tasks.py`) — task execution, status transitions, error handling
- **Model loading tests** (`test_model_loading.py`) — checkpoint/metadata validation

---

## Deployment (AWS)

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for full step-by-step instructions.

**Live stack:**
- ALB: `http://pawai-alb-1023454956.eu-north-1.elb.amazonaws.com`
- Region: `eu-north-1`
- ECS cluster: `pawai-cluster`
- 3 Fargate services: `pawai-api-service`, `pawai-worker-service`, `pawai-frontend-service`

---

## Branching Strategy

```
main        → production (auto-deployed via CI/CD)
develop     → integration branch
feature/*   → new features (branch from develop)
fix/*       → bug fixes (branch from develop)
```

---

## Known Tradeoffs

- **Model not in API container** — the `/health` endpoint reports `model_ready: false` because the model lives in the worker, not the API. Predictions still work correctly.
- **Public IPs on Fargate tasks** — frontend IP changes on redeploy; stable URL requires ALB (API has one, frontend does not yet).
- **No HTTPS** — TLS termination not configured; production deployment should add an ACM certificate to the ALB.
- **Single worker** — one Celery worker with 2 concurrency threads. Scale by increasing `desired_count` on `pawai-worker-service`.
