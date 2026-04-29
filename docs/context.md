# Project Context: PawAI — Pet Behavior Monitoring System

## Goal
Classify pet behavior in real-time from raw IMU (accelerometer + gyroscope) sensor windows.
Returns the most likely activity (sitting, galloping, etc.) along with confidence scores.

## Architecture

```
Client (POST /predict)
        ↓
FastAPI (submit job → DB, enqueue task)
        ↓
Redis (task broker + result backend)
        ↓
Celery Worker (loads model, runs inference)
        ↓
PostgreSQL (stores job status + results)
        ↑
Client polls GET /jobs/{job_id}
```

## Services

| Service | Technology | Purpose |
|---------|-----------|---------|
| API | FastAPI + uvicorn | Job submission, status polling |
| Queue Broker | Redis 7 | Async task queue |
| Worker | Celery 5 | ML inference |
| Database | PostgreSQL 16 | Job persistence |
| ML Model | CNN + BiLSTM (PyTorch) | Behavior classification |

## Key Concepts

**Async job processing**: Inference is slow — POSTing to `/predict` returns a `job_id` immediately (HTTP 202). The worker processes the job in the background. Clients poll `GET /jobs/{job_id}` until `status == "done"`.

**Sensor input**: Each request contains a 2D window of shape `(timesteps, channels)`.
- 12 raw IMU channels (accelerometer + gyroscope, back + neck sensors)
- 2 engineered ODBA channels (computed automatically if not provided)
- Default window size: 200 timesteps

**Time-frequency transform**: Optionally applies CWT (Continuous Wavelet Transform) before passing data to the CNN, converting each channel's signal into a 2D time-frequency representation.

**Model**: CNNBiLSTM — 1D or 2D CNN feature extractor → BiLSTM sequence model → attention-weighted context → classifier.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service liveness |
| POST | `/predict` | Submit inference job (async) |
| GET | `/jobs/{job_id}` | Poll job status and result |

## Current Features
- Async behavior classification from IMU windows
- Job queue with Redis + Celery
- PostgreSQL job persistence with status tracking
- Docker Compose stack (api + worker + redis + db)
- CI pipeline with lint, tests (70%+ coverage), and Docker build

## In Progress
- Real-time updates via WebSockets (remove need to poll)
- Frontend UI for sensor data upload and result display

## Future Goals
- Multi-pet recognition
- Health diagnostics from behavior patterns
- Model retraining pipeline

## Running Locally

```bash
# Full stack
docker-compose up

# API only (SQLite, no Redis needed for local dev)
uvicorn app.main:app --reload

# Worker only
celery -A app.celery_app.celery worker --loglevel=info

# Tests
pytest tests/ --cov=app
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker + backend |
| `DATABASE_URL` | `sqlite:///./pawai.db` | SQLAlchemy DB URL |
| `PET_BEHAVIOR_MODEL_PATH` | `best_model.pt` | Path to trained checkpoint |
| `PET_BEHAVIOR_METADATA_PATH` | `model_metadata.json` | Model config JSON |
| `PET_BEHAVIOR_DEVICE` | `cpu` | Inference device |
