# Project Context

## Overview
Pet behavior monitoring system. Classifies pet activity from raw IMU sensor windows using a CNN+BiLSTM model.
Inference is processed asynchronously via Redis + Celery worker. Results are stored in PostgreSQL.

## Architecture
- **API**: FastAPI — accepts sensor windows, enqueues jobs, returns job status
- **Queue**: Redis + Celery — async ML inference worker
- **Database**: PostgreSQL (SQLite for local dev)
- **Model**: CNN+BiLSTM on IMU channels with ODBA features

## Features Implemented
- [x] Data ingestion and cleaning (notebook)
- [x] Windowing and dog-level split (notebook)
- [x] Model training and evaluation (notebook)
- [x] Explainability plots (notebook)
- [x] FastAPI app layer (app/)
- [x] Async job queue (Redis + Celery)
- [x] Job persistence (SQLAlchemy + PostgreSQL)
- [x] Docker Compose stack
- [x] CI pipeline (GitHub Actions)
- [x] Test suite (unit + integration + queue, 70%+ coverage)
- [ ] Frontend UI
- [ ] WebSocket real-time updates

## Decisions Made
- Inference is async — `/predict` returns a job_id, client polls `/jobs/{id}`
- SQLite used as default local DB; Postgres in Docker/production
- Single Dockerfile for both API and worker (different CMD)
- Dog-level splits to avoid identity leakage in model training

## Branch Strategy
- `main` → production
- `develop` → staging
- `feature/*` → new features
- `fix/*` → bug fixes

## Setup Instructions
```bash
# Full stack
docker-compose up

# Local dev (no Docker)
pip install -r requirements.txt
uvicorn app.main:app --reload

# Run tests
pytest tests/ --cov=app
```

See `docs/context.md` for full system documentation.
