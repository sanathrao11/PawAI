from __future__ import annotations

import json
import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .config import AppConfig
from .database import PredictionJob, get_db, init_db
from .model_status import check_model_status
from .schemas import (
    JobStatusResponse,
    JobSubmitResponse,
    ModelStatusResponse,
    PredictionItem,
    PredictionResponse,
    WindowPredictionRequest,
)
from .tasks import run_prediction

app = FastAPI(title='Pet Behavior Monitoring API', version='0.2.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type'],
)


@app.on_event('startup')
def startup() -> None:
    init_db()


@app.get('/health')
def health() -> dict[str, object]:
    status = check_model_status(AppConfig())
    return {'status': 'ok', 'model_ready': status.ready}


@app.get('/model/status', response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    status = check_model_status(AppConfig())
    return ModelStatusResponse(
        ready=status.ready,
        checkpoint_exists=status.checkpoint_exists,
        metadata_exists=status.metadata_exists,
        checkpoint_path=status.checkpoint_path,
        metadata_path=status.metadata_path,
        metadata_valid=status.metadata_valid,
        class_names=status.class_names,
        window_size=status.window_size,
        error=status.error,
    )


@app.post('/predict', response_model=JobSubmitResponse, status_code=202)
def submit_predict(
    request: WindowPredictionRequest,
    db: Session = Depends(get_db),
) -> JobSubmitResponse:
    job_id = str(uuid.uuid4())
    job = PredictionJob(id=job_id, top_k_requested=request.top_k)
    db.add(job)
    db.commit()
    run_prediction.delay(job_id, request.window, request.top_k)
    return JobSubmitResponse(job_id=job_id, status='pending')


@app.get('/jobs/{job_id}', response_model=JobStatusResponse)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobStatusResponse:
    job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')

    result: PredictionResponse | None = None
    if job.result:
        raw = json.loads(job.result)
        result = PredictionResponse(
            predicted_index=raw['predicted_index'],
            predicted_class=raw['predicted_class'],
            confidence=raw['confidence'],
            top_k=[PredictionItem(**item) for item in raw['top_k']],
            window_size=raw['window_size'],
            model_ready=raw['model_ready'],
        )

    return JobStatusResponse(
        job_id=job_id,
        status=job.status.value,
        result=result,
        error=job.error,
    )
