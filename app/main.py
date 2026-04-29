from __future__ import annotations

import json
import uuid

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from .database import JobStatus, PredictionJob, get_db, init_db
from .schemas import (
    JobStatusResponse,
    JobSubmitResponse,
    PredictionItem,
    PredictionResponse,
    WindowPredictionRequest,
)
from .tasks import run_prediction

app = FastAPI(title='Pet Behavior Monitoring API', version='0.2.0')


@app.on_event('startup')
def startup() -> None:
    init_db()


@app.get('/health')
def health() -> dict[str, object]:
    return {'status': 'ok'}


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
