from __future__ import annotations

import json
import logging

from celery.signals import worker_ready

from .celery_app import celery
from .config import AppConfig
from .database import JobStatus, PredictionJob, SessionLocal
from .model_status import check_model_status
from .predictor import BehaviorPredictor

logger = logging.getLogger(__name__)

_predictor: BehaviorPredictor | None = None


@worker_ready.connect
def on_worker_ready(**kwargs) -> None:
    status = check_model_status(AppConfig())
    if status.ready:
        logger.info('Model ready. checkpoint=%s', status.checkpoint_path)
    else:
        logger.warning('Model NOT ready: %s', status.error)


def _get_predictor() -> BehaviorPredictor:
    global _predictor
    if _predictor is None:
        _predictor = BehaviorPredictor(AppConfig())
    return _predictor


@celery.task(name='app.tasks.run_prediction')
def run_prediction(job_id: str, window: list[list[float]], top_k: int) -> dict:
    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job is None:
            raise ValueError(f'Job {job_id} not found in database')

        job.status = JobStatus.processing
        db.commit()

        predictor = _get_predictor()
        result = predictor.predict(window, top_k=top_k)

        result_data = {
            'predicted_index': result.predicted_index,
            'predicted_class': result.predicted_class,
            'confidence': result.confidence,
            'top_k': result.top_k,
            'window_size': predictor.window_size,
            'model_ready': predictor.ready,
        }

        job.status = JobStatus.done
        job.result = json.dumps(result_data)
        db.commit()

        return result_data

    except Exception as exc:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.status = JobStatus.failed
            job.error = str(exc)
            db.commit()
        raise

    finally:
        db.close()
