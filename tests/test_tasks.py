from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, JobStatus, PredictionJob


@pytest.fixture(scope='module')
def task_session_factory():
    engine = create_engine(
        'sqlite://',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return factory


def _make_window(timesteps: int = 200, channels: int = 14) -> list[list[float]]:
    rng = np.random.default_rng(7)
    return rng.standard_normal((timesteps, channels)).tolist()


def _mock_predictor(predicted_class: str = 'sitting', confidence: float = 0.91) -> MagicMock:
    result = MagicMock()
    result.predicted_index = 0
    result.predicted_class = predicted_class
    result.confidence = confidence
    result.top_k = [{'class_index': 0, 'class_name': predicted_class, 'probability': confidence}]

    predictor = MagicMock()
    predictor.ready = True
    predictor.window_size = 200
    predictor.predict.return_value = result
    return predictor


def test_task_marks_job_done(task_session_factory):
    from app.tasks import run_prediction

    db = task_session_factory()
    job = PredictionJob(id='task-done-1', top_k_requested=3)
    db.add(job)
    db.commit()
    db.close()

    with patch('app.tasks.SessionLocal', task_session_factory):
        with patch('app.tasks._get_predictor', return_value=_mock_predictor()):
            result = run_prediction('task-done-1', _make_window(), 3)

    assert result['predicted_class'] == 'sitting'
    assert result['confidence'] == pytest.approx(0.91)

    db = task_session_factory()
    job = db.query(PredictionJob).filter(PredictionJob.id == 'task-done-1').first()
    assert job.status == JobStatus.done
    stored = json.loads(job.result)
    assert stored['predicted_class'] == 'sitting'
    db.close()


def test_task_marks_job_failed_on_error(task_session_factory):
    from app.tasks import run_prediction

    db = task_session_factory()
    job = PredictionJob(id='task-fail-1', top_k_requested=3)
    db.add(job)
    db.commit()
    db.close()

    bad_predictor = MagicMock()
    bad_predictor.predict.side_effect = RuntimeError('Model checkpoint missing')

    with patch('app.tasks.SessionLocal', task_session_factory):
        with patch('app.tasks._get_predictor', return_value=bad_predictor):
            with pytest.raises(RuntimeError, match='checkpoint'):
                run_prediction('task-fail-1', _make_window(), 3)

    db = task_session_factory()
    job = db.query(PredictionJob).filter(PredictionJob.id == 'task-fail-1').first()
    assert job.status == JobStatus.failed
    assert 'checkpoint' in job.error
    db.close()


def test_task_sets_processing_status_before_inference(task_session_factory):
    from app.tasks import run_prediction

    db = task_session_factory()
    job = PredictionJob(id='task-processing-1', top_k_requested=3)
    db.add(job)
    db.commit()
    db.close()

    observed_statuses: list[str] = []

    def _spy_predictor():
        db = task_session_factory()
        job = db.query(PredictionJob).filter(PredictionJob.id == 'task-processing-1').first()
        observed_statuses.append(job.status.value)
        db.close()
        return _mock_predictor()

    with patch('app.tasks.SessionLocal', task_session_factory):
        with patch('app.tasks._get_predictor', side_effect=_spy_predictor):
            run_prediction('task-processing-1', _make_window(), 3)

    assert 'processing' in observed_statuses


def test_enqueue_delay_called_with_correct_args():
    from app.tasks import run_prediction

    with patch.object(run_prediction, 'delay') as mock_delay:
        run_prediction.delay('job-xyz', [[0.0] * 14] * 200, 5)
        mock_delay.assert_called_once_with('job-xyz', [[0.0] * 14] * 200, 5)
