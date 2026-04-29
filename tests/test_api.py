from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.database import JobStatus, PredictionJob


def _make_window(timesteps: int = 200, channels: int = 14) -> list[list[float]]:
    return [[float(i % 10)] * channels for i in range(timesteps)]


def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_submit_predict_returns_job_id(client):
    with patch('app.main.run_prediction') as mock_task:
        mock_task.delay.return_value = MagicMock(id='celery-task-id')
        response = client.post('/predict', json={'window': _make_window(), 'top_k': 3})

    assert response.status_code == 202
    data = response.json()
    assert 'job_id' in data
    assert data['status'] == 'pending'


def test_submit_predict_enqueues_task(client):
    with patch('app.main.run_prediction') as mock_task:
        mock_task.delay.return_value = MagicMock()
        client.post('/predict', json={'window': _make_window(), 'top_k': 2})
        mock_task.delay.assert_called_once()
        _, window_arg, top_k_arg = mock_task.delay.call_args[0]
        assert top_k_arg == 2
        assert len(window_arg) == 200


def test_get_job_not_found(client):
    response = client.get('/jobs/nonexistent-id')
    assert response.status_code == 404


def test_get_job_pending(client, db_session):
    job = PredictionJob(id='pending-job-1', status=JobStatus.pending, top_k_requested=3)
    db_session.add(job)
    db_session.commit()

    response = client.get('/jobs/pending-job-1')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'pending'
    assert data['result'] is None


def test_get_job_done(client, db_session):
    result_payload = {
        'predicted_index': 2,
        'predicted_class': 'galloping',
        'confidence': 0.87,
        'top_k': [{'class_index': 2, 'class_name': 'galloping', 'probability': 0.87}],
        'window_size': 200,
        'model_ready': True,
    }
    job = PredictionJob(
        id='done-job-1',
        status=JobStatus.done,
        top_k_requested=3,
        result=json.dumps(result_payload),
    )
    db_session.add(job)
    db_session.commit()

    response = client.get('/jobs/done-job-1')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'done'
    assert data['result']['predicted_class'] == 'galloping'
    assert data['result']['confidence'] == pytest.approx(0.87)


def test_get_job_failed(client, db_session):
    job = PredictionJob(
        id='failed-job-1',
        status=JobStatus.failed,
        top_k_requested=3,
        error='Model checkpoint not found',
    )
    db_session.add(job)
    db_session.commit()

    response = client.get('/jobs/failed-job-1')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'failed'
    assert 'checkpoint' in data['error']


def test_predict_top_k_out_of_range(client):
    with patch('app.main.run_prediction'):
        response = client.post('/predict', json={'window': _make_window(), 'top_k': 0})
    assert response.status_code == 422


def test_predict_top_k_max(client):
    with patch('app.main.run_prediction') as mock_task:
        mock_task.delay.return_value = MagicMock()
        response = client.post('/predict', json={'window': _make_window(), 'top_k': 10})
    assert response.status_code == 202
