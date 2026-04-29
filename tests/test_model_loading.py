from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_config(tmp_path: Path, *, checkpoint: bool = False, metadata: bool = False, bad_json: bool = False):
    from app.config import AppConfig

    cp = tmp_path / 'model.pt'
    meta = tmp_path / 'meta.json'

    if checkpoint:
        cp.write_bytes(b'fake')
    if metadata:
        if bad_json:
            meta.write_text('not json', encoding='utf-8')
        else:
            meta.write_text(
                json.dumps({'class_names': ['sitting', 'walking'], 'window_size': 200}),
                encoding='utf-8',
            )

    config = AppConfig.__new__(AppConfig)
    for attr, val in [
        ('model_path', cp), ('metadata_path', meta), ('window_size', 200),
        ('stride', 100), ('use_odba', True), ('use_time_frequency', False),
        ('use_attention', True), ('tf_method', 'cwt'), ('cwt_w', 5.0),
        ('cwt_scales', tuple(range(1, 17))),
        ('feature_columns', tuple(f'c{i}' for i in range(14))),
        ('class_names', ('sitting', 'walking')), ('device', 'cpu'),
    ]:
        object.__setattr__(config, attr, val)
    return config


# --- model_status unit tests ---

def test_both_missing(tmp_path):
    from app.model_status import check_model_status
    status = check_model_status(_make_config(tmp_path))
    assert not status.checkpoint_exists
    assert not status.metadata_exists
    assert not status.ready
    assert status.error is not None


def test_checkpoint_missing_metadata_present(tmp_path):
    from app.model_status import check_model_status
    status = check_model_status(_make_config(tmp_path, metadata=True))
    assert not status.checkpoint_exists
    assert status.metadata_exists
    assert status.metadata_valid
    assert not status.ready
    assert 'missing' in status.error.lower()


def test_checkpoint_present_metadata_missing(tmp_path):
    from app.model_status import check_model_status
    status = check_model_status(_make_config(tmp_path, checkpoint=True))
    assert status.checkpoint_exists
    assert not status.metadata_exists
    assert not status.ready


def test_both_present_and_valid(tmp_path):
    from app.model_status import check_model_status
    status = check_model_status(_make_config(tmp_path, checkpoint=True, metadata=True))
    assert status.checkpoint_exists
    assert status.metadata_exists
    assert status.metadata_valid
    assert status.ready
    assert status.class_names == ['sitting', 'walking']
    assert status.window_size == 200
    assert status.error is None


def test_bad_metadata_json(tmp_path):
    from app.model_status import check_model_status
    status = check_model_status(_make_config(tmp_path, checkpoint=True, metadata=True, bad_json=True))
    assert not status.metadata_valid
    assert not status.ready
    assert status.error is not None


# --- /health endpoint ---

def test_health_model_not_ready(client):
    from app.model_status import ModelFileStatus
    mock_status = ModelFileStatus(
        checkpoint_exists=False, metadata_exists=False,
        checkpoint_path='best_model.pt', metadata_path='model_metadata.json',
        metadata_valid=False, class_names=None, window_size=None,
        error='Checkpoint missing: best_model.pt',
    )
    with patch('app.main.check_model_status', return_value=mock_status):
        response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    assert response.json()['model_ready'] is False


def test_health_model_ready(client):
    from app.model_status import ModelFileStatus
    mock_status = ModelFileStatus(
        checkpoint_exists=True, metadata_exists=True,
        checkpoint_path='best_model.pt', metadata_path='model_metadata.json',
        metadata_valid=True, class_names=['sitting', 'walking'],
        window_size=200, error=None,
    )
    with patch('app.main.check_model_status', return_value=mock_status):
        response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['model_ready'] is True


# --- /model/status endpoint ---

def test_model_status_not_ready(client):
    from app.model_status import ModelFileStatus
    mock_status = ModelFileStatus(
        checkpoint_exists=False, metadata_exists=True,
        checkpoint_path='best_model.pt', metadata_path='model_metadata.json',
        metadata_valid=True, class_names=['sitting'], window_size=200,
        error='Checkpoint missing: best_model.pt',
    )
    with patch('app.main.check_model_status', return_value=mock_status):
        response = client.get('/model/status')
    assert response.status_code == 200
    data = response.json()
    assert data['ready'] is False
    assert data['checkpoint_exists'] is False
    assert data['metadata_exists'] is True
    assert 'missing' in data['error'].lower()


def test_model_status_ready(client):
    from app.model_status import ModelFileStatus
    mock_status = ModelFileStatus(
        checkpoint_exists=True, metadata_exists=True,
        checkpoint_path='best_model.pt', metadata_path='model_metadata.json',
        metadata_valid=True, class_names=['sitting', 'walking', 'galloping'],
        window_size=200, error=None,
    )
    with patch('app.main.check_model_status', return_value=mock_status):
        response = client.get('/model/status')
    assert response.status_code == 200
    data = response.json()
    assert data['ready'] is True
    assert data['class_names'] == ['sitting', 'walking', 'galloping']
    assert data['window_size'] == 200
    assert data['error'] is None
