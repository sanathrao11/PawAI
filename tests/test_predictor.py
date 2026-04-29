from __future__ import annotations

import numpy as np
import pytest

from app.preprocessing import compute_odba, ensure_feature_matrix, transform_for_model


def _make_window(timesteps: int = 200, channels: int = 14) -> list[list[float]]:
    rng = np.random.default_rng(42)
    return rng.standard_normal((timesteps, channels)).tolist()


# --- preprocessing unit tests ---

def test_compute_odba_output_shape():
    window = np.random.default_rng(0).standard_normal((200, 12)).astype(np.float32)
    odba = compute_odba(window)
    assert odba.shape == (200, 2)


def test_compute_odba_wrong_channels_raises():
    with pytest.raises(ValueError, match='12 raw channels'):
        compute_odba(np.zeros((200, 14), dtype=np.float32))


def test_ensure_feature_matrix_expands_12_to_14():
    matrix = ensure_feature_matrix(_make_window(200, 12))
    assert matrix.shape == (200, 14)


def test_ensure_feature_matrix_passes_through_14():
    matrix = ensure_feature_matrix(_make_window(200, 14))
    assert matrix.shape == (200, 14)


def test_ensure_feature_matrix_wrong_channels_raises():
    with pytest.raises(ValueError):
        ensure_feature_matrix(_make_window(200, 5))


def test_transform_for_model_no_tf():
    out = transform_for_model(
        _make_window(200, 14),
        use_time_frequency=False,
        scales=tuple(range(1, 17)),
        w=5.0,
    )
    assert out.shape == (200, 14)
    assert out.dtype == np.float32


def test_transform_for_model_with_cwt():
    out = transform_for_model(
        _make_window(200, 14),
        use_time_frequency=True,
        scales=tuple(range(1, 5)),
        w=5.0,
    )
    assert out.ndim == 3
    assert out.shape[0] == 14
    assert out.dtype == np.float32


# --- predictor unit tests ---

def test_predictor_not_ready_without_checkpoint(tmp_path):
    from app.config import AppConfig
    from app.predictor import BehaviorPredictor

    config = AppConfig.__new__(AppConfig)
    object.__setattr__(config, 'model_path', tmp_path / 'missing.pt')
    object.__setattr__(config, 'metadata_path', tmp_path / 'meta.json')
    object.__setattr__(config, 'window_size', 200)
    object.__setattr__(config, 'stride', 100)
    object.__setattr__(config, 'use_odba', True)
    object.__setattr__(config, 'use_time_frequency', False)
    object.__setattr__(config, 'use_attention', True)
    object.__setattr__(config, 'tf_method', 'cwt')
    object.__setattr__(config, 'cwt_w', 5.0)
    object.__setattr__(config, 'cwt_scales', tuple(range(1, 17)))
    object.__setattr__(config, 'feature_columns', tuple(f'c{i}' for i in range(14)))
    object.__setattr__(config, 'class_names', tuple(f'class_{i}' for i in range(8)))
    object.__setattr__(config, 'device', 'cpu')

    predictor = BehaviorPredictor(config)
    assert predictor.ready is False


def test_predictor_raises_runtime_when_not_ready(tmp_path):
    from app.config import AppConfig
    from app.predictor import BehaviorPredictor

    config = AppConfig.__new__(AppConfig)
    object.__setattr__(config, 'model_path', tmp_path / 'missing.pt')
    object.__setattr__(config, 'metadata_path', tmp_path / 'meta.json')
    object.__setattr__(config, 'window_size', 200)
    object.__setattr__(config, 'stride', 100)
    object.__setattr__(config, 'use_odba', True)
    object.__setattr__(config, 'use_time_frequency', False)
    object.__setattr__(config, 'use_attention', True)
    object.__setattr__(config, 'tf_method', 'cwt')
    object.__setattr__(config, 'cwt_w', 5.0)
    object.__setattr__(config, 'cwt_scales', tuple(range(1, 17)))
    object.__setattr__(config, 'feature_columns', tuple(f'c{i}' for i in range(14)))
    object.__setattr__(config, 'class_names', tuple(f'class_{i}' for i in range(8)))
    object.__setattr__(config, 'device', 'cpu')

    predictor = BehaviorPredictor(config)
    with pytest.raises(RuntimeError, match='checkpoint'):
        predictor.predict(_make_window(), top_k=3)
