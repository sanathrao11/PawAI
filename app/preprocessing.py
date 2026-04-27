from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal as scipy_signal

from .config import ENGINEERED_COLUMNS, SENSOR_COLUMNS


def _cwt_compat(sig_1d: np.ndarray, scales: Sequence[int], w: float = 5.0) -> np.ndarray:
    if hasattr(scipy_signal, 'cwt') and hasattr(scipy_signal, 'morlet2'):
        return scipy_signal.cwt(sig_1d, scipy_signal.morlet2, scales, w=w)

    scales = np.asarray(scales, dtype=float)
    sig_1d = np.asarray(sig_1d, dtype=np.float32)
    out = np.empty((len(scales), sig_1d.shape[0]), dtype=np.complex64)

    for index, scale in enumerate(scales):
        half_width = int(max(8, np.ceil(5 * float(scale))))
        t = np.arange(-half_width, half_width + 1, dtype=np.float32)
        x = t / float(scale)
        wavelet = np.exp(1j * float(w) * x) * np.exp(-0.5 * x ** 2)
        wavelet = wavelet - wavelet.mean()
        norm = np.sqrt(np.sum(np.abs(wavelet) ** 2))
        if norm > 0:
            wavelet = wavelet / norm
        conv = np.convolve(sig_1d, np.conj(wavelet[::-1]), mode='same')
        out[index] = conv.astype(np.complex64)

    return out


def _cwt_window(window_2d: np.ndarray, scales: Sequence[int], w: float = 5.0) -> np.ndarray:
    channels = []
    for channel_index in range(window_2d.shape[1]):
        coeff = _cwt_compat(window_2d[:, channel_index], scales=scales, w=w)
        channels.append(np.abs(coeff))
    return np.stack(channels, axis=0).astype(np.float32)


def compute_odba(window_2d: np.ndarray) -> np.ndarray:
    if window_2d.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(f'Expected {len(SENSOR_COLUMNS)} raw channels, got {window_2d.shape[1]}')

    odba_columns = []
    for axis_start in (0, 3):
        triad = window_2d[:, axis_start:axis_start + 3]
        static = np.mean(triad, axis=0, keepdims=True)
        odba_columns.append(np.sum(np.abs(triad - static), axis=1, keepdims=True))
    return np.concatenate(odba_columns, axis=1).astype(np.float32)


def ensure_feature_matrix(window: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(window, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError('window must be a 2D matrix shaped (timesteps, channels)')
    if matrix.shape[1] == len(SENSOR_COLUMNS):
        odba = compute_odba(matrix)
        return np.concatenate([matrix, odba], axis=1).astype(np.float32)
    if matrix.shape[1] == len(SENSOR_COLUMNS) + len(ENGINEERED_COLUMNS):
        return matrix.astype(np.float32)
    raise ValueError(f'Expected 12 or 14 channels, got {matrix.shape[1]}')


def transform_for_model(window: Sequence[Sequence[float]], *, use_time_frequency: bool, scales: Sequence[int], w: float) -> np.ndarray:
    feature_matrix = ensure_feature_matrix(window)
    if not use_time_frequency:
        return feature_matrix.astype(np.float32)
    return _cwt_window(feature_matrix, scales=scales, w=w).astype(np.float32)
