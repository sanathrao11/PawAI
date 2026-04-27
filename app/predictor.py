from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import AppConfig, DEFAULT_CLASS_NAMES
from .model import build_model, load_model_weights
from .preprocessing import transform_for_model


@dataclass
class PredictionResult:
    predicted_index: int
    predicted_class: str
    confidence: float
    top_k: list[dict[str, Any]]


class BehaviorPredictor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.metadata = self._load_metadata()
        self.class_names = tuple(self.metadata.get('class_names', list(DEFAULT_CLASS_NAMES)))
        self.use_time_frequency = bool(self.metadata.get('use_time_frequency', config.use_time_frequency))
        self.use_attention = bool(self.metadata.get('use_attention', config.use_attention))
        self.cwt_scales = tuple(self.metadata.get('cwt_scales', list(config.cwt_scales)))
        self.cwt_w = float(self.metadata.get('cwt_w', config.cwt_w))
        self.feature_columns = tuple(self.metadata.get('feature_columns', list(config.feature_columns)))
        self.window_size = int(self.metadata.get('window_size', config.window_size))
        self.model = build_model(
            n_channels=len(self.feature_columns),
            n_classes=len(self.class_names),
            use_time_frequency=self.use_time_frequency,
            use_attention=self.use_attention,
        ).to(self.device)
        self.ready = False
        if config.model_path.exists():
            load_model_weights(self.model, config.model_path, self.device)
            self.model.eval()
            self.ready = True

    def _load_metadata(self) -> dict[str, Any]:
        if self.config.metadata_path.exists():
            return json.loads(self.config.metadata_path.read_text(encoding='utf-8'))
        return {}

    def predict(self, window: list[list[float]], top_k: int = 3) -> PredictionResult:
        if not self.ready:
            raise RuntimeError(
                f'Model checkpoint not found at {self.config.model_path}. '
                'Place the exported notebook checkpoint there or set PET_BEHAVIOR_MODEL_PATH.'
            )

        transformed = transform_for_model(
            window,
            use_time_frequency=self.use_time_frequency,
            scales=self.cwt_scales,
            w=self.cwt_w,
        )
        tensor = torch.tensor(transformed, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        ranking = np.argsort(probabilities)[::-1][:top_k]
        top_results = [
            {
                'class_index': int(index),
                'class_name': self.class_names[index] if index < len(self.class_names) else str(index),
                'probability': float(probabilities[index]),
            }
            for index in ranking
        ]

        best = top_results[0]
        return PredictionResult(
            predicted_index=best['class_index'],
            predicted_class=best['class_name'],
            confidence=best['probability'],
            top_k=top_results,
        )
