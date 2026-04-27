from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
from typing import Any


SENSOR_COLUMNS = [
    'ABack_x', 'ABack_y', 'ABack_z',
    'ANeck_x', 'ANeck_y', 'ANeck_z',
    'GBack_x', 'GBack_y', 'GBack_z',
    'GNeck_x', 'GNeck_y', 'GNeck_z',
]

ENGINEERED_COLUMNS = ['ODBA_ABack', 'ODBA_ANeck']
DEFAULT_FEATURE_COLUMNS = SENSOR_COLUMNS + ENGINEERED_COLUMNS
DEFAULT_CLASS_NAMES = tuple(f'class_{index}' for index in range(8))
DEFAULT_CWT_SCALES = tuple(range(1, 17))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}


def _parse_int_list(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return default
    items = [piece.strip() for piece in value.split(',') if piece.strip()]
    if not items:
        return default
    return tuple(int(piece) for piece in items)


@dataclass(frozen=True)
class AppConfig:
    model_path: Path = field(default_factory=lambda: Path(os.getenv('PET_BEHAVIOR_MODEL_PATH', 'best_model.pt')))
    metadata_path: Path = field(default_factory=lambda: Path(os.getenv('PET_BEHAVIOR_METADATA_PATH', 'model_metadata.json')))
    window_size: int = field(default_factory=lambda: int(os.getenv('PET_BEHAVIOR_WINDOW_SIZE', '200')))
    stride: int = field(default_factory=lambda: int(os.getenv('PET_BEHAVIOR_STRIDE', '100')))
    use_odba: bool = field(default_factory=lambda: _env_bool('PET_BEHAVIOR_USE_ODBA', True))
    use_time_frequency: bool = field(default_factory=lambda: _env_bool('PET_BEHAVIOR_USE_TIME_FREQUENCY', True))
    use_attention: bool = field(default_factory=lambda: _env_bool('PET_BEHAVIOR_USE_ATTENTION', True))
    tf_method: str = field(default_factory=lambda: os.getenv('PET_BEHAVIOR_TF_METHOD', 'cwt'))
    cwt_w: float = field(default_factory=lambda: float(os.getenv('PET_BEHAVIOR_CWT_W', '5.0')))
    cwt_scales: tuple[int, ...] = field(default_factory=lambda: _parse_int_list(os.getenv('PET_BEHAVIOR_CWT_SCALES'), DEFAULT_CWT_SCALES))
    feature_columns: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_FEATURE_COLUMNS))
    class_names: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_CLASS_NAMES))
    device: str = field(default_factory=lambda: os.getenv('PET_BEHAVIOR_DEVICE', 'cpu'))

    @property
    def input_channels(self) -> int:
        return len(self.feature_columns)

    def resolve_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding='utf-8'))
