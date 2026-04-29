from __future__ import annotations

import json
from dataclasses import dataclass

from .config import AppConfig


@dataclass
class ModelFileStatus:
    checkpoint_exists: bool
    metadata_exists: bool
    checkpoint_path: str
    metadata_path: str
    metadata_valid: bool
    class_names: list[str] | None
    window_size: int | None
    error: str | None

    @property
    def ready(self) -> bool:
        return self.checkpoint_exists and self.metadata_exists and self.metadata_valid


def check_model_status(config: AppConfig | None = None) -> ModelFileStatus:
    if config is None:
        config = AppConfig()

    checkpoint_exists = config.model_path.exists()
    metadata_exists = config.metadata_path.exists()
    metadata_valid = False
    class_names: list[str] | None = None
    window_size: int | None = None
    error: str | None = None

    if not checkpoint_exists:
        error = f'Checkpoint missing: {config.model_path}'

    if metadata_exists:
        try:
            meta = json.loads(config.metadata_path.read_text(encoding='utf-8'))
            metadata_valid = True
            class_names = meta.get('class_names')
            window_size = meta.get('window_size')
        except (json.JSONDecodeError, OSError) as exc:
            metadata_valid = False
            error = f'Metadata invalid: {exc}'
    else:
        if error is None:
            error = f'Metadata missing: {config.metadata_path}'

    return ModelFileStatus(
        checkpoint_exists=checkpoint_exists,
        metadata_exists=metadata_exists,
        checkpoint_path=str(config.model_path),
        metadata_path=str(config.metadata_path),
        metadata_valid=metadata_valid,
        class_names=class_names,
        window_size=window_size,
        error=error,
    )
