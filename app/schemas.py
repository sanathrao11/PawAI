from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class WindowPredictionRequest(BaseModel):
    window: List[List[float]] = Field(..., description='A single time window of sensor rows.')
    top_k: int = Field(3, ge=1, le=10)


class PredictionItem(BaseModel):
    class_index: int
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_index: int
    predicted_class: str
    confidence: float
    top_k: list[PredictionItem]
    window_size: int
    model_ready: bool
