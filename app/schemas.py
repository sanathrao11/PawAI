from __future__ import annotations

from typing import List, Optional

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


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[PredictionResponse] = None
    error: Optional[str] = None


class ModelStatusResponse(BaseModel):
    ready: bool
    checkpoint_exists: bool
    metadata_exists: bool
    checkpoint_path: str
    metadata_path: str
    metadata_valid: bool
    class_names: Optional[list[str]] = None
    window_size: Optional[int] = None
    error: Optional[str] = None
