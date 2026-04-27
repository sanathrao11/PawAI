from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import AppConfig
from .predictor import BehaviorPredictor
from .schemas import PredictionItem, PredictionResponse, WindowPredictionRequest


app = FastAPI(title='Pet Behavior Monitoring API', version='0.1.0')
predictor: BehaviorPredictor | None = None


@app.on_event('startup')
def load_predictor() -> None:
    global predictor
    predictor = BehaviorPredictor(AppConfig())


@app.get('/health')
def health() -> dict[str, object]:
    return {
        'status': 'ok',
        'model_ready': bool(predictor and predictor.ready),
        'window_size': predictor.window_size if predictor else AppConfig().window_size,
    }


@app.post('/predict', response_model=PredictionResponse)
def predict(request: WindowPredictionRequest) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail='Predictor has not started yet.')
    if not predictor.ready:
        raise HTTPException(
            status_code=503,
            detail='Model checkpoint is missing. Export the notebook checkpoint and set PET_BEHAVIOR_MODEL_PATH.',
        )

    try:
        result = predictor.predict(request.window, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictionResponse(
        predicted_index=result.predicted_index,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        top_k=[PredictionItem(**item) for item in result.top_k],
        window_size=predictor.window_size,
        model_ready=predictor.ready,
    )
