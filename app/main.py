"""FastAPI: rutas y ciclo de vida de la app.

No importa sklearn ni sabe cómo se arma un DataFrame — eso vive en
`predictor.py`. Acá solo se validan requests (vía `schemas.py`), se llama
al predictor, y se traducen resultados/errores a respuestas HTTP.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.config import settings
from app.metrics import metrics
from app.predictor import ModelNotLoadedError, predictor
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnPrediction,
    CustomerFeatures,
    HealthResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # se carga UNA vez al arrancar, no en cada request
    try:
        predictor.load()
    except ModelNotLoadedError as exc:
        print(f"[startup] No se pudo cargar el modelo: {exc}")
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predice el riesgo de deserción (churn) de un cliente de telecomunicaciones.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model_loaded() -> None:
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="El modelo no está cargado todavía.")


@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    metrics.record(endpoint=request.url.path, latency_ms=latency_ms, is_error=response.status_code >= 400)
    return response


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics.snapshot()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_type=predictor.model_type,
    )


@app.get("/model/schema")
def model_schema() -> dict:
    _require_model_loaded()
    return {
        "features": CustomerFeatures.model_json_schema()["properties"],
        "target": predictor.metadata["target"],
        "model_type": predictor.metadata["model_type"],
        "trained_at": predictor.metadata["trained_at"],
        "metrics_test": predictor.metadata["metrics_test"],
        "default_threshold": settings.churn_threshold,
        "training_ranges": predictor.training_ranges,
    }


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures) -> ChurnPrediction:
    _require_model_loaded()
    return predictor.predict(customer, threshold=settings.churn_threshold)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    _require_model_loaded()
    predictions = predictor.predict_batch(request.customers, threshold=settings.churn_threshold)
    return BatchPredictionResponse(predictions=predictions)
