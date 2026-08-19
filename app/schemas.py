"""Contratos Pydantic de entrada y salida de la API.

Los nombres de campo de `CustomerFeatures` coinciden a propósito con las
columnas crudas del dataset (mismas que espera `training/features.py`), para
que `predictor.py` pueda armar el DataFrame sin tener que traducir nombres.

Cada campo categórico usa `Literal` con los valores exactos que el modelo
conoce: un valor fuera de esa lista se rechaza con 422 antes de llegar al
modelo, en vez de calcularse mal en silencio.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]

YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=100, description="Meses como cliente")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., gt=0, description="Cobro mensual en USD")
    TotalCharges: float = Field(..., ge=0, description="Cobro acumulado en USD")


class ChurnPrediction(BaseModel):
    churn: bool = Field(..., description="True si la probabilidad supera el umbral configurado")
    churn_probability: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0, le=1, description="Umbral usado para decidir `churn`")
    out_of_range_features: list[str] = Field(
        default_factory=list,
        description="Features numéricas fuera del rango (percentil 1-99) visto en entrenamiento. "
        "No bloquea la predicción, es una señal de que el input es inusual.",
    )


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[ChurnPrediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_type: str | None = None
