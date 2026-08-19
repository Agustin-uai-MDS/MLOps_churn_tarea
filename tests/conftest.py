"""Fixtures compartidas por todos los tests.

No requieren red ni credenciales: el modelo se carga desde `models/`,
que ya está versionado en el repo (ver training/train.py).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    # el `with` dispara el lifespan de FastAPI (predictor.load()), igual que un server real
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def valid_customer() -> dict:
    return {
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
