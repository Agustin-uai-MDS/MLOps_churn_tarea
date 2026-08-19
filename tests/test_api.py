"""Tests de contrato de la API: rutas, validación de entrada, errores esperados."""

from __future__ import annotations

from app.metrics import metrics
from app.predictor import predictor


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_schema_exposes_features_and_metrics(client):
    response = client.get("/model/schema")
    assert response.status_code == 200

    body = response.json()
    assert "tenure" in body["features"]
    assert "roc_auc" in body["metrics_test"]


def test_predict_valid_customer_returns_probability(client, valid_customer):
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn"], bool)
    assert body["threshold"] == 0.5
    assert body["out_of_range_features"] == []


def test_predict_flags_out_of_range_numeric_input(client, valid_customer):
    # 500 mensuales y 90 meses de antigüedad superan por lejos el percentil 99
    # visto en entrenamiento (ver models/metadata.json -> training_ranges)
    payload = {**valid_customer, "MonthlyCharges": 500.0, "tenure": 90}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "MonthlyCharges" in body["out_of_range_features"]
    assert "tenure" in body["out_of_range_features"]


def test_predict_rejects_blank_total_charges(client, valid_customer):
    payload = {**valid_customer, "TotalCharges": " "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "TotalCharges" in str(response.json()["detail"])


def test_predict_rejects_invalid_categorical_value(client, valid_customer):
    payload = {**valid_customer, "OnlineSecurity": "yes"}  # el modelo solo conoce "Yes"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_tenure(client, valid_customer):
    payload = {**valid_customer, "tenure": -1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client, valid_customer):
    payload = dict(valid_customer)
    del payload["Contract"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_returns_one_prediction_per_customer(client, valid_customer):
    response = client.post("/predict/batch", json={"customers": [valid_customer, valid_customer]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_predict_batch_rejects_empty_customer_list(client):
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422


def test_predict_returns_503_when_model_not_loaded(client, valid_customer, monkeypatch):
    monkeypatch.setattr(predictor, "model", None)
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 503


def test_metrics_tracks_requests_and_errors(client, valid_customer):
    metrics.reset()

    client.get("/health")
    client.get("/health")
    client.post("/predict", json={**valid_customer, "gender": "invalid"})  # 422

    body = client.get("/metrics").json()
    assert body["requests_total"] >= 3
    assert body["errors_total"] >= 1
    assert body["by_endpoint"]["/health"]["requests"] == 2
    assert body["by_endpoint"]["/health"]["errors"] == 0
