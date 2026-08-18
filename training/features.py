"""Feature engineering fila a fila, para aplicar sobre datos nuevos de un cliente.

Replica sobre una fila cruda lo mismo que hace la sección de feature
engineering de `notebooks/etl.ipynb` sobre el dataset completo. Lo usa
`app/predictor.py` para transformar cada request entrante antes de pasarlo
por `preprocessor.joblib`.
"""

from __future__ import annotations

import pandas as pd

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def tenure_bucket(months: int) -> str:
    if months <= 12:
        return "0-1y"
    if months <= 24:
        return "1-2y"
    if months <= 48:
        return "2-4y"
    return "4y+"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega al DataFrame las columnas derivadas que espera el preprocesador.

    Recibe las columnas "crudas" de un cliente (las mismas del dataset
    original, sin `customerID` ni `Churn`) y devuelve una copia con los
    features nuevos ya calculados, lista para `preprocessor.transform(...)`.
    """
    out = df.copy()

    out["num_add_on_services"] = (out[SERVICE_COLS] == "Yes").sum(axis=1)
    out["has_internet"] = (out["InternetService"] != "No").astype(int)
    out["tenure_group"] = out["tenure"].apply(tenure_bucket)
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["avg_monthly_spend"] = out["TotalCharges"] / out["tenure"].replace(0, 1)

    return out
