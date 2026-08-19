"""Tests de training/features.py: feature engineering fila a fila."""

from __future__ import annotations

import pandas as pd

from training.features import engineer_features, tenure_bucket


def test_tenure_bucket_edges():
    assert tenure_bucket(0) == "0-1y"
    assert tenure_bucket(12) == "0-1y"
    assert tenure_bucket(13) == "1-2y"
    assert tenure_bucket(24) == "1-2y"
    assert tenure_bucket(25) == "2-4y"
    assert tenure_bucket(48) == "2-4y"
    assert tenure_bucket(49) == "4y+"


def test_engineer_features_counts_addon_services_and_flags():
    row = pd.DataFrame(
        [
            {
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "InternetService": "DSL",
                "Contract": "Month-to-month",
                "tenure": 5,
                "TotalCharges": 100.0,
            }
        ]
    )
    out = engineer_features(row)

    assert out.loc[0, "num_add_on_services"] == 2
    assert out.loc[0, "has_internet"] == 1
    assert out.loc[0, "is_month_to_month"] == 1


def test_engineer_features_avoids_division_by_zero_on_new_customer():
    # cliente recién llegado: tenure=0, TotalCharges=0 (mismo caso que las
    # 11 filas problemáticas del dataset original, ver notebooks/etl.ipynb)
    row = pd.DataFrame(
        [
            {
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "InternetService": "No",
                "Contract": "Two year",
                "tenure": 0,
                "TotalCharges": 0.0,
            }
        ]
    )
    out = engineer_features(row)

    assert out.loc[0, "avg_monthly_spend"] == 0.0
    assert out.loc[0, "has_internet"] == 0
