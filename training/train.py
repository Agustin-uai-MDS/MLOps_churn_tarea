"""Entrenamiento reproducible del clasificador de churn.

Separa explícitamente entrenamiento de inferencia: este script produce tres
artefactos versionados en `models/` (modelo, preprocesador y metadatos) que
`app/predictor.py` consume sin depender de este módulo.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "Churn"

# columnas continuas: se estandarizan

NUMERIC_SCALED = ["tenure", "MonthlyCharges", "TotalCharges", "num_add_on_services", "avg_monthly_spend"]

# flags ya binarias (0/1): no necesitan escalado

NUMERIC_PASSTHROUGH = ["SeniorCitizen", "has_internet", "is_month_to_month"]

# variables de texto: se codifican con one-hot

CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]

FEATURES = NUMERIC_SCALED + NUMERIC_PASSTHROUGH + CATEGORICAL

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "churn_clean.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models"


def build_model(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Modelo desconocido: {name!r} (opciones: 'logreg', 'rf')")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num_scaled", StandardScaler(), NUMERIC_SCALED),
            ("num_passthrough", "passthrough", NUMERIC_PASSTHROUGH),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def compute_training_ranges(X_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Percentiles 1 y 99 de cada feature numérica del train set.

    No usa min/max real porque un solo outlier extremo dejaría el rango
    inútil para detectar drift. Se calcula solo sobre X_train (nunca sobre
    test) para no filtrar información del set de evaluación al artefacto.
    """
    ranges = {}
    for col in NUMERIC_SCALED:
        low, high = X_train[col].quantile([0.01, 0.99])
        ranges[col] = {"min": round(float(low), 2), "max": round(float(high), 2)}
    return ranges


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def train(data_path: Path, output_dir: Path, seed: int = SEED, model_name: str = "logreg") -> dict:

    df = pd.read_csv(data_path)

    missing = set(FEATURES + [TARGET]) - set(df.columns)

    if missing:
        raise ValueError(f"Faltan columnas en {data_path}: {sorted(missing)}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    preprocessor = build_preprocessor()

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = build_model(model_name, seed)
    model.fit(X_train_t, y_train)

    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    metadata = {
        "target": TARGET,
        "features": {
            "numeric_scaled": NUMERIC_SCALED,
            "numeric_passthrough": NUMERIC_PASSTHROUGH,
            "categorical": CATEGORICAL,
        },
        "model_type": type(model).__name__,
        "seed": seed,
        "trained_at": datetime.now(UTC).isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics_test": metrics,
        "sklearn_version": sklearn.__version__,
        "training_ranges": compute_training_ranges(X_train),
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo de churn.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = train(args.data_path, args.output_dir, args.seed, args.model)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
