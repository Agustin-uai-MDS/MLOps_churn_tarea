"""Recalcula métricas sobre datos no vistos a partir de los artefactos ya entrenados.

No entrena nada: carga `model.joblib` y `preprocessor.joblib` desde `models/`
y verifica el desempeño reportado por `train.py`, sin volver a hacer fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from training.train import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    FEATURES,
    SEED,
    TARGET,
    compute_metrics,
)


def evaluate(data_path: Path, model_dir: Path, seed: int = SEED) -> dict:

    model = joblib.load(model_dir / "model.joblib")
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")

    df = pd.read_csv(data_path)

    X = df[FEATURES]
    y = df[TARGET]

    # misma partición y misma semilla que train.py: aísla el mismo test set,
    # que nunca participó del fit del preprocesador ni del modelo.

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    X_test_t = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    return compute_metrics(y_test, y_pred, y_proba)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa el modelo de churn ya entrenado.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.data_path, args.model_dir, args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
