"""Carga los artefactos entrenados y calcula predicciones de churn.

Es la única parte de `app/` que sabe que existen `model.joblib`,
`preprocessor.joblib` y `training/features.py`. `main.py` no debería
importar sklearn ni saber cómo se arma el DataFrame de features — solo le
pasa un `CustomerFeatures` y recibe un `ChurnPrediction`.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from app.config import settings
from app.schemas import ChurnPrediction, CustomerFeatures
from training.features import engineer_features


class ModelNotLoadedError(RuntimeError):
    """El modelo no pudo cargarse desde `model_dir`, o no se llamó a `load()`."""


class ChurnPredictor:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.metadata: dict | None = None
        self.feature_order: list[str] = []
        self.training_ranges: dict[str, dict[str, float]] = {}

    def load(self) -> None:
        """Carga los artefactos desde disco. Se llama una vez, al arrancar la app."""
        try:
            self.model = joblib.load(self.model_dir / "model.joblib")
            self.preprocessor = joblib.load(self.model_dir / "preprocessor.joblib")
            self.metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ModelNotLoadedError(f"No se encontraron artefactos en {self.model_dir}") from exc

        features = self.metadata["features"]
        self.feature_order = (
            features["numeric_scaled"] + features["numeric_passthrough"] + features["categorical"]
        )
        self.training_ranges = self.metadata.get("training_ranges", {})

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    @property
    def model_type(self) -> str | None:
        return self.metadata.get("model_type") if self.metadata else None

    def _to_frame(self, customers: list[CustomerFeatures]) -> pd.DataFrame:
        raw = pd.DataFrame([c.model_dump() for c in customers])
        engineered = engineer_features(raw)
        return engineered[self.feature_order]

    def _out_of_range_features(self, row: pd.Series) -> list[str]:
        """Columnas numéricas cuyo valor cae fuera del percentil 1-99 visto en
        entrenamiento (ver `training.train.compute_training_ranges`). No bloquea
        la predicción, solo la marca como potencialmente menos confiable."""
        flagged = []
        for col, bounds in self.training_ranges.items():
            value = row[col]
            if value < bounds["min"] or value > bounds["max"]:
                flagged.append(col)
        return flagged

    def predict(self, customer: CustomerFeatures, threshold: float) -> ChurnPrediction:
        return self.predict_batch([customer], threshold)[0]

    def predict_batch(self, customers: list[CustomerFeatures], threshold: float) -> list[ChurnPrediction]:
        if not self.is_loaded:
            raise ModelNotLoadedError("El modelo no está cargado: llama a load() al arrancar la app.")

        X = self._to_frame(customers)
        X_t = self.preprocessor.transform(X)
        probabilities = self.model.predict_proba(X_t)[:, 1]
        out_of_range = [self._out_of_range_features(row) for _, row in X.iterrows()]

        return [
            ChurnPrediction(
                churn=bool(p >= threshold),
                churn_probability=round(float(p), 4),
                threshold=threshold,
                out_of_range_features=oor,
            )
            for p, oor in zip(probabilities, out_of_range, strict=True)
        ]


# instancia única que main.py importa y carga en el lifespan de la app
predictor = ChurnPredictor(settings.model_dir)
