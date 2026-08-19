"""Configuración de la app, leída desde variables de entorno.

Todo lo que puede cambiar según el ambiente (dónde vive el modelo, con qué
umbral se decide si un cliente "va a churnear") vive acá — no repartido
como constantes sueltas en main.py o predictor.py.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # carpeta donde predictor.py busca model.joblib, preprocessor.joblib y metadata.json.
    # absoluta a propósito: si fuera Path("models") relativo, dependería de desde qué
    # directorio se invoque uvicorn/pytest (mismo patrón que REPO_ROOT en training/train.py)
    model_dir: Path = REPO_ROOT / "models"

    # probabilidad mínima para clasificar a un cliente como "en riesgo de churn".
    # el modelo se entrenó con class_weight="balanced" (ver models/metadata.json),
    # así que 0.5 ya favorece detectar churners por sobre evitar falsos positivos.
    churn_threshold: float = 0.5


settings = Settings()
