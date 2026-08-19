# Predicción de churn: servicio MLOps end-to-end

Servicio que predice si un cliente de una compañía de telecomunicaciones va a darse de baja (*churn*), a partir de sus datos de cuenta y servicios contratados. Expuesto como API con FastAPI, empaquetado en Docker, con entrenamiento reproducible y verificación automática en cada `push`.

## Problema y datos

- **Problema**: clasificación binaria. Dado un cliente, predecir si va a hacer churn, entregando probabilidad y un umbral configurable (no solo la etiqueta).
- **Datos**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, dataset de ejemplo de IBM), con 7043 clientes y 21 columnas (demografía, contrato, servicios contratados, cobros). Dataset ficticio, sin información personal real.
- **ETL**: `notebooks/etl.ipynb` documenta el análisis (nulos, outliers, variables categóricas) y el diseño del feature engineering. **Es solo para exploración**: la implementación real que usa el servicio en producción vive en `training/features.py` (ver [Limitaciones](#limitaciones-conocidas)).

## Modelo

`RandomForestClassifier` (scikit-learn), entrenado con `training/train.py`, con semilla fija (`42`) y split 80/20 estratificado. Métricas sobre el test set (1409 clientes nunca vistos durante el entrenamiento):

| Métrica | Valor |
|---|---|
| Accuracy | 0.7658 |
| Precision | 0.5412 |
| Recall | 0.7727 |
| F1 | 0.6366 |
| ROC-AUC | 0.8428 |

**Interpretación**: el target está desbalanceado (26.5% churn), así que un modelo que no aprendiera nada y siempre predijera "no churn" ya sacaría 73.5% de accuracy. Por eso el **ROC-AUC (0.84)** es la métrica que realmente importa acá, no la accuracy. Ese valor está en línea con el techo reportado por la comunidad para este dataset específico con estas mismas columnas (ver limitaciones).

## Arquitectura del repositorio

```
training/   entrenamiento reproducible (train.py), evaluación independiente (evaluate.py)
            y feature engineering compartido con la API (features.py)
app/        servicio FastAPI: config.py, schemas.py (contratos Pydantic),
            predictor.py (carga de artefactos + inferencia), main.py (rutas)
models/     artefactos versionados: model.joblib, preprocessor.joblib, metadata.json
tests/      18 tests pytest (contrato de API, validación, casos borde)
.github/    workflow de CI/CD: lint → test → build → smoke → publish (GHCR, solo en tags)
```

## Cómo levantar el servicio

Requisitos: Docker y Docker Compose.

```bash
git clone <url-de-este-repositorio>
cd MLOps_churn_tarea
docker compose up --build
```

Con eso el servicio queda respondiendo en `http://localhost:8000`, sin pasos manuales adicionales. Documentación interactiva (Swagger) en **http://localhost:8000/docs**.

## Cómo probarlo

```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","model_loaded":true,"model_type":"RandomForestClassifier"}
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85
  }'
```

En Windows (PowerShell):

```powershell
$body = @{
    gender = "Female"; SeniorCitizen = 0; Partner = "Yes"; Dependents = "No"
    tenure = 1; PhoneService = "No"; MultipleLines = "No phone service"
    InternetService = "DSL"; OnlineSecurity = "No"; OnlineBackup = "Yes"
    DeviceProtection = "No"; TechSupport = "No"; StreamingTV = "No"; StreamingMovies = "No"
    Contract = "Month-to-month"; PaperlessBilling = "Yes"; PaymentMethod = "Electronic check"
    MonthlyCharges = 29.85; TotalCharges = 29.85
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

```json
{"churn":true,"churn_probability":0.7234,"threshold":0.5,"out_of_range_features":[]}
```

Ambas respuestas son reales, capturadas corriendo el contenedor en local. `POST /predict/batch` acepta el mismo objeto dentro de una lista `{"customers": [...]}`, y `GET /model/schema` expone las features esperadas, las métricas y los rangos de entrenamiento del modelo cargado.

`out_of_range_features` avisa, sin bloquear la predicción, si alguna feature numérica del request cae fuera del percentil 1 a 99 visto en entrenamiento (ver `models/metadata.json` → `training_ranges`). Es una señal simple de que el input es inusual y la predicción podría ser menos confiable. Por ejemplo, un cliente con `MonthlyCharges: 500` (muy por sobre el máximo de ~115 visto en entrenamiento) responde:

```json
{"churn":false,"churn_probability":0.4722,"threshold":0.5,"out_of_range_features":["tenure","MonthlyCharges","avg_monthly_spend"]}
```

`GET /metrics` expone conteo de requests, errores y latencia promedio por endpoint:

```bash
curl http://localhost:8000/metrics
```
```json
{"requests_total":2,"errors_total":0,"by_endpoint":{"/health":{"requests":1,"errors":0,"avg_latency_ms":0.9},"/predict":{"requests":1,"errors":0,"avg_latency_ms":46.15}}}
```

## Variables de entorno

Ver `.env.example`. Ninguna es secreta.

| Variable | Default | Qué hace |
|---|---|---|
| `MODEL_DIR` | `models` | Carpeta donde la API busca los artefactos del modelo |
| `CHURN_THRESHOLD` | `0.5` | Probabilidad mínima para marcar `churn: true` |

## Reentrenar el modelo

```bash
pip install -r requirements-dev.txt
python -m training.train              # guarda nuevos artefactos en models/
python -m training.evaluate            # verifica las métricas de forma independiente
```

Nota: si reentrenas, hay que reconstruir la imagen (`docker compose up --build`) para que el contenedor use los artefactos nuevos. `COPY models/` en el `Dockerfile` los deja congelados en el momento del build, no se actualizan solos.

## Tests y calidad de código

```bash
pytest -v            # 16 tests, sin red ni credenciales
ruff check .          # linter
ruff format --check . # formateador
```

## CI/CD

`.github/workflows/ci.yml` corre en cada `push`/`pull_request` a `main`: `lint → test → build → smoke`, cada uno condicionado al anterior. `smoke` levanta el contenedor de verdad y le hace `curl` a `/health` y `/predict`; no es un test contra el código fuente, es contra la imagen final. Al crear un tag `v*`, se agrega un quinto job (`publish`) que sube esa misma imagen ya probada a GHCR.

## Limitaciones conocidas

- **Techo del dataset**: con las 21 columnas disponibles (sin historial de reclamos, ofertas de competencia, ni satisfacción del cliente), un ROC-AUC entre 0.84 y 0.86 es el rango reportado consistentemente por la comunidad para este dataset. No es una limitación del pipeline, es información que el dataset no contiene.
- **Sin registro de modelos** (tipo MLflow): los artefactos se versionan directamente en Git, no en un model registry.
- **Métricas en memoria, no distribuidas**: `GET /metrics` acumula requests/errores/latencia en el proceso de un solo worker de uvicorn. Con más de un worker o más de un contenedor corriendo en paralelo, cada uno lleva su propio conteo por separado, no hay un total agregado real. Suficiente para esta entrega, no para un despliegue con réplicas.
- **Imagen Docker de ~760MB**: esperable para un stack con scikit-learn + pandas + numpy + scipy; ya se evita instalar dependencias de entrenamiento extra o usar la imagen base completa de Python.

## Qué haríamos con más tiempo

Registro de modelos con MLflow en vez de artefactos versionados directamente en Git, y probar `HistGradientBoostingClassifier` para exprimir un poco más de ROC-AUC.

## Uso de asistentes de IA

Se usó Claude (Anthropic, vía Claude Code) como asistente durante todo el desarrollo: diseño del feature engineering y del ETL, arquitectura de la API (separación config/schemas/predictor/main), configuración de Docker, diseño y depuración del pipeline de CI/CD (incluyendo pruebas locales con `act` antes de cada push), y redacción de este README. Cada decisión de diseño fue discutida y revisada antes de aplicarse; el equipo puede explicar y defender cualquier parte del código.
