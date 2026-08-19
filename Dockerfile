# imagen base "slim": Debian mínimo con Python, sin las herramientas de
# compilación/documentación que trae la imagen completa (~120MB vs ~1GB)
FROM python:3.12-slim

WORKDIR /app

# 1) dependencias primero, código después: Docker cachea cada instrucción
# por separado. Si solo cambias código (no requirements.txt), esta capa
# se reusa y el build siguiente es mucho más rápido.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) recién ahora el código: solo lo que la API necesita para servir,
# nunca notebooks/ ni data/ (no hacen falta para responder un /predict)
COPY app/ ./app/
COPY training/ ./training/
COPY models/ ./models/

# 3) usuario no-root: si alguien logra ejecutar código arbitrario dentro
# del contenedor, que no tenga privilegios de root sobre él
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# 4) healthcheck: Docker (y luego Cloud Run) usan esto para saber si el
# contenedor está realmente sirviendo, no solo "encendido"
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
