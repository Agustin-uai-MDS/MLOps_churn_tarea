"""Métricas de servicio en memoria: conteo de requests, errores y latencia.

Deliberadamente simple y en memoria (no Prometheus/Redis) para no agregar
infraestructura extra solo para esto en un proyecto de este tamaño. Limitación
conocida: con más de un worker de uvicorn, cada proceso lleva sus propios
contadores por separado (ver README).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class EndpointStats:
    count: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return round(self.total_latency_ms / self.count, 2) if self.count else 0.0


class MetricsTracker:
    def __init__(self) -> None:
        self._stats: dict[str, EndpointStats] = defaultdict(EndpointStats)

    def reset(self) -> None:
        self._stats.clear()

    def record(self, endpoint: str, latency_ms: float, is_error: bool) -> None:
        stats = self._stats[endpoint]
        stats.count += 1
        stats.total_latency_ms += latency_ms
        if is_error:
            stats.errors += 1

    def snapshot(self) -> dict:
        return {
            "requests_total": sum(s.count for s in self._stats.values()),
            "errors_total": sum(s.errors for s in self._stats.values()),
            "by_endpoint": {
                endpoint: {
                    "requests": s.count,
                    "errors": s.errors,
                    "avg_latency_ms": s.avg_latency_ms,
                }
                for endpoint, s in self._stats.items()
            },
        }


# instancia única, igual que predictor.py: un solo lugar acumulando estado
metrics = MetricsTracker()
