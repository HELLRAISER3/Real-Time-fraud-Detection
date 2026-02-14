from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from fastapi import Response

registry = CollectorRegistry()

predictions_total = Counter(
    name='predictions_total',
    documentation='Total predictions',
    labelnames=['result'],  # "fraud" or "ok"
    registry=registry
)

api_latency = Histogram(
    name='api_latency_seconds',
    documentation='API response time',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry
)

errors_total = Counter(
    name='errors_total',
    documentation='Total errors',
    registry=registry
)

def track_prediction(is_fraud: bool):
    result = "fraud" if is_fraud else "ok"
    predictions_total.labels(result=result).inc()


def track_latency(seconds: float):
    api_latency.observe(seconds)


def track_error():
    errors_total.inc()


def get_metrics() -> Response:
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
