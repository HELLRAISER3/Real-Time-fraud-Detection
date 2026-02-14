# Real-Time Fraud Detection

ML system for real-time fraud detection in loan applications using XGBoost. Deployed as a containerized FastAPI service with MLflow model tracking, Feast feature store, and Prometheus/Grafana monitoring.

## What It Does

- Predicts fraud probability in loan applications
- Returns risk level (low/medium/high/critical) based on probability thresholds
- Serves predictions via REST API with <500ms latency
- Tracks model performance metrics in real-time

## Tech Stack

**ML & Data:**
- XGBoost for classification
- Scikit-learn for preprocessing
- Pandas/NumPy for data handling
- Optuna for hyperparameter tuning

**Serving:**
- FastAPI for REST API
- Feast + Redis for feature store
- MLflow for model versioning and serving
- Docker + Docker Compose for deployment

**Monitoring:**
- Prometheus for metrics collection
- Grafana for dashboards
- Custom performance tracking

## Project Structure

```
.
├── src/
│   ├── data_pipeline/        # Data processing and feature engineering
│   ├── training/              # Model training scripts
│   ├── tuning/                # Hyperparameter optimization with Optuna
│   ├── serving/               # FastAPI application
│   └── monitoring/            # Prometheus metrics and alerting
├── feature_store/             # Feast feature definitions
├── configs/                   # Configuration files (YAML)
├── data/                      # Training and processed data
├── tests/                     # Integration tests
├── logger/                    # Custom logging setup
├── docker-compose.yml         # Multi-container orchestration
├── Dockerfile.serving         # API container
└── Dockerfile.training        # Training container
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)

### Run Everything with Docker

```bash
# Start all services (API, MLflow, Redis, Prometheus, Grafana)
docker compose up -d

# Check if everything is running
docker compose ps

# View logs
docker compose logs -f api
```

Services will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SK_ID_CURR": 100002,
    "AMT_CREDIT": 406597.5,
    "AMT_GOODS_PRICE": 351000.0,
    "AMT_ANNUITY": 24700.5,
    "DAYS_BIRTH": -12005,
    "DAYS_EMPLOYED": -4542
  }'
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.08,
  "risk_level": "low",
  "message": "Transaction processed successfully"
}
```

## Local Development

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-base.txt
pip install -r requirements-training.txt
pip install -r requirements-serving.txt
```

### Train Model Locally

```bash
# Prepare feature store
cd feature_store
python build_features.py
cd ..

# Train model
python -m src.training.train
```

### Run API Locally

```bash
# Start Redis (required for feature store)
docker run -d -p 6379:6379 redis:7-alpine

# Start API
uvicorn src.serving.app:app --reload --port 8000
```

## Configuration

All config files are in `configs/`:

- `training_config.yaml` - Model parameters, data paths, MLflow settings
- `serving_config.yaml` - API settings, feature store config (local)
- `serving_config.docker.yaml` - API settings for Docker deployment
- `optuna_config.yaml` - Hyperparameter search space
- `prometheus.yml` - Metrics scraping configuration

## Monitoring

The system tracks:
- Total predictions (fraud vs ok)
- API latency (average response time)
- Error rate

**View metrics:**
- Prometheus: http://localhost:9090
- Grafana dashboard: http://localhost:3000
- Performance endpoint: http://localhost:8000/performance

## Model Training

### Standard Training

```bash
python -m src.training.train
```

Trains XGBoost model with settings from `configs/training_config.yaml`. Model and metrics are logged to MLflow.

### Hyperparameter Tuning

```bash
python -m src.tuning.tune
```

Runs Optuna trials to find optimal hyperparameters. Search space defined in `configs/optuna_config.yaml`.

## Testing

```bash
# Run all tests
pytest tests/

# Integration tests only
pytest tests/integration/
```

## API Endpoints

### `POST /predict`

Make a fraud prediction.

**Request body:**
```json
{
  "SK_ID_CURR": 100002,
  "AMT_CREDIT": 406597.5,
  "AMT_GOODS_PRICE": 351000.0,
  "AMT_ANNUITY": 24700.5,
  "DAYS_BIRTH": -12005,
  "DAYS_EMPLOYED": -4542
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.08,
  "risk_level": "low",
  "message": "Transaction processed successfully"
}
```

### `GET /health`

Health check endpoint.

### `GET /metrics`

Prometheus metrics in text format.

### `GET /performance`

Current performance stats (JSON).

## Risk Levels

| Probability | Risk Level |
|-------------|------------|
| < 0.3       | low        |
| 0.3 - 0.6   | medium     |
| 0.6 - 0.8   | high       |
| > 0.8       | critical   |


## Performance

- **Latency**: ~150ms average (p95: ~300ms)
- **Throughput**: Handles concurrent requests
- **Model size**: ~2MB (XGBoost serialized)
- **Memory usage**: ~500MB per container

## License

MIT
