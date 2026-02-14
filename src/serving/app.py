from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
import yaml
import pandas as pd
import numpy as np
import time
from src.serving.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)
from src.serving.model_loader import ModelLoader
from src.serving.feature_service import FeatureService
from src.monitoring.metrics_exporter import (
    get_metrics,
    track_prediction,
    track_latency,
    track_error
)
from src.monitoring.performance_monitor import get_monitor
from src.monitoring.alerting import alert_model_failure
from logger.log import logging
import os

config_path = os.getenv("SERVING_CONFIG", "configs/serving_config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

config["api"]["host"] = os.getenv("API_HOST", config["api"]["host"])
config["api"]["port"] = int(os.getenv("API_PORT", config["api"]["port"]))
config["mlflow"]["tracking_uri"] = os.getenv(
    "MLFLOW_TRACKING_URI", 
    config["mlflow"]["tracking_uri"]
)

REPO_PATH = os.getenv("FEAST_REPO_PATH", "feature_store/feature_repo")

app = FastAPI(title="Fraud Detection API", version="2.0.0")

performance_monitor = get_monitor()

feature_service = FeatureService(repo_path=REPO_PATH)

model_loader = ModelLoader(
    tracking_uri=config["mlflow"]["tracking_uri"],
    experiment_name=config["mlflow"]["experiment_name"]
)

@app.on_event("startup")
async def startup():
    print("Starting API...")
    try:
        if model_loader.load_model():
            print("Model loaded successfully")
        else:
            print("WARNING: Model failed to load")
            alert_model_failure("Model failed to load during startup")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        alert_model_failure(f"Exception during model loading: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health():
    feast_status = feature_service.store is not None
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded(),
        feast_connected=feast_status
    )

EXPECTED_COLUMNS = [
    "SK_ID_CURR", "ext_source_mean", "ext_sources_prod", "EXT_SOURCE_3",
    "ext_sources_sum", "EXT_SOURCE_2", "EXT_SOURCE_1", "DAYS_BIRTH",
    "age_years", "years_employed", "goods_price_to_credit_ratio",
    "REGION_RATING_CLIENT_W_CITY", "REGION_RATING_CLIENT",
    "DAYS_LAST_PHONE_CHANGE", "is_male", "DAYS_ID_PUBLISH",
    "REG_CITY_NOT_WORK_CITY", "FLAG_EMP_PHONE", "DAYS_EMPLOYED",
    "REG_CITY_NOT_LIVE_CITY", "FLAG_DOCUMENT_3"
]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Transasction prediction endpoint.
    """
    start_time = time.time()
    success = False

    try:
        logging.info('=== Prediction started ===')
        request_data = request.model_dump(exclude_unset=True)

        feast_features = feature_service.get_online_features(request_data.get("SK_ID_CURR"))
        merged_features = {**feast_features, **request_data}

        amt_credit = request_data.get("AMT_CREDIT", 1.0)
        amt_goods = request_data.get("AMT_GOODS_PRICE", 0.0)
        merged_features["goods_price_to_credit_ratio"] = amt_goods / amt_credit
        features_df = pd.DataFrame([merged_features])

        logging.info('features used for prediction: ', features_df)

        for col in EXPECTED_COLUMNS:
            if col not in features_df.columns:
                features_df[col] = np.nan

        features_df = features_df[EXPECTED_COLUMNS]

        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        features_df = features_df.fillna(0)

        clean_features_dict = features_df.iloc[0].to_dict()

        prediction, probability = model_loader.predict(features_df)
        risk_level = model_loader.get_risk_level(probability)

        track_prediction(is_fraud=(prediction == 1))
        success = True

        return PredictionResponse(
            prediction=int(prediction),
            probability=round(float(probability), 4),
            risk_level=risk_level,
            used_features=clean_features_dict,
            timestamp=datetime.now(timezone.utc)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        track_error()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        total_duration = time.time() - start_time
        track_latency(total_duration)
        performance_monitor.record_request(latency_seconds=total_duration, success=success)

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    return get_metrics()


@app.get("/performance")
async def get_performance():
    """
    Get current performance metrics.
    """
    return performance_monitor.get_stats()


@app.get("/")
async def root():
    return {
        "name": "Loan fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "metrics": "/metrics (Prometheus)",
            "performance": "/performance (JSON stats)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.app:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True
    )