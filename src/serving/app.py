from fastapi import FastAPI, HTTPException
from datetime import datetime
import yaml

from src.serving.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)
from src.serving.model_loader import ModelLoader

with open("configs/serving_config.yaml") as f:
    config = yaml.safe_load(f)


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0"
)

model_loader = ModelLoader(
    tracking_uri=config["mlflow"]["tracking_uri"],
    experiment_name=config["mlflow"]["experiment_name"]
)


@app.on_event("startup")
async def startup():
    print("Starting API...")
    if model_loader.load_model():
        print("Model loaded successfully")
    else:
        print("Model failed to load")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        request_dict = request.model_dump()
        features = model_loader.prepare_features(request_dict)
        prediction, probability = model_loader.predict(features)
        risk_level = model_loader.get_risk_level(probability)

        return PredictionResponse(
            prediction=int(prediction),
            probability=round(float(probability), 4),
            risk_level=risk_level,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
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