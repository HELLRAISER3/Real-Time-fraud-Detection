from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timezone

class PredictionRequest(BaseModel):
    SK_ID_CURR: int = Field(..., description="Unique ID of the applicant")
    
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    
    age_years: Optional[float] = None
    ext_source_mean: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "SK_ID_CURR": 100002,
                "AMT_CREDIT": 202500.0
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    used_features: Dict[str, Any]  
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feast_connected: bool # for feast
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))