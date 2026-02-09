from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone


class PredictionRequest(BaseModel):
    SK_ID_CURR: Optional[float] = None
    ext_source_mean: Optional[float] = None
    ext_sources_prod: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    ext_sources_sum: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None
    DAYS_BIRTH: Optional[float] = None
    age_years: Optional[float] = None
    years_employed: Optional[float] = None
    goods_price_to_credit_ratio: Optional[float] = None
    REGION_RATING_CLIENT_W_CITY: Optional[float] = None
    REGION_RATING_CLIENT: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    is_male: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    REG_CITY_NOT_WORK_CITY: Optional[float] = None
    FLAG_EMP_PHONE: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    REG_CITY_NOT_LIVE_CITY: Optional[float] = None
    FLAG_DOCUMENT_3: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    timestamp: datetime = Field(default_factory=datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now(timezone.utc))
