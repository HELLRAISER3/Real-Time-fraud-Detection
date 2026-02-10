from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field, Project, ValueType
from feast.types import Float32, Int64, String

project = Project(name="loan_risk_prediction", description="Credit Risk Prediction")

applicant = Entity(
    name="applicant",
    join_keys=["SK_ID_CURR"],
    value_type=ValueType.INT64,
    description="ID for loan applicant",
)

source = FileSource(
    name="applicant_features_source",
    path="../../data/feature_store/customer_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

applicant_risk_features = FeatureView(
    name="applicant_risk_features",
    entities=[applicant],
    ttl=timedelta(days=3650), 
    schema=[
        Field(name="credit_to_income_ratio", dtype=Float32),
        Field(name="annuity_to_income_ratio", dtype=Float32),
        Field(name="credit_term_approx", dtype=Float32),
        Field(name="goods_price_to_credit_ratio", dtype=Float32),
        
        Field(name="age_years", dtype=Float32),
        Field(name="years_employed", dtype=Float32),
        Field(name="employed_to_age_ratio", dtype=Float32),
        Field(name="income_per_person", dtype=Float32),
        
        Field(name="ext_source_1", dtype=Float32),
        Field(name="ext_source_2", dtype=Float32),
        Field(name="ext_source_3", dtype=Float32),
        Field(name="ext_source_mean", dtype=Float32),
        
        Field(name="flag_own_car", dtype=Int64),
        Field(name="flag_own_realty", dtype=Int64),
        
    ],
    source=source,
    tags={"team": "risk_scoring", "model_type": "credit_default"},
)