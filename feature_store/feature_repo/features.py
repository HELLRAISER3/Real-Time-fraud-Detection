from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field, ValueType,Project
from feast.types import Float32, Int64

project = Project(name="fraud_detection", description="A project for fraud detection")

customer = Entity(
    name="customerId",
    join_keys=["customerId"],
    value_type=ValueType.STRING,
    description="Customer / Account ID",
)

source = FileSource(
    path="../../data/feature_store/customer_features.parquet",
    event_timestamp_column="event_timestamp",
)

customer_fraud_features = FeatureView(
    name="customer_fraud_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="txn_count_30d", dtype=Int64),
        Field(name="avg_amount_30d", dtype=Float32),
        Field(name="max_amount_30d", dtype=Float32),
        Field(name="card_present_ratio", dtype=Float32),
        Field(name="fraud_rate", dtype=Float32),
    ],
    source=source,
)
