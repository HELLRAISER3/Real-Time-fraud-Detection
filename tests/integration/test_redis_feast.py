from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_store/feature_repo")

entity_rows = [{"SK_ID_CURR": 100002}]
all_features = [
    "applicant_risk_features:age_years",
    "applicant_risk_features:ext_source_mean",
    "applicant_risk_features:credit_to_income_ratio"
]

feature_vector = store.get_online_features(
    features=all_features,
    entity_rows=entity_rows
).to_dict()

print("--- Data Retrieved from Redis ---")
for feature, value in feature_vector.items():
    print(f"{feature}: {value[0]}")