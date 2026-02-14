from feast import FeatureStore

store = FeatureStore(repo_path="feature_store/feature_repo")

entity_id = 100002

print(f"--- Checking Online Store for SK_ID_CURR: {entity_id} ---")

try:
    feature_vector = store.get_online_features(
        features=[
            "applicant_risk_features:credit_to_income_ratio",
            "applicant_risk_features:annuity_to_income_ratio",
            "applicant_risk_features:age_years"
        ],
        entity_rows=[{"SK_ID_CURR": entity_id}]
    ).to_dict()

    print("Result:")
    print(feature_vector)
    
    if feature_vector['applicant_risk_features:age_years'][0] is None:
        print("Key found, but data is None. (Check your parquet file schema vs feature_repo types)")
    else:
        print("Data successfully retrieved from Redis!")

except Exception as e:
    print(f"Connection failed: {e}")