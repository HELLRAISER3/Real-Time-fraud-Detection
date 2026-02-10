from feast import FeatureStore
from typing import Dict, Any

class FeatureService:
    def __init__(self, repo_path: str):
        self.store = FeatureStore(repo_path=repo_path)
        self.feature_refs = [
            "applicant_risk_features:credit_to_income_ratio",
            "applicant_risk_features:annuity_to_income_ratio",
            "applicant_risk_features:credit_term_approx",
            "applicant_risk_features:goods_price_to_credit_ratio",
            "applicant_risk_features:age_years",
            "applicant_risk_features:years_employed",
            "applicant_risk_features:employed_to_age_ratio",
            "applicant_risk_features:income_per_person",
            "applicant_risk_features:ext_source_1",
            "applicant_risk_features:ext_source_2",
            "applicant_risk_features:ext_source_3",
            "applicant_risk_features:ext_source_mean",
            "applicant_risk_features:flag_own_car",
            "applicant_risk_features:flag_own_realty"
        ]

    def get_online_features(self, entity_id: int) -> Dict[str, Any]:
        try:
            resp = self.store.get_online_features(
                features=self.feature_refs,
                entity_rows=[{"SK_ID_CURR": entity_id}]
            ).to_dict()
            
            return {k.split(":")[-1]: v[0] for k, v in resp.items() if k != "SK_ID_CURR"}
        except Exception as e:
            print(f"Feast Fetch Error: {e}")
            return {}