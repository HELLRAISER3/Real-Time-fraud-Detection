from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

def get_model(config):
    name = config["model"]["name"]

    if name == "logistic_regression":
        params = config["model"]["logistic_regression"]
        return LogisticRegression(**params)

    if name == "random_forest":
        params = config["model"]["random_forest"]
        return RandomForestClassifier(**params)

    # if name == "xgboost":
    #     params = config["model"]["xgboost"]
    #     return XGBClassifier(
    #         **params,
    #         eval_metric="auc",
    #         tree_method="hist",
    #         random_state=42,
    #     )

    raise ValueError(f"Unknown model: {name}")
