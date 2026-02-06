import yaml
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score
from src.training.models import get_model


with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

df = pd.read_parquet("data/training/training_dataset.parquet")

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=config["training"]["test_size"],
    random_state=config["training"]["random_state"],
    stratify=y,
)

mlflow.set_experiment("fraud_detection")

with mlflow.start_run(run_name=config["model"]["name"]):

    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = get_model(config, xgb_scale_pos_weight=imbalance_ratio)

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)

    precision_arr, recall_arr, thresholds = precision_recall_curve(
        y_val, y_pred_proba
    )

    precision_t = precision_arr[:-1]
    recall_t = recall_arr[:-1]

    mask = precision_t >= 0.05

    if mask.any():
        best_idx = recall_t[mask].argmax()
        best_threshold = thresholds[mask][best_idx]

        best_precision = precision_t[mask][best_idx]
        best_recall = recall_t[mask][best_idx]
    else:
        best_threshold = 0.5
        best_precision = precision_score(y_val, (y_pred_proba >= 0.5).astype(int))
        best_recall = recall_score(y_val, (y_pred_proba >= 0.5).astype(int))

    mlflow.log_param("model_name", config["model"]["name"])
    mlflow.log_params(config["model"][config["model"]["name"]])
    mlflow.log_metric("val_auc", auc)
    mlflow.log_param("best_threshold", float(best_threshold))
    mlflow.log_metric("best_precision", float(best_precision))
    mlflow.log_metric("best_recall", float(best_recall))

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
    )
    print(f"Validation AUC: {auc:.4f}")
