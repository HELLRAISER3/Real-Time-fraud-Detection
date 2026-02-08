import yaml
import mlflow
from src.data_pipeline.data_pipeline import preprocess_data
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score
from src.training.models import get_model
from src.data_pipeline.load_dataset import load_dataset
import numpy as np


with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

df = load_dataset(filepath="data/training/training_dataset.parquet",
                  ext="parquet")

X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(df=df, 
                                                                 target='label',
                                                                 test_size=0.15,
                                                                 val_size=0.1,
                                                                 standardization=config["training"]["standardization"])

mlflow.set_experiment("loan_risk_prediction")

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

    f1_scores = (2 * precision_t * recall_t) / (precision_t + recall_t + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    best_threshold = thresholds[best_idx]
    best_precision = precision_t[best_idx]
    best_recall = recall_t[best_idx]
    best_f1 = f1_scores[best_idx]


    mlflow.log_param("model_name", config["model"]["name"])
    mlflow.log_params(config["model"][config["model"]["name"]])
    mlflow.log_metric("val_auc", auc)
    mlflow.log_param("best_threshold", float(best_threshold))
    mlflow.log_metric("best_precision", float(best_precision))
    mlflow.log_metric("best_recall", float(best_recall))
    mlflow.log_metric("best_f1", float(best_f1))

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
    )
    print(f"Validation AUC: {auc:.4f}")

