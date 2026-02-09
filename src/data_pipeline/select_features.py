import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan_risk_prediction")

experiment = mlflow.get_experiment_by_name("loan_risk_prediction")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

if runs.empty:
    print("No trained model found")
    exit(1)

latest_run = runs.iloc[0]
run_id = latest_run["run_id"]

model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

df = pd.read_parquet("data/training/training_dataset.parquet")
X = df.drop(['SK_ID_CURR', 'label'], axis=1)
feature_names = X.columns.tolist()

print(f"\nOriginal num of features: {len(feature_names)}")

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_

    if len(importances) != len(feature_names):
        y = df['label']
        X = df.drop(['SK_ID_CURR', 'label'], axis=1)

        correlations = X.corrwith(y).abs().sort_values(ascending=False)

        feature_importance_df = pd.DataFrame({
            'feature': correlations.index,
            'importance': correlations.values
        }).dropna()
    else:
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

    print("\nTop 20 most important features:")
    print(feature_importance_df.head(20).to_string(index=False))

    plt.figure(figsize=(12, 8))
    top_n = 30
    plt.barh(range(top_n), feature_importance_df['importance'].head(top_n))
    plt.yticks(range(top_n), feature_importance_df['feature'].head(top_n))
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('data/training/feature_importance.png', dpi=150, bbox_inches='tight')

    feature_importance_df['cumulative_importance'] = feature_importance_df['importance'].cumsum()
    feature_importance_df['cumulative_importance_pct'] = (
        feature_importance_df['cumulative_importance'] /
        feature_importance_df['importance'].sum() * 100
    )

    thresholds = [0.90, 0.95, 0.99]

    for threshold in thresholds:
        n_features = (feature_importance_df['cumulative_importance_pct'] <= threshold * 100).sum()
        if n_features <= 20:
            selected_features = feature_importance_df.head(n_features)['feature'].tolist()
            print(f"Features: {', '.join(selected_features)}")

    recommended_threshold = 0.95
    n_recommended = (feature_importance_df['cumulative_importance_pct'] <= recommended_threshold * 100).sum()

    import sys
    if len(sys.argv) > 1:
        n_features_to_keep = int(sys.argv[1])
    else:
        n_features_to_keep = min(30, n_recommended)  

    selected_features = feature_importance_df.head(n_features_to_keep)['feature'].tolist()

    for i, feat in enumerate(selected_features, 1):
        imp = feature_importance_df[feature_importance_df['feature'] == feat]['importance'].values[0]

    OUTPUT_FILE = "data/training/selected_features.txt"
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    reduced_df = df[['SK_ID_CURR', 'label'] + selected_features]

    reduced_df.to_parquet("data/training/training_dataset_reduced.parquet", index=False)
    reduced_df.to_csv("data/training/training_dataset_reduced.csv", index=False)

else:
    print("Model does not have feature_importances_ attribute")
