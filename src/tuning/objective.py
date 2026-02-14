import yaml
import copy  
from src.data_pipeline.data_pipeline import preprocess_data
from src.data_pipeline.load_dataset import load_dataset
from src.training.models import get_model
from sklearn.metrics import precision_recall_curve
import numpy as np

def create_objective(config_path="configs/training_config.yaml", tuning_config_path="configs/optuna_config.yaml"):
    def objective(trial):
        trial.suggest_int("max_depth", 3, 10)
        trial.suggest_float("learning_rate", 0.01, 0.3)
        trial.suggest_int("n_estimators", 100, 1000)
        trial.suggest_float("subsample", 0.6, 1.0)
        
        df = load_dataset(filepath="data/training/training_dataset.parquet", ext="parquet")
        training_config = yaml.safe_load(open(config_path))
        trial_config = copy.deepcopy(training_config)
        
        model_name = trial_config["model"]["name"]
        for param in ["max_depth", "learning_rate", "n_estimators", "subsample"]:
            trial_config["model"][model_name][param] = trial.params[param]
        
        X_train, y_train, X_val, y_val, _, _ = preprocess_data(
            df, target='label', test_size=0.15, val_size=0.1,
            standardization=trial_config["training"]["standardization"]
        )
        
        imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        model = get_model(trial_config, xgb_scale_pos_weight=imbalance_ratio)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        precision_arr, recall_arr, thresholds = precision_recall_curve(y_val, y_pred_proba)
        precision_t, recall_t = precision_arr[:-1], recall_arr[:-1]
        f1_scores = (2 * precision_t * recall_t) / (precision_t + recall_t + 1e-10)
        best_f1 = np.max(f1_scores)
        
        return best_f1 
    
    return objective
