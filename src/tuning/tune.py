import optuna
import mlflow
from src.tuning.objective import create_objective
import yaml
import os

tuning_config = yaml.safe_load(open("configs/optuna_config.yaml"))["tuning"]
mlflow.set_experiment(tuning_config["study_name"])

os.makedirs("optuna", exist_ok=True)

def objective_with_logging(trial):
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        result = objective(trial)
        mlflow.log_params(trial.params)
        mlflow.log_metric("best_f1", result)
        return result

objective = create_objective()
mlflow.end_run()  

with mlflow.start_run(run_name="optuna_study"):
    study = optuna.create_study(
        study_name=tuning_config["study_name"],
        direction=tuning_config["direction"],
        pruner=getattr(optuna.pruners, tuning_config["pruner"])()
    )
    study.optimize(objective_with_logging, n_trials=tuning_config["n_trials"])
    
    best_params = study.best_params
    print(f"Best {tuning_config['metric']}: {study.best_value:.4f}")
    print("Best params:", best_params)
    
    mlflow.log_params({"best_" + k: v for k, v in best_params.items()})
    mlflow.log_metric(f"best_{tuning_config['metric']}", study.best_value)
    
    with open("optuna/best_params.yaml", "w") as f:
        yaml.dump(best_params, f)
    mlflow.log_artifact("optuna/best_params.yaml")
