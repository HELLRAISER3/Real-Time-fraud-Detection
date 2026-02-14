import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.impute import SimpleImputer

class ModelLoader:
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.model = None
        self.threshold = 0.5
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None

        mlflow.set_tracking_uri(tracking_uri)

    def load_model(self) -> bool:
        try:
            mlflow.set_experiment(self.experiment_name)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if not experiment:
                print(f"Experiment '{self.experiment_name}' not found")
                return False

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )

            if runs.empty:
                print("No runs found")
                return False

            latest_run = runs.iloc[0]
            run_id = latest_run["run_id"]

            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.sklearn.load_model(model_uri)

            if "params.best_threshold" in latest_run.index:
                self.threshold = float(latest_run["params.best_threshold"])

            print(f"Model loaded: {run_id}")

            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def prepare_features(self, request_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([request_data])

        if self.feature_names is None:
            self.feature_names = df.columns.tolist()
            self.imputer.fit(df)

        df_imputed = pd.DataFrame(
            self.imputer.transform(df),
            columns=self.feature_names
        )

        return df_imputed

    def predict(self, features: pd.DataFrame) -> tuple:
        if self.model is None:
            raise ValueError("Model not loaded")

        probabilities = self.model.predict_proba(features)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        return predictions[0], probabilities[0]

    def get_risk_level(self, probability: float) -> str:
        if probability < 0.3:
            return "low"
        elif probability < 0.7:
            return "medium"
        else:
            return "high"

    def is_loaded(self) -> bool:
        return self.model is not None
