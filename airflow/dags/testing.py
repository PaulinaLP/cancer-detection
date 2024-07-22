import mlflow
import joblib

EXPERIMENT_NAME = "cancer_detection"
MLFLOW_TRACKING_URI=r"sqlite:///mlflow/mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run():
       artifact_path = "preprocessor/preprocessor"
       mlflow.log_artifact(local_path="output/preprocessor.pkl",artifact_path=artifact_path)
   
