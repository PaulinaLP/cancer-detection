import mlflow
import joblib
import numpy as np


def testing():
       EXPERIMENT_NAME = "test"
       MLFLOW_TRACKING_URI=r"sqlite:///mlflow/mlflow.db"
       mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
       mlflow.set_experiment(EXPERIMENT_NAME)
       with mlflow.start_run():     
              with open("prueba.txt","wt") as file:
                     file.write("prueba")         
              mlflow.log_artifact("prueba.txt")
              artifact_path = "preprocessor"
              mlflow.log_artifact(local_path="output/preprocessor.pkl",artifact_path=artifact_path)   


def testing_airflow(mlflow_tracking_uri):
       EXPERIMENT_NAME = "test_airflow"       
       mlflow.set_tracking_uri(mlflow_tracking_uri)
       mlflow.set_experiment(EXPERIMENT_NAME)
       with mlflow.start_run():     
              with open("prueba.txt","wt") as file:
                     file.write("prueba")         
              mlflow.log_artifact("prueba.txt")
              artifact_path = "preprocessor"
              mlflow.log_artifact(local_path="output/preprocessor.pkl",artifact_path=artifact_path)   
   
