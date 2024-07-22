mlflow server \
    --backend-store-uri sqlite:////workspaces/cancer-detection/airflow/mlflow/mlflow.db
    --default-artifact-root ./mlruns \
    --host 0.0.0.0