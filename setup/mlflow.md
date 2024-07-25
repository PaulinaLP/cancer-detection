Mlflow is run on airflow.
You must cd to airflow.
Then you must install pipenv and install all dependency from pipfile.
Then run pipenv shell and run.

mlflow server \
    --backend-store-uri sqlite:////workspaces/cancer-detection/airflow/mlflow/mlflow.db
    --default-artifact-root ./mlruns \
    --host 0.0.0.0

After that you can see the UI on port 5000.