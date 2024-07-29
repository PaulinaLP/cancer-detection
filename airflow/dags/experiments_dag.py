from datetime import timedelta

import joblib
import mlflow
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from experiments import run_optimization
from ingest import ingest_data
from preprocess import Preprocessor, prepare_train
from register import run_register_model

from airflow import DAG

EXPERIMENT_NAME = "cancer_detection_lgbm"
EXPERIMENT_NAME_SELECTED = "cancer_detection_lgbm_best"
MLFLOW_TRACKING_URI = r"sqlite:////opt/airflow/mlflow/mlflow.db"
ARTIFACT_BUCKET = "/opt/airflow/mlruns/"


default_args = {'owner': 'airflow', 'execution_timeout': timedelta(minutes=5)}


def prep_train(ti):
    df_train, _ = ti.xcom_pull(task_ids='read_csv_file')
    df_train = prepare_train(df_train)
    return df_train


def preprocess_train(ti):
    df_train = ti.xcom_pull(task_ids='prep_train')
    preprocessor = Preprocessor()
    df_train = df_train.head(10000)
    preprocessor.fit(df_train)
    df_train = preprocessor.transform(df_train)
    joblib.dump(preprocessor, "/opt/airflow/output/preprocessor.pkl")
    return df_train


def run_experiments(ti):
    df_train = ti.xcom_pull(task_ids='preprocess_train')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME, artifact_location=ARTIFACT_BUCKET
        )
    except:
        pass
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_optimization(df_train)


def register_best_model(ti):
    df_train = ti.xcom_pull(task_ids='preprocess_train')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME_SELECTED, artifact_location=ARTIFACT_BUCKET
        )
    except:
        pass
    mlflow.set_experiment(EXPERIMENT_NAME_SELECTED)
    run_register_model(df_train, EXPERIMENT_NAME_SELECTED)


with DAG(
    dag_id='experimenting',
    description='Running a Python pipeline for training',
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval='@once',
    tags=['python', 'transform', 'pipeline'],
) as dag:
    read_csv_file = PythonOperator(
        task_id='read_csv_file',
        python_callable=ingest_data,
        op_kwargs={'path': "/opt/airflow/input/"},
    )
    prep_train = PythonOperator(task_id='prep_train', python_callable=prep_train)
    preprocess_train = PythonOperator(
        task_id='preprocess_train', python_callable=preprocess_train
    )
    run_experiments = PythonOperator(
        task_id='run_experiments', python_callable=run_experiments
    )
    register_best_model = PythonOperator(
        task_id='register_best_model', python_callable=register_best_model
    )

    # pylint: disable=pointless-statement
    (
        read_csv_file
        >> prep_train
        >> preprocess_train
        >> run_experiments
        >> register_best_model
    )
