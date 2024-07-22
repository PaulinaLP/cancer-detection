from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from ingest import ingest_data
from preprocess import prepare_train, Preprocessor
import os
import sys
import pandas as pd
import joblib
import mlflow


EXPERIMENT_NAME = "cancer_detection"
MLFLOW_TRACKING_URI=r"sqlite:////opt/airflow/mlflow/mlflow.db"


default_args = {
   'owner': 'airflow'
}

                 
def prep_train(ti):
    df_train, _ = ti.xcom_pull(task_ids='read_csv_file') 
    df_train = prepare_train(df_train)
    return df_train


def preprocess_train(ti):
    df_train = ti.xcom_pull(task_ids='prep_train') 
    preprocessor = Preprocessor()
    preprocessor.fit(df_train)
    df_train=preprocessor.transform(df_train)    
    joblib.dump(preprocessor, "/opt/airflow/output/preprocessor.pkl")
    return df_train  


def log_preprocessor():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    preprocessor = joblib.load("/opt/airflow/output/preprocessor.pkl")
    with mlflow.start_run():
       artifact_path = "preprocessor/preprocessor"
       mlflow.log_artifact(local_path="/opt/airflow/output/preprocessor.pkl",artifact_path=artifact_path)
   

def save_csv_file(ti):
    df_train = ti.xcom_pull(task_ids='preprocess_train')  
    df_train = df_train.head(10)   
    df_train.to_csv (os.path.join("/opt/airflow/output/example.csv"))


with DAG(
    dag_id = 'training_pipeline_5',
    description = 'Running a Python pipeline for training',
    default_args = default_args,
    start_date = days_ago(1),
    schedule_interval = '@once',
    tags = ['python', 'transform', 'pipeline']
) as dag:   
    read_csv_file = PythonOperator(
        task_id='read_csv_file',
        python_callable= ingest_data,
        op_kwargs={'path': "/opt/airflow/input/"}
    )        
    prep_train = PythonOperator(
        task_id='prep_train',
        python_callable= prep_train
        )    
    preprocess_train = PythonOperator(
        task_id='preprocess_train',
        python_callable= preprocess_train
        )  
    log_preprocessor = PythonOperator(
        task_id='log_preprocessor',
        python_callable=log_preprocessor        
    )
    save_csv_file = PythonOperator(
        task_id='save_csv_file',
        python_callable=save_csv_file
    )   

    
read_csv_file >> prep_train >> preprocess_train >> [log_preprocessor, save_csv_file]