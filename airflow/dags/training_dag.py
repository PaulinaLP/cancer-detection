from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from ingest import ingest_data
import os
import sys
import pandas as pd

default_args = {
   'owner': 'airflow'
}


def save_csv_file(ti):
    df_train, df_test = ti.xcom_pull(task_ids='read_csv_file')  
    example=df_train.head(10)
    example.to_csv (os.path.join("/opt/airflow/output//example-metadata.csv"))


def hello ():       
    with open("/opt/airflow/output/file.txt", 'w') as file:        
        file.write("hello")   


def read_file():
    file_path = "/opt/airflow/output/file.txt"
    with open(file_path, 'r') as file:
            content = file.read()
            print(f"File content: {content}")
           

with DAG(
    dag_id = 'training_pipeline_2',
    description = 'Running a Python pipeline for training',
    default_args = default_args,
    start_date = days_ago(1),
    schedule_interval = '@once',
    tags = ['python', 'transform', 'pipeline']
) as dag:    
    hello = PythonOperator(
        task_id='hello',
        python_callable= hello
    )
    read_file = PythonOperator(
        task_id='read_file',
        python_callable= read_file
    )
    read_csv_file = PythonOperator(
        task_id='read_csv_file',
        python_callable= ingest_data,
        op_kwargs={'path':  "/opt/airflow/input/"}
    )
    save_csv_file = PythonOperator(
        task_id='save_csv_file',
        python_callable=save_csv_file
    )   

    
hello >> read_file >> read_csv_file >> save_csv_file