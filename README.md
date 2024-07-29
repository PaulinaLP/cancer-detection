# Problem description

The data for this project is taken from Kaggle competition: ISIC 2024 - Skin Cancer Detection with 3D-TBP (https://www.kaggle.com/competitions/isic-2024-challenge)
The complete description of the problem can be found on this website, so I provide here just a brief summary:

## Objective:
Develop binary classification algorithms to identify malignant skin lesions from benign ones.

## Data:
The model will leverage only the metadata of the photos to make predictions.
Some sample of the data are placed in airflow/input.
The complete data can be downloaded from: https://www.kaggle.com/competitions/isic-2024-challenge/data

## Model:
Type: LightGBM (Light Gradient Boosting Machine)
Purpose: To utilize LightGBM's efficiency and effectiveness in handling the classification task with the provided data.

## Evaluation Metric:
Primary Metric: Partial Area Under the ROC Curve (pAUC) above an 80% True Positive Rate (TPR).
Purpose: Focuses on areas where TPR is above 80% to ensure high sensitivity in cancer detection.

## Mode:
For this project cloud is not used. The project runs locally.


# ML-OPS features

## Experiment tracking and model registry
Experiment tracking and model registry is managed with **MlFlow**. MlFlow runs inside airflow pipeline. The basic code for MlFlow may be found in airflow/plugins/experiments.py, airflow/plugins/register.py and airflow/dags/experiments_dag.py. The instructions how to setup MlFlow are located in setup/mlflow.md

## Workflow orchestration
Workflow orchestration is managed with **airflow**. Airflow is run with docker. The airflow dags may be found in airflow/dags. They cover ingesting data, preprocessing, training, experiments running and registry of the model.
The instructions how to setup airflow are located in setup/airflow.md. At the end of this file I leave a short description about airflow functioning.

## Model deployment
The model is deployed as a web service with **Flask** and it is containerized with docker.The instructions how to setup webservice are located in setup/webservice.md.

## Model monitoring
The project includes a basic Dashboard created with **evidently**. The instructions how to setup evidently are located in setup/webservice.md.

## Testing
The test are run with pytest. They can be found in webservice/tests.

## Lintering and formating.
Pylint, black and isort are used for code formatting and linting.

### How to run the application?
You must run make setup or make setup --always-make 

### Airflow functioning:
Apache Airflow is an open-source platform used to programmatically author, schedule, and monitor workflows. It allows users to define workflows as Directed Acyclic Graphs (DAGs), which represent a sequence of tasks and their dependencies.

#### Key Concepts:

DAG (Directed Acyclic Graph): A DAG is a collection of tasks organized in a way that defines their execution order and dependencies. It ensures that each task runs only after its upstream tasks have completed.

Task: A single unit of work within a DAG. Tasks can be anything from running a script to executing a SQL query.

Operator: An operator defines what a task will do. Examples include BashOperator for running shell commands, PythonOperator for executing Python functions, and EmailOperator for sending emails.
