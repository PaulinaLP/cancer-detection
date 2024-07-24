import os
import joblib
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from lightgbm import LGBMClassifier
from experiments import spliting, comp_score


def convert_params_to_float(params):
    """
    try to convert to int
    then try to convert to float
    leave as it is if it is not a number
    """
    for key, value in params.items():
        try:
            params[key] = int(value)
        except (ValueError, TypeError):
            try:
                params[key] = float(value)
            except (ValueError, TypeError):
                pass 
            pass  
    return params


def train_and_log_model(df, params, experiment_name):
    n_splits=10
    df = spliting(df, n_splits=n_splits)
    print(params)
    params=convert_params_to_float(params)
    with mlflow.start_run() as run:   
        run_id = run.info.run_id  
        current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
        current_experiment_id =current_experiment['experiment_id']        
        mlflow.log_params(params)
        lgb_scores = []
        lgb_models = []
        train_cols = list(df.columns)
        for c in ["fold", "target", "isic_id", "patient_id"]:
            train_cols.remove(c)        
        for fold in range(n_splits):
            _df_train = df[df["fold"] != fold].reset_index(drop=True)
            _df_valid = df[df["fold"] == fold].reset_index(drop=True)
            model = LGBMClassifier(**params)
            model.fit(_df_train[train_cols], _df_train["target"])
            preds = model.predict_proba(_df_valid[train_cols])[:, 1]
            score = comp_score(_df_valid[["target"]], pd.DataFrame(preds, columns=["prediction"]), "")
            lgb_scores.append(score)        
            lgb_models.append(model)  
        lgbm_score = np.mean(lgb_scores)
        mlflow.log_metric('partial_auc',lgbm_score)      
        joblib.dump(lgb_models, "/opt/airflow/output/models.pkl")
        mlflow.log_artifact(local_path="/opt/airflow/output/models.pkl",artifact_path=str(current_experiment_id)+"/model")       
        mlflow.log_artifact(local_path="/opt/airflow/output/preprocessor.pkl",artifact_path=str(current_experiment_id)+"preprocessor")
    return run_id   


def run_register_model(df, hpo_experiment_name, experiment_name, top_n=1 ):
    client = MlflowClient()    
    #Select the model 
    experiment = client.get_experiment_by_name(hpo_experiment_name)
    best_run = client.search_runs(experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=(["metrics.partial_auc DESC"]))[0]
    run_id=train_and_log_model(df, best_run.data.params,experiment_name)
    # Register the best model
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name="best_model"
    )
