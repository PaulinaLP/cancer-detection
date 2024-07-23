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

def train_and_log_model(df, params, name):
    n_splits=10
    df = spliting(df, n_splits=n_splits)
    with mlflow.start_run():     
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
        mlflow.log_artifact(local_path="/opt/airflow/outputmodels.pkl",artifact_path="models")
        mlflow.log_artifact(local_path="/opt/airflow/output/preprocessor.pkl",artifact_path="preprocessor")
        

def run_register_model(df, hpo_experiment_name, top_n=1 ):
    client = MlflowClient()    
    #Select the model 
    experiment = client.get_experiment_by_name(hpo_experiment_name)
    best_run = client.search_runs(experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=(["metrics.partial_auc DESC"]))[0]
    train_and_log_model(params=best_run.data.params, name=str('best_model'))
    # Register the best model
    mlflow.register_model(
        model_uri=f"runs:/{best_run.info.run_id}/model",
        name="best_model"
    )
