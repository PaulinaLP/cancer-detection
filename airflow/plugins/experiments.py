import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def comp_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    min_tpr: float = 0.80,
):
    v_gt = abs(np.asarray(solution.values) - 1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (
        partial_auc_scaled - 0.5
    )
    return partial_auc


def spliting(df, n_splits=10):
    n_splits = 10
    gkf = GroupKFold(n_splits=n_splits)
    df = df.sample(frac=1).reset_index(drop=True)
    df["fold"] = -1
    for idx, (_, val_idx) in enumerate(
        gkf.split(df, df["target"], groups=df["patient_id"])
    ):
        df.loc[val_idx, "fold"] = idx
    return df


def run_optimization(df, n_splits=10, num_trials=10):
    df = spliting(df, n_splits=n_splits)

    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)
            lgb_scores = []
            train_cols = list(df.columns)
            for c in ["fold", "target", "isic_id", "patient_id"]:
                train_cols.remove(c)
            for fold in range(n_splits):
                _df_train = df[df["fold"] != fold].reset_index(drop=True)
                _df_valid = df[df["fold"] == fold].reset_index(drop=True)
                model = LGBMClassifier(**params)
                model.fit(_df_train[train_cols], _df_train["target"])
                preds = model.predict_proba(_df_valid[train_cols])[:, 1]
                score = comp_score(
                    _df_valid[["target"]],
                    pd.DataFrame(preds, columns=["prediction"]),
                    "",
                )
                lgb_scores.append(score)
            lgbm_score = np.mean(lgb_scores)
            mlflow.log_metric('partial_auc', lgbm_score)
            return {
                'loss': -lgbm_score,
                'status': STATUS_OK,
            }  # Negative because we want to maximize the score

    search_space = {
        'num_leaves': scope.int(hp.quniform('num_leaves', 20, 50, 1)),
        'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 20, 100, 1)),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': scope.int(hp.quniform('bagging_freq', 1, 10, 1)),
        'lambda_l1': hp.uniform('lambda_l1', 0.0, 1.0),
        'lambda_l2': hp.uniform('lambda_l2', 0.0, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -5, -2),
        'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1500, 1)),
        'objective': 'binary',
        'random_state': 42,
    }

    rstate = np.random.default_rng(42)  # for reproducible results

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate,
    )

    print(f"Best parameters: {best}")
