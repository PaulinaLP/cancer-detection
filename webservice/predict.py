import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np


def preprocess(features):
    preprocessor = joblib.load('model/preprocessor.pkl')
    df = pd.DataFrame([features])
    df = preprocessor.transform(df)
    return df


def predict(features):
    df = preprocess(features)
    test_cols = list(df.columns)
    for c in ["isic_id", "patient_id"]:
        if c in test_cols:
            test_cols.remove(c)
    models = joblib.load('model/models.pkl')
    lgb_preds = np.mean(
        [model.predict_proba(df[test_cols])[:, 1] for model in models], axis=0
    )
    return float(lgb_preds)


app = Flask('cancer-detection')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    features = request.get_json()
    pred = predict(features)

    result = {'cancer_probability': pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
