import os
import joblib
import pandas as pd
import json

preprocessor=joblib.load('model/preprocessor.pkl')
with open('example.json', 'r') as file:    
    features = json.load(file)

df = pd.DataFrame([features])
df = preprocessor.transform(df)
df_transformed = pd.DataFrame(df)
df_transformed.to_json('transformed_data.json', orient='records', lines=True)

