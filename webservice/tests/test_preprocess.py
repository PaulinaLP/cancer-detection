from predict import preprocess
import json
import pandas


def test_preprocess():
    with open('example.json', 'r') as file:    
        features = json.load(file)
    actual_features=preprocess(features)
    with open('transformed_data.json', 'r') as file:
        expected_features_json= json.load(file)  
    expected_features=pd.DataFrame([expected_features])  
    assert actual_features.equals(expected_features) 