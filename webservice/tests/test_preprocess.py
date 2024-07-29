import json

import pandas as pd
from predict import preprocess
from pandas.testing import assert_frame_equal


def test_preprocess():
    with open('example.json', 'r', encoding='utf-8') as file:
        features = json.load(file)
    actual_features = preprocess(features)
    with open('transformed_data.json', 'r', encoding='utf-8') as file:
        expected_features_json = json.load(file)
    expected_features = pd.DataFrame([expected_features_json])
    try:
        assert_frame_equal(actual_features, expected_features)
        print("Test passed: DataFrames are equal.")
        assert True
    except AssertionError as e:
        print("Test failed: DataFrames are not equal.")
        print(e)
        assert False  # Fail the test explicitly
