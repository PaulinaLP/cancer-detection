import json

import pytest
from predict import app


# pylint: disable=redefined-outer-name
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


# pylint: disable=redefined-outer-name
def test_predict(client):

    with open('example.json', 'r', encoding='utf-8') as file:
        sample_input = json.load(file)

    # Send a POST request to the /predict endpoint
    response = client.post('/predict', json=sample_input)

    # Check the response status code
    assert response.status_code == 200

    # Check the response data
    response_data = json.loads(response.data)
    assert 'cancer_probability' in response_data
    assert isinstance(response_data['cancer_probability'], float)
