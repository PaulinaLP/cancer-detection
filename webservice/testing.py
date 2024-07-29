import requests
import json
import pandas as pd

with open('example.json', 'r') as file:
    features = json.load(file)


url = 'http://localhost:9696/predict'
response = requests.post(url, json=features)
content = response.json()
print(content)
