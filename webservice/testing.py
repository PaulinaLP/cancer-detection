import json

import pandas as pd
import requests

with open('example.json', 'r', encoding='utf-8') as file:
    features = json.load(file)


url = 'http://localhost:9696/predict'
response = requests.post(url, json=features)
content = response.json()
print(content)
