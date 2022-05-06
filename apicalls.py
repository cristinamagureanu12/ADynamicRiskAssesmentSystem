"""
Author: Cristina Magureanu
Date: May 2021
"""
import json
import logging
import os
import sys

import requests

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Call each API endpoint and store the responses
logging.info(f"Post request /prediction for {test_data_path}/'testdata.csv'")
response1 = requests.post(
    f"{URL}/prediction", json={"dataset_path": os.path.join(test_data_path, "testdata.csv")}).text

logging.info("Get request /scoring")
response2 = requests.get(f"{URL}/scoring").text

logging.info("Get request /summarystats")
response3 = requests.get(f"{URL}/summarystats").text

logging.info("Get request /diagnostics")
response4 = requests.get(f"{URL}/diagnostics").text

# write the responses to your workspace
logging.info(f"Save api responses to {model_path}/apireturns.txt")
with open(os.path.join(model_path, 'apireturns.txt'), 'w', encoding='utf8') as file:
    file.write('Model Predictions\n')
    file.write(response1)
    file.write('\nModel Score\n')
    file.write(response2)
    file.write('Statistics Summary\n')
    file.write(response3)
    file.write('\nDiagnostics Summary\n')
    file.write(response4)
