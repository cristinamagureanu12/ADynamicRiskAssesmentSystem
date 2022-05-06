"""
Author: Cristina Magureanu
Date: May 2021
"""
import json
import logging
import os
import sys

import pandas as pd
from flask import Flask, jsonify, request, session
from joblib import load
from sklearn import metrics

from preprocessing import preprocess_data

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#################Load config.json and get path variables
with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    """
    this function should take a trained model, load test data,
    and calculate an F1 score for the model relative to the test data
    it should write the result to the latestscore.txt file
    """

    logging.info(f"Loading test data from {test_data_path}/testdata.csv")
    data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    logging.info(f"Loading trained model from {model_path}/trainedmodel.pkl")
    model = load(os.path.join(model_path, "trainedmodel.pkl"))

    logging.info("Predict test data")
    data_x, data_y = preprocess_data(data)
    y_pred = model.predict(data_x)

    f1_score = metrics.f1_score(data_y, y_pred)
    logging.info(f"Calculate F1 score: {f1_score}")

    logging.info(f"Save f1 score to {model_path}/latestscore.txt")
    with open(os.path.join(model_path, "latestscore.txt"), "w", encoding="utf8") as score_file:
        score_file.write(f"{str(f1_score)}\n")

    return f1_score

if __name__ == "__main__":
    logging.info("Running scoring.py")
    score_model()
