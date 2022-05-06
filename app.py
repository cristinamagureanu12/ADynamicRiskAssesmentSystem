"""
Author: Cristina Magureanu
Date: May 2021
"""
import json
import logging
import os
import pickle
import sys

import pandas as pd
from flask import Flask, jsonify, request

import diagnostics
import reporting
import scoring

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """call the prediction function you created in Step 3"""
    dataset_path = request.get_json()['dataset_path']
    dataframe = pd.read_csv(dataset_path)
    dataframe = dataframe.drop(['corporation', 'exited'], axis=1)

    y_pred = diagnostics.model_predictions(dataframe)
    return jsonify(y_pred.tolist())  # add return value for prediction outputs

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """check the score of the deployed model"""
    f1_score = scoring.score_model()
    return str(f1_score)  # add return value (a single F1 score number)

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """check means, medians, and modes for each column"""
    summary = diagnostics.dataframe_summary()
    return jsonify(summary)  # return a list of all calculated summary statistics

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagn():
    """check timing and percent NA values"""
    time = diagnostics.execution_time()
    missing_data = diagnostics.missing_data()
    outdated_pack = diagnostics.outdated_packages_list()

    results = {
        "execution_time": time,
        "missing_data": missing_data,
        "outdated_packages": outdated_pack,
    }

    return jsonify(results)  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
