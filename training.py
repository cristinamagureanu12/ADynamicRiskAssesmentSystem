"""
Author: Cristina Magureanu
Date: May 2021
"""
import json
import logging
import os
import sys

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_data

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get path variables
with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


def train_model():
    """
    Function for training the model
    """
    logging.info(f"Loading data from {dataset_csv_path}/finaldata.csv")
    data = pd.read_csv(f"{dataset_csv_path}/finaldata.csv")
    data_x, data_y = preprocess_data(data)

    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='ovr', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # fit the logistic regression to your data
    logging.info("Training model")
    x_train, _, y_train, _ = train_test_split(
        data_x, data_y, test_size=0.20)
    model.fit(x_train, y_train)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info(f"Saving model to {model_path}/trainedmodel.pkl")
    dump(model, os.path.join(model_path, "trainedmodel.pkl"))


if __name__ == "__main__":
    logging.info("Running training.py")
    train_model()
