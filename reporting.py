"""
Author: Cristina Magureanu
Date: May 2021
"""
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

from diagnostics import model_predictions
from preprocessing import preprocess_data
from pretty_confusion_matrix import plot_confusion_matrix_from_data


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

###############Load config.json and get path variables
with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

def generate_confusion_matrix():
    """
    Function for reporting
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    """
    logging.info(f"Loading data from {test_data_path}/testdata.csv")
    data = pd.read_csv(f"{test_data_path}/testdata.csv")
    data_x, data_y = preprocess_data(data)
    logging.info("Predicting test data")
    y_pred = model_predictions(data_x)

    logging.info("Plotting and saving confusion matrix")
    fig, ax = plot_confusion_matrix_from_data(
        data_y, y_pred, columns=[0, 1], cmap='Blues')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(model_path, "confusionmatrix.png"))

if __name__ == '__main__':
    logging.info("Running reporting.py")
    test_dataframe = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    generate_confusion_matrix()
