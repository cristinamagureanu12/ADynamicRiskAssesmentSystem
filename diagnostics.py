"""
Author: Cristina Magureanu
Date: May 2021
"""
import json
import os
import subprocess
import sys
import timeit
import logging

import numpy as np
import pandas as pd
from joblib import load

from preprocessing import preprocess_data

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get environment variables
with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def model_predictions(x_data):
    """
    Function to get model predictions
    Read the deployed model and a test dataset, calculate predictions
    """
    logging.info(
        f"Loading deployed model from {prod_deployment_path}/trainedmodel.pkl")
    model = load(os.path.join(prod_deployment_path, "trainedmodel.pkl"))

    logging.info("Calculate predictions on data")
    y_pred = model.predict(x_data)

    return y_pred


def dataframe_summary():
    """
    Function to get summary statistics
    Return value should be a list containing all summary statistics
    """
    logging.info(f"Loading data from {dataset_csv_path}/finaldata.csv")
    dataframe = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    dataframe = dataframe.drop(['exited'], axis=1)

    numeric_columns = list(
        dataframe.dtypes[dataframe.dtypes == np.int64].index)

    logging.info("Calculating statistics for data")
    statistics = {}
    for col in numeric_columns:
        statistics[col] = {"mean": dataframe[col].mean(),
                           "median": dataframe[col].median(),
                           "standard deviation": dataframe[col].std()}

    return statistics


def missing_data():
    """
    Function to check missing data
    Return a list with number of NA values in each column of dataset
    and a percent of NA values from each column
    """
    logging.info(f"Loading data from {dataset_csv_path}/finaldata.csv")
    dataframe = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    logging.info("Calculating missing data percentage")
    missing_list = {}
    total_values = dataframe.shape[0]
    for column in dataframe.columns:
        missing_list[column] = {'missing_value_perc': (
            dataframe[column].isna().sum() / total_values * 100)}

    return missing_list


def execution_time():
    """
    Function to get timings
    Calculate timing of training.py and ingestion.py
    Return a list of 2 timing values in seconds
    """
    timing = {}
    scripts = ["training.py", "ingestion.py"]
    for procedure in scripts:
        logging.info(f"Calculating time for {procedure}")
        starttime = timeit.default_timer()
        _ = subprocess.run(['python3', procedure], capture_output=True)
        time = timeit.default_timer() - starttime
        timing[procedure] = time

    return timing


def outdated_packages_list():
    """
    Function to check dependencies
    Get a list of outdated packages
    """
    logging.info("Checking outdated dependencies")
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated_packages)


if __name__ == '__main__':
    logging.info("Running diagnostics.py")
    logging.info(f"Loading and preparing data from {test_data_path}/testdata.csv")
    test_dataframe = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    x_data = test_dataframe.drop(['corporation', 'exited'], axis=1)

    print("Model predictions:",
          model_predictions(x_data), end='\n')

    print("Summary statistics")
    print(json.dumps(dataframe_summary(), indent=4), end='\n')

    print("Missing percentage")
    print(json.dumps(missing_data(), indent=4), end='\n')

    print("Execution time")
    print(json.dumps(execution_time(), indent=4), end='\n')

    print("Outdated Packages")
    print(outdated_packages_list())
