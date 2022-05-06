"""
Author: Cristina Magureanu
Date: May 2021
"""
import json
import logging
import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

import deployment
import diagnostics
import ingestion
import preprocessing
import reporting
import scoring
import training

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open("config.json", "r", encoding="utf8") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

def main():
    # Check and read new data
    # first, read ingestedfiles.txt
    logging.info("Checking for new data")
    ingested_files = []
    ingested_files_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
    with open(ingested_files_path, "r", encoding="utf8") as report_file:
        for line in report_file:
            ingested_files.append(line.rstrip())

    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    NEW_FILES = False
    for filename in os.listdir(input_folder_path):
        filename_path = os.path.join(input_folder_path, filename)
        if filename_path not in ingested_files:
            NEW_FILES = True

    # # Deciding whether to proceed, part 1
    # # if you found new data, you should proceed. otherwise, do end the process here
    if not NEW_FILES:
        logging.info("No new data")
        sys.exit()

    logging.info("Ingesting new data")
    ingestion.merge_multiple_dataframe()

    # Checking for model drift
    # check whether the score from the deployed model is different from the score 
    # from the model that uses the newest ingested data
    logging.info("Checking for model drift")
    latest_score_path = os.path.join(prod_deployment_path, "latestscore.txt")
    with open(latest_score_path, "r", encoding="utf8") as latest_score_file:
        latest_score = float(latest_score_file.read())

    dataframe = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    data_x, data_y = preprocessing.preprocess_data(dataframe)
    y_pred = diagnostics.model_predictions(data_x)
    new_score = f1_score(data_y.values, y_pred)

    logging.info(f"Deployed score: {latest_score}")
    logging.info(f"New score: {new_score}")

    # Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the process here
    if (new_score > latest_score):
        logging.info("No model drift occurred")
        sys.exit()

    logging.info("Re-training model")
    training.train_model()

    # Re-deployment
    # if you found evidence for model drift, re-run the deployment.py script
    logging.info("Recalculate f1 score")
    scoring.score_model()

    logging.info("Redeploying model")
    deployment.store_model_into_pickle()

    # ##################Diagnostics and reporting
    # # run diagnostics.py and reporting.py for the re-deployed model
    logging.info("Running diagnostics and reporting")
    reporting.generate_confusion_matrix()
    print(json.dumps(diagnostics.dataframe_summary(), indent=4), end='\n')
    print(json.dumps(diagnostics.missing_data(), indent=4), end='\n')

    logging.info("Calling api")
    os.system("python apicalls.py")

if __name__ == '__main__':
    main()
