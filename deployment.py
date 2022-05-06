"""
Author: Cristina Magureanu
Date: May 2021
"""
import json
import os
from shutil import copy2
import sys
import logging


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and correct path variable
with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])


# function for deployment
def store_model_into_pickle():
    """
    copy the latest pickle file, the latestscore.txt value,
    and the ingestfiles.txt file into the deployment directory
    """
    logging.info("Deploying trained model to production")
    files = [(model_path, "latestscore.txt"),
             (dataset_csv_path, "ingestedfiles.txt"),
             (model_path, "trainedmodel.pkl")]

    for file in files:
        old_path = os.path.join(file[0], file[1])
        new_path = os.path.join(prod_deployment_path, file[1])
        logging.info(f"Copying {old_path} to {new_path}")
        copy2(old_path, new_path)


if __name__ == '__main__':
    logging.info("Running deployment.py")
    store_model_into_pickle()
