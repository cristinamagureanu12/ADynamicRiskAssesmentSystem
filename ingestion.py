"""
Author: Cristina Magureanu
Date: May 2021
"""
import glob
import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get input and output paths
with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    """
    check for datasets, compile them together, and write to an output file
    """
    logging.info(f"Reading files from {input_folder_path}")
    datasets = glob.glob(f"{input_folder_path}/*.csv")
    data = pd.concat(map(pd.read_csv, datasets), ignore_index=True)

    logging.info("Dropping duplicates")
    data.drop_duplicates(inplace=True)

    logging.info(f"Saving new dataset to {output_folder_path}/finaldata.csv")
    data.to_csv(f"{output_folder_path}/finaldata.csv", index=False)

    logging.info(f"Saving ingested files to {output_folder_path}/ingestedfiles.txt")
    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w", encoding="utf8") as file:
        for dataset in datasets:
            file.write(f"{dataset}\n")


if __name__ == '__main__':
    logging.info("Running ingestion.py")
    merge_multiple_dataframe()
