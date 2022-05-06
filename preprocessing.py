"""
Author: Cristina Magureanu
Date: May 2021
"""

def preprocess_data(data):
    """
    This function is used to prepare datasets to be trained
    Provise the data split into x_data and y_data and an improvised encoder
    """
    data_y = data.pop("exited")
    data_x = data.drop(["corporation"], axis=1)

    return data_x, data_y
