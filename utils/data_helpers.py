# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import logging
import json
import csv
import torch
import pandas as pd

from texttable import Texttable


def option():
    """
    Choose training or restore pattern.

    Returns:
        The OPTION
    """
    OPTION = input("[Input] Train or Restore? (T/R): ")
    while not (OPTION.upper() in ['T', 'R']):
        OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    return OPTION.upper()


def logger_fn(name, input_file, level=logging.INFO):
    """
    The Logger.

    Args:
        name: The name of the logger
        input_file: The logger file path
        level: The logger level
    Returns:
        The logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(input_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)
    return logger


def tab_printer(args, logger):
    """
    Function to print the logs in a nice tabular format.

    Args:
        args: Parameters used for the model.
        logger: The logger
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]])
    logger.info('\n' + t.draw())


def get_model_name():
    """
    Get the model name used for test.

    Returns:
        The model name
    """
    MODEL = input("[Input] Please input the model file you want to test, it should be like (1490175368): ")

    while not (MODEL.isdigit() and len(MODEL) == 10):
        MODEL = input("[Warning] The format of your input is illegal, "
                      "it should be like (1490175368), please re-input: ")
    return MODEL


def create_prediction_file(save_dir, identifiers, predictions):
    """
    Create the prediction file.

    Args:
        save_dir: The all classes predicted results provided by network
        identifiers: The data record id
        predictions: The predict scores
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    preds_file = os.path.abspath(os.path.join(save_dir, 'predictions.csv'))
    out = pd.DataFrame()
    out["id"] = identifiers
    out["predictions"] = [round(float(i), 4) for i in predictions]
    out.to_csv(preds_file, index=None)


def load_data_and_labels(input_file):
    """
    Create the research data tokenindex based on the word2vec model file.
    Return the class _Data() (includes the data tokenindex and data labels).

    Args:
        input_file: The stock file
    Returns:
        The Class _Data() (includes the data features and data labels)
    Raises:
        IOError: If the input file is not the .json file
    """
    if not input_file.endswith('.json'):
        raise IOError("[Error] The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    with open(input_file) as fin:
        id_list = []
        features_list = []
        labels_list = []

        for eachline in fin:
            data = json.loads(eachline)
            id = data['stock']
            features = data['features']
            labels = float(data['label'][0])

            features_list.append(features)
            labels_list.append(labels)

    class _Data:
        def __init__(self):
            pass

        @property
        def id(self):
            return id_list

        @property
        def features(self):
            return features_list

        @property
        def labels(self):
            return labels_list

    return _Data()


def convert_data(data):
    """
    Args:
        data: The research data
    Returns:
        features: The features
        labels: The data labels
    """
    features = torch.tensor(data.features)
    labels = torch.tensor(data.labels)

    return features, labels
