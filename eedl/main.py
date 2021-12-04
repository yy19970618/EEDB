import os
import pickle
import time
import logging
from typing import Dict, Any, Tuple

import numpy as np
from EHmscn_base import train_baseline
from TestEHmscn import Test_all

config = {"census13": [600, 4], "forest10": [5000, 8], "dmv11": [10000, 256], "power7": [10000, 16]}

for dataset_name in ["forest10"]:
    train_baseline(dataset_name, config[dataset_name])


config = {"census13": [600, 4], "dmv11": [10000, 256], "forest10": [5000, 8]}
for dataset_name in ["census13", "dmv11", "forest10"]:
    for model_query_num_true in [2500, 5000, 7500, 10000]:
        if dataset_name == "census13" and model_query_num_true <= 5000:
            print(dataset_name + " " + str(model_query_num_true) + " finished..")
        else:
            Test_all(dataset_name, model_query_num_true, config[dataset_name][0], config[dataset_name][1])
            print(dataset_name + " " + str(model_query_num_true) + " finished..")
