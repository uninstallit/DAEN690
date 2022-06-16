from ast import Param
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import copy
import random
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.preprocessing import LabelEncoder

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(root)


class ParamWriter:
    def __init__(self, path):
        self.path = path

    def write(self, param_dict):
        with open(self.path, "r", encoding="utf-8") as f:
            _dict = json.loads(f.read())
            _dict.update(param_dict)
            f.close()

        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_dict, ensure_ascii=False, indent=4))
            f.close()


def get_matches_index_dict(matches_df, launches_df):
    index = launches_df["LAUNCHES_REC_ID"].apply(
        lambda x: matches_df[matches_df["LAUNCHES_REC_ID"] == x][
            "NOTAM_REC_ID"
        ].tolist()
    )
    index_dict = dict(
        (key, val) for (key, val) in index.to_dict().items() if len(val) != 0
    )
    return index_dict


def get_triplet_index_dict():
    conn = sqlite3.Connection(root + "/data/svo_db_20201027.db")
    sql = """ SELECT * FROM human_matches"""
    matches_df = pd.read_sql_query(sql, conn)
    sql = """ SELECT * FROM launches"""
    launches_df = pd.read_sql_query(sql, conn)
    conn.close()

    anchor_index = []
    positive_index = []
    negative_index = []
    matches_dict = get_matches_index_dict(matches_df, launches_df)
    matches_dict_keys = list(matches_dict.keys())

    for key, values in matches_dict.items():
        temp_keys = copy.deepcopy(matches_dict_keys)
        temp_keys.remove(key)
        perms = permutations(values, 2)

        for pair in perms:
            rand_key = random.choice(temp_keys)
            temp_values = matches_dict[rand_key]
            anchor, positive = pair
            negative = random.choice(temp_values)
            anchor_index.append(anchor)
            positive_index.append(positive)
            negative_index.append(negative)
    return (anchor_index, positive_index, negative_index)
