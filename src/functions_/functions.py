import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from pipelines_.pipelines import features_pipeline


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


def get_notams_data():
    conn = sqlite3.Connection(root + "/data/svo_db_20201027.db")
    sql = """ SELECT * FROM notams"""
    notams_df = pd.read_sql_query(sql, conn)
    conn.close()

    # create dataframe with coluimns of interest
    notams_df = notams_df[
        [
            "TEXT",
            "ISSUE_DATE",
            "POSSIBLE_START_DATE",
            "CLASSIFICATION",
            "LOCATION_CODE",
            "ACCOUNT_ID",
        ]
    ]
    notams_df = notams_df.dropna()
    # notams_df = notams_df.head(2)

    # run data pipeline on the notams dataframe
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(notams_df["CLASSIFICATION"])
    features_pipeline.set_params(
        **{
            "preprocess__columns__location_code_idx_3__cat_boost__target": target,
            "add_delta_time_feature_idx_6__column_indexes": [1, 2],
            # "add_text_embedder_feature_idx_5__column_index": 0,
        }
    )
    notams_data = features_pipeline.fit_transform(notams_df)
    return notams_data
