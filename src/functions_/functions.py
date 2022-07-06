import sqlite3
import numpy as np
import pandas as pd
import json
import copy
import random
import matplotlib.pyplot as plt
from itertools import permutations
from numpy.random import default_rng

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


class ParamReader:
    def __init__(self, path):
        self.path = path
        self.param_dict = None

    def read(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.param_dict = json.loads(f.read())
            f.close()
        return self.param_dict


def get_matches_index_dict(matches_df):
    matches_dict = (
        matches_df[["LAUNCHES_REC_ID", "NOTAM_REC_ID"]]
        .groupby("LAUNCHES_REC_ID")
        .agg(lambda x: list(x))
        .to_dict()
    )["NOTAM_REC_ID"]
    return matches_dict


def get_triplet_index():
    conn = sqlite3.Connection(root + "/data/svo_db_20201027.db")
    sql = """ SELECT * FROM human_matches"""
    matches_df = pd.read_sql_query(sql, conn)
    conn.close()

    anchor_index = []
    positive_index = []
    negative_index = []
    matches_dict = get_matches_index_dict(matches_df)
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

    # this returns NOTAM_REC_ID - not dataframe index
    return (anchor_index, positive_index, negative_index)


def fromBuffer(byte_embeddings):
    _embeddings = np.array(
        [
            np.frombuffer(byte_embeddings, dtype=np.float32)
            for byte_embeddings in byte_embeddings
        ]
    ).astype(np.float32)
    return _embeddings


def inputNoneValues(df):
    _df = copy.deepcopy(df)
    _df["E_CODE"] = _df["E_CODE"].fillna("UNKNOWN")
    _df["ISSUE_DATE"] = _df["ISSUE_DATE"].fillna(
        _df["ISSUE_DATE"].value_counts().idxmax()
    )
    _df["POSSIBLE_START_DATE"] = _df["POSSIBLE_START_DATE"].fillna(
        _df["POSSIBLE_START_DATE"].value_counts().idxmax()
    )
    _df["CLASSIFICATION"] = _df["CLASSIFICATION"].fillna("UNKNOWN")
    _df["LOCATION_CODE"] = _df["LOCATION_CODE"].fillna("UNKNOWN")
    _df["ACCOUNT_ID"] = _df["ACCOUNT_ID"].fillna("UNKNOWN")
    return _df


def fromBuffer(byte_embeddings):
    _embeddings = np.array(
        [
            np.frombuffer(byte_embeddings, dtype=np.float32)
            for byte_embeddings in byte_embeddings
        ]
    ).astype(np.float32)
    return _embeddings


def get_triplet_index_from_good_bad():
    good_df = pd.read_csv(root + "/data/good_notams_list_2022.07.03.csv", sep=",")
    bad_df = pd.read_csv(root + "/data/bad_notams_list_2022.07.03.csv", sep=",")
    good_data = np.squeeze(good_df[["notam_rec_id"]].to_numpy())
    bad_data = np.squeeze(bad_df[["notam_rec_id"]].to_numpy())
    perms = permutations(good_data, 2)

    anchor = []
    positive = []
    negative_one = []
    negative_two = []
    rng = default_rng()
    for p in perms:
        anchor.append(p[0])
        positive.append(p[1])
    negatives = rng.choice(bad_data, size=2 * len(anchor), replace=True).tolist()
    negative_one = negatives[: len(anchor)]
    negative_two = negatives[len(anchor) :]
    assert len(anchor) == len(negative_one)
    assert len(negative_one) == len(negative_two)
    return (anchor, positive, negative_one, negative_two)
