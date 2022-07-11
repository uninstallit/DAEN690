import sqlite3
import numpy as np
import pandas as pd
import json
import copy
import random
import matplotlib.pyplot as plt
from numpy import genfromtxt
from itertools import permutations

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


def get_quadruplet_index():
    conn = sqlite3.Connection(root + "/data/svo_db_20201027.db")
    sql = """ SELECT * FROM human_matches"""
    matches_df = pd.read_sql_query(sql, conn)
    conn.close()

    bad_df = pd.read_csv(
        root + "/data/negative_unique_notams.0709.csv", sep=","
    )
    bad_data = np.squeeze(bad_df[['NOTAM_REC_ID']].to_numpy())

    anchor_list = []
    positive_list = []
    negative_one_list = []
    negative_two_list = []
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
            anchor_list.append(anchor)
            positive_list.append(positive)
            negative_one_list.append(negative)

    rng = np.random.default_rng()
    negative_two_list = rng.choice(bad_data, size=len(anchor_list), replace=False).tolist()
    
    assert len(anchor_list) == len(positive_list)
    assert len(anchor_list) == len(negative_one_list)
    assert len(anchor_list) == len(negative_two_list)

    # this returns NOTAM_REC_ID - not dataframe index
    return (anchor_list, positive_list, negative_one_list, negative_two_list)


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
    good_df = pd.read_csv(
        root + "/data/possitive_unique_notams.0709.csv", sep=","
    )
    bad_df = pd.read_csv(
        root + "/data/negative_unique_notams.0709.csv", sep=","
    )
    good_data = np.squeeze(good_df[['NOTAM_REC_ID']].to_numpy())
    bad_data = np.squeeze(bad_df[['NOTAM_REC_ID']].to_numpy())
    perms = permutations(good_data, 2)

    anchor_index = []
    positive_index = []
    negative_index = []
    rng = np.random.default_rng()
    for p in perms:
        anchor_index.append(p[0])
        positive_index.append(p[1])
    negative_index = rng.choice(bad_data, size=len(anchor_index), replace=True).tolist()
    return (anchor_index, positive_index, negative_index)

