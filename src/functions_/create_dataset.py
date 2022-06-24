import sqlite3
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from functions_.functions import get_triplet_index_dict
from pipelines_.pipelines import features_pipeline


def main():

    # retrieve data from database
    conn = sqlite3.Connection(root + "/data/svo_db_20201027.db")
    sql = """ SELECT * FROM notams"""
    notams_df = pd.read_sql_query(sql, conn)
    conn.close()

    # fill out none values for selected features
    notams_df["TEXT"] = notams_df["TEXT"].fillna("UNKNOWN")
    notams_df["ISSUE_DATE"] = notams_df["ISSUE_DATE"].fillna(
        notams_df["ISSUE_DATE"].value_counts().idxmax()
    )
    notams_df["POSSIBLE_START_DATE"] = notams_df["POSSIBLE_START_DATE"].fillna(
        notams_df["POSSIBLE_START_DATE"].value_counts().idxmax()
    )
    notams_df["CLASSIFICATION"] = notams_df["CLASSIFICATION"].fillna("UNKNOWN")
    notams_df["LOCATION_CODE"] = notams_df["LOCATION_CODE"].fillna("UNKNOWN")
    notams_df["ACCOUNT_ID"] = notams_df["ACCOUNT_ID"].fillna("UNKNOWN")

    # create triplet indexes
    (anchor_index, positive_index, negative_index) = get_triplet_index_dict()
    anchor_df = notams_df[notams_df["NOTAM_REC_ID"].isin(anchor_index)]
    positive_df = notams_df[notams_df["NOTAM_REC_ID"].isin(positive_index)]
    negative_df = notams_df[notams_df["NOTAM_REC_ID"].isin(negative_index)]

    # run features pipeline
    features_pipeline.set_params(**{"idx_7__column_indexes": [2, 3]})
    features_pipeline.set_params(**{"idx_8__column_index": 1})

    # populate normalization params from population
    notams_data = features_pipeline.fit_transform(notams_df)

    features_pipeline.set_params(
        **{"preprocess__columns__idx_2__start_date_normalize__isInference": True}
    )
    features_pipeline.set_params(
        **{"preprocess__columns__idx_3__issue_date_normalize__isInference": True}
    )
    features_pipeline.set_params(
        **{"preprocess__columns__idx_4__location_code_normalize__isInference": True}
    )
    features_pipeline.set_params(
        **{"preprocess__columns__idx_5__classification_normalize__isInference": True}
    )
    features_pipeline.set_params(
        **{"preprocess__columns__idx_6__account_id_normalize__isInference": True}
    )
    features_pipeline.set_params(**{"idx_8__skip": False})

    anchor_data = features_pipeline.fit_transform(anchor_df)
    positive_data = features_pipeline.fit_transform(positive_df)
    negative_data = features_pipeline.fit_transform(negative_df)

    # ensure datasets have same length
    data_lengths = np.array(
        [anchor_data.shape[0], positive_data.shape[0], negative_data.shape[0]]
    )
    min_length = data_lengths.min()
    
    anchor_data = anchor_data[:min_length]
    positive_data = positive_data[:min_length]
    negative_data = negative_data[:min_length]

    # print(anchor_data.shape)
    # print(positive_data.shape)
    # print(negative_data.shape)

    # save data to file
    np.save("./data/anchor_data", anchor_data)
    np.save("./data/positive_data", positive_data)
    np.save("./data/negative_data", negative_data)


if __name__ == "__main__":
    main()
