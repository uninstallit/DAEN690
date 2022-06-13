import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import random
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.preprocessing import LabelEncoder

# source: https://keras.io/examples/vision/siamese_network/


import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(root)

from pipelines_.pipelines import features_pipeline


def main():

    # retrieve data from database
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
            # "ACCOUNT_ID",
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
            "add_delta_time_feature_idx_4__column_indexes": [1, 2],
            "add_text_embedder_feature_idx_5__column_index": 0,
        }
    )
    notams_data = features_pipeline.fit_transform(notams_df)

    # save data to file
    np.save("./data/output/notams_data", notams_data)


if __name__ == "__main__":
    main()
