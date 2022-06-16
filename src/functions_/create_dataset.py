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
            "ACCOUNT_ID",
        ]
    ]

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

    features_pipeline.set_params(**{"idx_6__column_indexes": [1, 2]})
    notams_data = features_pipeline.fit_transform(notams_df)

    # # # save data to file
    np.save("./data/notams_data", notams_data)

if __name__ == "__main__":
    main()
