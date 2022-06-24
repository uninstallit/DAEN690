import sqlite3
import numpy as np
import pandas as pd

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(root)

from functions_.functions import fromBuffer, inputNoneValues
from pipelines_.pipelines import features_pipeline


def main():

    conn = sqlite3.Connection(root + "/data/svo_db_20201027.db")

    sql = """ SELECT * FROM notams"""
    notams_df = pd.read_sql_query(sql, conn)

    sql = """ SELECT * FROM notams WHERE DATETIME(POSSIBLE_START_DATE) >= '2016-05-04 03:16:00' and DATETIME(POSSIBLE_START_DATE) >= '2016-05-10 03:16:00' and DATETIME(notams.POSSIBLE_END_DATE) <= '2016-06-06 03:16:00' """
    query_df = pd.read_sql_query(sql, conn)
    conn.close()

    query_ids = query_df["NOTAM_REC_ID"].tolist()

    print("Total nr. queries: ", len(query_ids))

    # fill out none values for selected features
    notams_df = inputNoneValues(notams_df)

    # [21915, 21916, 21917]
    notam_df = notams_df[notams_df["NOTAM_REC_ID"] == 21915]

    features_pipeline.set_params(**{"idx_7__column_indexes": [2, 3]})

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
    features_pipeline.set_params(**{"idx_7__column_indexes": [2, 3]})
    features_pipeline.set_params(**{"idx_8__column_index": 1})
    features_pipeline.set_params(**{"idx_8__skip": False})

    notam_data = features_pipeline.fit_transform(notam_df)
    notam_embeddings = fromBuffer(notam_data[:, 8])
    notam_data = notam_data[:, 2:-1].astype("float32")

    query_data = features_pipeline.fit_transform(query_df)
    query_embeddings = fromBuffer(query_data[:, 8])
    query_data = query_data[:, 2:-1].astype("float32")

    base_network = tf.keras.models.load_model(root + "/src/saved_models_/test_model")

    for idx in range(0, len(query_df.index)):
        _query_data = np.expand_dims(query_data[idx], 0)
        _query_embd = np.expand_dims(query_embeddings[idx], axis=(0, -1))
        anch_prediction = base_network.predict([notam_data, notam_embeddings])
        other_prediction = base_network.predict([_query_data, _query_embd])
        cosine_similarity = tf.keras.metrics.CosineSimilarity()
        positive_similarity = cosine_similarity(anch_prediction, other_prediction)
        print(
            "Cosine similarity for rec id {} = {}".format(
                query_ids[idx], positive_similarity.numpy()
            )
        )
    print("Total nr. queries: ", len(query_ids))


if __name__ == "__main__":
    main()
