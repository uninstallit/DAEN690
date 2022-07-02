import sqlite3
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import tensorflow as tf


# balltree source
# source: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)
sys.path.append(root)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from functions_.functions import fromBuffer, inputNoneValues, get_matches_index_dict
from pipelines_.pipelines import features_pipeline

tf.get_logger().setLevel("ERROR")

def print_results(conn, launch_rec_id, tfr_df, selected):   
    sql = """ SELECT notam_rec_id, e_code, possible_start_date, possible_end_date FROM notams where notam_rec_id in {list_rec_id}
          """.format(list_rec_id=tuple([rec_id for rec_id, s in selected]))
    notams_df = pd.read_sql_query(sql, conn)
    
    related_notams = []
    for rec_id, _ in  selected:
       related_notams.append(notams_df[notams_df['NOTAM_REC_ID'] ==  rec_id])
        
    related_notams_df = pd.concat(related_notams)
    print(f"TFR {tfr_df[['NOTAM_REC_ID', 'E_CODE', 'POSSIBLE_START_DATE', 'POSSIBLE_END_DATE']]}")
    print('Related NOTAMs:')
    print(related_notams_df)


def set_param_features_pipeline():
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
    
def main():

    conn = sqlite3.Connection(root + "/data/svo_db_20201027.db")

    sql = """ SELECT * FROM human_matches """
    matches_df = pd.read_sql_query(sql, conn)

    sql = """ SELECT * FROM notams """
    notams_df = pd.read_sql_query(sql, conn)

    sql = """ SELECT * FROM notam_centroids """
    centroid_df = pd.read_sql_query(sql, conn)

    launch_id = 284 # 2016-12-18 19:13:00
    matches_dict = get_matches_index_dict(matches_df)
    matches = matches_dict[launch_id]
    notam_id = matches[2] # let pick a  notam from the list
    # TODO to find TFR - replace with your own not from the matches

    notam_df = notams_df[notams_df["NOTAM_REC_ID"] == notam_id]
    print('TFR Notam:')
    print(notam_df[['NOTAM_REC_ID', 'E_CODE', 'POSSIBLE_START_DATE', 'POSSIBLE_END_DATE']])

    start_date = notam_df["POSSIBLE_START_DATE"].tolist()[0]  
    end_date = notam_df["POSSIBLE_END_DATE"].tolist()[0]  

    sql = """ SELECT * FROM notams 
        LEFT JOIN notam_centroids 
        USING(NOTAM_REC_ID) 
        WHERE DATETIME(notams.POSSIBLE_START_DATE) >= DATETIME(\"{start}\", '-1 day') 
        AND DATETIME(notams.POSSIBLE_START_DATE) <= DATETIME(\"{start}\", '+1 day') 
        AND DATETIME(notams.POSSIBLE_END_DATE) >= DATETIME(\"{end}\", '-1 day') 
        AND DATETIME(notams.POSSIBLE_END_DATE) <= DATETIME(\"{end}\", '+1 day'); """.format(
        start=start_date, end=end_date
    )
    query_df = pd.read_sql_query(sql, conn)

    print('query_df:', query_df[['NOTAM_REC_ID', 'E_CODE', 'POSSIBLE_START_DATE', 'POSSIBLE_END_DATE']])
    
    query_df = query_df[query_df["LATITUDE"].notna()]
    query_df = query_df[query_df["LONGITUDE"].notna()]

    query_ids = query_df["NOTAM_REC_ID"].tolist()
    if len(query_ids) == 0:
        raise RuntimeError("Notam query is empty!")

    notam_centroid_df = centroid_df[centroid_df["NOTAM_REC_ID"] == notam_id]
    notam_df = pd.merge(notam_df, notam_centroid_df)

    notam_x = notam_df[["LATITUDE", "LONGITUDE"]].to_numpy()
    query_x = query_df[["LATITUDE", "LONGITUDE"]].to_numpy()

    # balltree filter
    tree = BallTree(query_x, metric="haversine")
    K = 100 # TODO percent how to choose K
    Radius = 30
    if len(notam_x) > K:
        print(f'Balltree filter by k:{K}') # KNN
        dist, ind = tree.query(notam_x, k=K)
    else: 
        print(f'Balltree filter by radius:{Radius}nm')
        ind = tree.query_radius(notam_x, r=Radius)

    print(f'ind:{ind}')

    # # print(ind)  # indices of 3 closest neighbors
    # # print(dist)  # distances to 3 closest neighbors
    print("BallTree Notams count within range: ", len(ind[0]))
    query_df = query_df.iloc[ind[0]]
   
    # # fill out none values for selected features
    notams_df = inputNoneValues(notams_df)
    query_df = inputNoneValues(query_df)

    set_param_features_pipeline()
    
    notam_data = features_pipeline.fit_transform(notam_df)
    notam_embeddings = fromBuffer(notam_data[:, 8])
    notam_data = notam_data[:, 2:-1] .astype("float32") 
    
    query_data = features_pipeline.fit_transform(query_df)
    query_ids = query_data[:, 0] # get notam_rec_id
    query_embeddings = fromBuffer(query_data[:, 8])
    query_data = query_data[:, 2:-1].astype("float32") 

    # semantic search filter - select the top 100
    ss_selected = []
    for idx, notam_rec_id in enumerate(query_ids):
        similarity_score = tf.keras.metrics.CosineSimilarity()(notam_embeddings, query_embeddings[idx])
        ss_selected.append((notam_rec_id, similarity_score.numpy()))
    ss_selected = sorted(ss_selected, key=lambda x: x[1], reverse=True)[:10]
  
    
    # base_network
    base_network = tf.keras.models.load_model(root + "/src/saved_models_/sm1_model")
    cosine_similarity = tf.keras.metrics.CosineSimilarity()
    anch_prediction = base_network.predict([notam_data, notam_embeddings])
   
    ms_selected = []
    for idx, notam_rec_id in enumerate(query_ids):
        _query_data = np.expand_dims(query_data[idx], 0)
        _query_embd = np.expand_dims(query_embeddings[idx], axis=(0, -1))
        other_prediction = base_network.predict([_query_data, _query_embd])
        cosine_similarity.reset_state()
        cosine_similarity.update_state(anch_prediction, other_prediction)
        similarity = cosine_similarity.result().numpy()
        ms_selected.append((query_ids[idx], similarity))

    ms_selected = sorted(ms_selected, key=lambda x: x[1], reverse=True)[:10] 
    print([(rec_id, s) for rec_id, s in ms_selected])
   
    print("\nSemantic Search")
    for notam_rec_id, similarity in ss_selected:
        print(f"ss Notam id: {notam_rec_id} - cos score: {similarity}")

    print("\nModel Score")
    for notam_rec_id, similarity in ms_selected:
        print(f"ms Notam id: {notam_rec_id} - cos score: {similarity}")

    print("\nSemantic Search")
    print([i for i, s in ss_selected])
    print("\nModel Score")
    print([i for i, s in ms_selected])

    print(f'\n---Semantic Search Results launch_id: {launch_id}')
    print_results(conn, launch_id, notam_df, ss_selected)

    print(f'\n---Model Results launch_id: {launch_id}')
    print_results(conn, launch_id, notam_df, ms_selected)

    conn.close()
    

if __name__ == "__main__":
    main()
