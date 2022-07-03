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

def pack_results(conn, launch_rec_id, tfr_df, related_notams):   
    sql = """ SELECT notam_rec_id, e_code, possible_start_date, possible_end_date, location_code, min_alt as MIN_ALT_K, max_alt as MAX_ALT_K, 
              issue_date, account_id FROM notams where notam_rec_id in {list_rec_id} """.format(list_rec_id=tuple([rec_id for rec_id, s in related_notams]))
    notams_df = pd.read_sql_query(sql, conn)
    
    notams = []
    for rec_id, _ in  related_notams:
       notams.append(notams_df[notams_df['NOTAM_REC_ID'] ==  rec_id])
        
    notams_df = pd.concat(notams)
    print(f"TFR {tfr_df[['NOTAM_REC_ID', 'E_CODE', 'POSSIBLE_START_DATE', 'POSSIBLE_END_DATE']]}")
    print('Related NOTAMs:')
    print(notams_df)

    return notams_df


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

    
def query(conn, spaceports_dict, launch, tfr_df):
    launch_rec_id, launch_date = launch['LAUNCHES_REC_ID'], launch['LAUNCH_DATE']
    launch_spaceport_rec_id = int(launch['SPACEPORT_REC_ID']) if launch['SPACEPORT_REC_ID'] else np.NaN
    launch_location = spaceports_dict[launch_spaceport_rec_id]['LOCATION_1'] if launch_spaceport_rec_id != np.NaN else ""
    
    tfr_rec_id = tfr_df.iloc[0]['NOTAM_REC_ID'] # Only TFR per launch event

    sql = """ SELECT * FROM notam_centroids """
    centroid_df = pd.read_sql_query(sql, conn)

    print('TFR Notam:')
    print(tfr_df[['NOTAM_REC_ID', 'E_CODE', 'POSSIBLE_START_DATE', 'POSSIBLE_END_DATE']])

    start_date = tfr_df["POSSIBLE_START_DATE"].tolist()[0]  
    end_date = tfr_df["POSSIBLE_END_DATE"].tolist()[0]  

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

    query_df = query_df[query_df["LATITUDE"].notna()]
    query_df = query_df[query_df["LONGITUDE"].notna()]

    query_ids = query_df["NOTAM_REC_ID"].tolist()
    if len(query_ids) == 0:
        raise RuntimeError("Notam query is empty!")

    tfr_centroid_df = centroid_df[centroid_df["NOTAM_REC_ID"] == tfr_rec_id]
    tfr_df = pd.merge(tfr_df, tfr_centroid_df)

    tfr_x = tfr_df[["LATITUDE", "LONGITUDE"]].to_numpy()
    query_x = query_df[["LATITUDE", "LONGITUDE"]].to_numpy()

    # balltree filter
    tree = BallTree(query_x, metric="haversine")
    Radius = 50  # Tune param
    
    print(f'Balltree filter by radius:{Radius}nm')
    ind = tree.query_radius(tfr_x, r=Radius)
    print(f'ind:{ind}')
    
    # # print(ind)  # indices of 3 closest neighbors
    # # print(dist)  # distances to 3 closest neighbors
    print("BallTree Notams count within range: ", len(ind[0]))
    query_df = query_df.iloc[ind[0]]
   
    # # fill out none values for selected features
    query_df = inputNoneValues(query_df)

    set_param_features_pipeline()
    
    tfr_data = features_pipeline.fit_transform(tfr_df)
    tfr_embeddings = fromBuffer(tfr_data[:, 8])
    tfr_data = tfr_data[:, 2:-1] .astype("float32") 
    
    query_data = features_pipeline.fit_transform(query_df)
    query_ids = query_data[:, 0] # get notam_rec_id
    query_embeddings = fromBuffer(query_data[:, 8])
    query_data = query_data[:, 2:-1].astype("float32") 

    top_picks = 10
    # semantic search filter - select the top 100
    ss_selected = []
    for idx, notam_rec_id in enumerate(query_ids):
        similarity_score = tf.keras.metrics.CosineSimilarity()(tfr_embeddings, query_embeddings[idx])
        ss_selected.append((notam_rec_id, similarity_score.numpy()))
    ss_selected = sorted(ss_selected, key=lambda x: x[1], reverse=True)[:top_picks]
  
    # base_network
    base_network = tf.keras.models.load_model(root + "/src/saved_models_/sm1_model")
    cosine_similarity = tf.keras.metrics.CosineSimilarity()
    anch_prediction = base_network.predict([tfr_data, tfr_embeddings])
   
    ms_selected = []
    for idx, notam_rec_id in enumerate(query_ids):
        _query_data = np.expand_dims(query_data[idx], 0)
        _query_embd = np.expand_dims(query_embeddings[idx], axis=(0, -1))
        other_prediction = base_network.predict([_query_data, _query_embd])
        cosine_similarity.reset_state()
        cosine_similarity.update_state(anch_prediction, other_prediction)
        similarity = cosine_similarity.result().numpy()
        ms_selected.append((query_ids[idx], similarity))

    ms_selected = sorted(ms_selected, key=lambda x: x[1], reverse=True)[:top_picks] 
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

    
    print(f'\n---Semantic Search Results launch_id: {launch_rec_id} date: {launch_date} {launch_location}')
    ss_results  = pack_results(conn, launch_rec_id, tfr_df, ss_selected)

    print(f'\n---Model Results launch_id: {launch_rec_id} date: {launch_date} {launch_location}')
    ms_results = pack_results(conn, launch_rec_id, tfr_df, ms_selected)
    
    return (launch_rec_id, ss_results, ms_results)
    
    
def main():
    
    conn = sqlite3.Connection(root + "/data/svo_db_20201027.db")

    sql = """ SELECT * FROM notams """
    notams_df = pd.read_sql_query(sql, conn)

    sql = """ SELECT * from launches """
    launches_df = pd.read_sql_query(sql, conn)
    
    sql = """ SELECT * from spaceports """
    spaceports_df = pd.read_sql_query(sql, conn)
    spaceports_dict = {}
    for _, space in spaceports_df.iterrows():
        spaceports_dict[space['SPACEPORT_REC_ID']] = space

    tfr_notams_df = pd.read_csv(root + "/data/tfr_notams.csv" , engine="python" )

    ss_results = []
    ms_results = []
    for idx, launch in launches_df.iterrows():
        launch_rec_id = launch['LAUNCHES_REC_ID']
        matched_tfr_df = tfr_notams_df[tfr_notams_df['LAUNCHES_REC_ID'] == launch_rec_id]
        if len(matched_tfr_df):
            tfr_rec_id = matched_tfr_df.iloc[0]['NOTAM_REC_ID']
            tfr_df = notams_df[notams_df["NOTAM_REC_ID"] == tfr_rec_id]
            if launch_rec_id == 391: ##### TODO remove test
               (launch_rec_id, ss_matches, ms_matches) = query(conn, spaceports_dict, launch, tfr_df)
               ss_matches.insert(0, "LAUNCHES_REC_ID", ss_matches.apply(lambda row : launch_rec_id, axis = 1))
               ss_matches.insert(1, "LAUNCH_DATE", ss_matches.apply(lambda row : launch['LAUNCH_DATE'], axis = 1))
               ss_results.append(ss_matches)
               ms_matches.insert(0, "LAUNCHES_REC_ID", ms_matches.apply(lambda row : launch_rec_id, axis = 1))
               ms_matches.insert(1, "LAUNCH_DATE", ms_matches.apply(lambda row : launch['LAUNCH_DATE'], axis = 1))
               ms_results.append(ms_matches)
    
    ss_results_df = pd.concat(ss_results)
    ms_results_df = pd.concat(ms_results)

    # TODO take care (sanjiv's step3) - need to do a final filter to get NOTAMs that the model was not able to select since the unselected list may contain the same account_id with others 

    cols=['LAUNCHES_REC_ID','NOTAM_REC_ID','MIN_ALT_K','MAX_ALT_K',
          'LAUNCH_DATE','ISSUE_DATE', 'POSSIBLE_START_DATE','POSSIBLE_END_DATE',
          'E_CODE','LOCATION_CODE','ACCOUNT_ID']
    ss_results_df = ss_results_df.reindex(cols, axis=1)
    ms_results_df = ms_results_df.reindex(cols, axis=1)
 
    ss_results_df.to_csv(f'./data/team_bravo_semantic_matches.csv', index=False)
    ms_results_df.to_csv(f'./data/team_bravo_siamese_matches.csv', index=False)

    ss_results_df.to_sql('team_bravo_semantic_matches', conn, if_exists='replace', index = False)
    ms_results_df.to_sql('team_bravo_siamese_matches', conn, if_exists='replace', index = False)
    
    conn.close()


if __name__ == "__main__":
    main()
