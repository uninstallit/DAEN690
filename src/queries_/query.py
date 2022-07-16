import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from datetime import timedelta
from sklearn.neighbors import BallTree

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

from functions_.functions import fromBuffer, inputNoneValues
from functions_.spaceports_dict import get_launch_location, get_spaceports_dict
from pipelines_.pipelines import features_pipeline

tf.keras.utils.disable_interactive_logging()


def get_selected_notams(conn, tfr_rec_id, selected_notams):
    sql = """ SELECT notam_rec_id, e_code, possible_start_date, possible_end_date, location_code, 
              min_alt as MIN_ALT_K, max_alt as MAX_ALT_K, issue_date, account_id FROM notams 
              where notam_rec_id in {list_rec_id} """
    sql = sql.format(list_rec_id=tuple([rec_id for rec_id, s in selected_notams]))
    notams_df = pd.read_sql_query(sql, conn)

    # set scores to notams_df
    scores = []
    for i, row in notams_df.iterrows():
        found = [item for item in selected_notams if item[0] == row["NOTAM_REC_ID"]]
        if found:
            score = found[0][1]
            scores.append(score)
        else:
            scores.append(0)
    notams_df.insert(1, 'SCORE', pd.Series(scores))
    notams_df = notams_df.sort_values('SCORE', ascending=False)

    # denote a row is a TFR
    notams_df["TFR_FLAG"] = notams_df.apply(
        lambda row: 1 if row["NOTAM_REC_ID"] == tfr_rec_id else 0, axis=1
    )
    return notams_df


def add_launch_col(tfr, df):
    df["LAUNCHES_REC_ID"] = tfr["LAUNCHES_REC_ID"]
    new_launch_rec_id_col = df.pop("LAUNCHES_REC_ID")
    df.insert(0, "LAUNCHES_REC_ID", new_launch_rec_id_col)
    df["LAUNCH_DATE"] = tfr["LAUNCH_DATE"]
    new_launch_date_col = df.pop("LAUNCH_DATE")
    df.insert(1, "LAUNCH_DATE", new_launch_date_col)
    return df


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


def semantic_search(query_ids, tfr_embeddings, query_embeddings, top_pick_param):
    ss_selected = []
    for idx, notam_rec_id in enumerate(query_ids):
        similarity_score = tf.keras.metrics.CosineSimilarity()(
            tfr_embeddings, query_embeddings[idx]
        )
        ss_selected.append((notam_rec_id, similarity_score.numpy()))
    ss_selected = sorted(ss_selected, key=lambda x: x[1], reverse=True)[:top_pick_param]
    return ss_selected


def siamese_text_search(query_ids, tfr_embeddings, query_embeddings, top_pick_param):
    base_network = tf.keras.models.load_model(root + "/src/saved_models_/smux_model")
    cosine_similarity = tf.keras.metrics.CosineSimilarity()
    anch_prediction = base_network.predict(tfr_embeddings)

    text_s_selected = []
    for idx, notam_rec_id in enumerate(query_ids):
        _query_embd = np.expand_dims(query_embeddings[idx], axis=(0, -1))
        other_prediction = base_network.predict(_query_embd)
        cosine_similarity.reset_state()
        cosine_similarity.update_state(anch_prediction, other_prediction)
        similarity = cosine_similarity.result().numpy()
        text_s_selected.append((notam_rec_id, similarity))

    text_s_selected = sorted(text_s_selected, key=lambda x: x[1], reverse=True)[
        :top_pick_param
    ]
    print([(rec_id, s) for rec_id, s in text_s_selected])
    return text_s_selected


def siamese_mix_search(
    query_ids, tfr_data, tfr_embeddings, query_data, query_embeddings, top_pick_param
):
    base_network = tf.keras.models.load_model(root + "/src/saved_models_/qsmy_model")
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
        ms_selected.append((notam_rec_id, similarity))

    ms_selected = sorted(ms_selected, key=lambda x: x[1], reverse=True)[:top_pick_param]
    print([(rec_id, s) for rec_id, s in ms_selected])
    return ms_selected


SQL = """ SELECT * FROM notams 
        LEFT JOIN notam_centroids 
        USING(NOTAM_REC_ID) 
        WHERE DATETIME(notams.POSSIBLE_START_DATE) >= DATETIME(\"{start}\", '-1 day') 
        AND DATETIME(notams.POSSIBLE_START_DATE) <= DATETIME(\"{start}\", '+1 day') 
        AND DATETIME(notams.POSSIBLE_END_DATE) >= DATETIME(\"{end}\", '-1 day') 
        AND DATETIME(notams.POSSIBLE_END_DATE) <= DATETIME(\"{end}\", '+1 day'); """


def query(conn, centroid_df, tfr_df, top_pick_param, radius_param, debug_flag):
    print(f"query.....")
    tfr_rec_id = tfr_df.iloc[0]["NOTAM_REC_ID"]  # Only one TFR per launch event
    start_date = tfr_df["POSSIBLE_START_DATE"].tolist()[0]
    end_date = tfr_df["POSSIBLE_END_DATE"].tolist()[0]

    # query notams in the range of  +1 day and -1 day of the TFR date
    sql = SQL.format(start=start_date, end=end_date)
    query_df = pd.read_sql_query(sql, conn)

    query_df = query_df[query_df["LATITUDE"].notna()]
    query_df = query_df[query_df["LONGITUDE"].notna()]

    query_ids = query_df["NOTAM_REC_ID"].tolist()
    if len(query_ids) == 0:
        raise RuntimeError("Notam query is empty!")

    tfr_centroid_df = centroid_df[centroid_df["NOTAM_REC_ID"] == tfr_rec_id]
    if len(tfr_centroid_df) == 0:
        raise RuntimeError("TFR Notam centroid is empty!")

    tfr_df = pd.merge(tfr_df, tfr_centroid_df)

    tfr_x = tfr_df[["LATITUDE", "LONGITUDE"]].to_numpy()
    query_x = query_df[["LATITUDE", "LONGITUDE"]].to_numpy()
    
    # balltree filter
    tree = BallTree(query_x, metric="haversine")
    ind = tree.query_radius(tfr_x, r=radius_param)
    if debug_flag:
        print(f"Balltree filter by radius:{radius_param}nm")
        print(f"ind:{ind}")
        # # print(ind)  # indices of 3 closest neighbors
        # # print(dist)  # distances to 3 closest neighbors
        print("BallTree Notams count within range: ", len(ind[0]))

    # Model search
    query_df = query_df.iloc[ind[0]]
    query_df = inputNoneValues(query_df)  # # fill out none values for selected features
    set_param_features_pipeline()

    tfr_data = features_pipeline.fit_transform(tfr_df)
    tfr_embeddings = fromBuffer(tfr_data[:, 8])
    tfr_data = tfr_data[:, 4:-1].astype("float32")

    query_data = features_pipeline.fit_transform(query_df)
    query_ids = query_data[:, 0]  # get notam_rec_id
    query_embeddings = fromBuffer(query_data[:, 8])
    query_data = query_data[:, 4:-1].astype(
        "float32"
    ) 
    ss_selected = semantic_search(
        query_ids, tfr_embeddings, query_embeddings, top_pick_param
    )
    
    ts_selected = siamese_text_search(
        query_ids, tfr_embeddings, query_embeddings, top_pick_param
    )
    ms_selected = siamese_mix_search(
        query_ids,
        tfr_data,
        tfr_embeddings,
        query_data,
        query_embeddings,
        top_pick_param,
    )

    if debug_flag:
        print("TFR Notam:")
        print(
            tfr_df[
                ["NOTAM_REC_ID", "E_CODE", "POSSIBLE_START_DATE", "POSSIBLE_END_DATE"]
            ]
        )

        print("\nSemantic Search Model Score")
        for notam_rec_id, similarity in ss_selected:
            print(f"ss Notam id: {notam_rec_id} - cos score: {similarity}")
        print([i for i, s in ss_selected])

        print("\nText Model Score")
        for notam_rec_id, similarity in ts_selected:
            print(f"ts Notam id: {notam_rec_id} - cos score: {similarity}")
        print([i for i, s in ts_selected])

        print("\nMix Model Score")
        for notam_rec_id, similarity in ms_selected:
            print(f"ms Notam id: {notam_rec_id} - cos score: {similarity}")
        print([i for i, s in ms_selected])

    ss_results_df = get_selected_notams(conn, tfr_rec_id, ss_selected)
    ts_results_df = get_selected_notams(conn, tfr_rec_id, ts_selected)
    ms_results_df = get_selected_notams(conn, tfr_rec_id, ms_selected)
    return (ss_results_df, ts_results_df, ms_results_df)


def nlp_match(
    conn,
    centroid_df,
    input_tfrs_df,
    notams_df,
    launch_ids_param,
    top_pick_param,
    radius_param,
    debug_flag,
):
    cols = [
        "LAUNCHES_REC_ID",
        "NOTAM_REC_ID",
        "SCORE",
        "LAUNCH_DATE",
        "POSSIBLE_START_DATE",
        "POSSIBLE_END_DATE",
        "E_CODE",
        "LOCATION_CODE",
        "ACCOUNT_ID",
        "TFR_FLAG",
        "MIN_ALT_K",
        "MAX_ALT_K",
    ]

    # use launch_ids_param list otherwise the whole input list  input_tfrs_df input 
    input_tfrs_df = (
        input_tfrs_df.loc[input_tfrs_df["LAUNCHES_REC_ID"].isin(launch_ids_param)]
        if len(launch_ids_param)
        else input_tfrs_df
    )
    results = {}  # { launch_rec_id: [tfr, ss, ts, ms]}
    for idx, tfr_info in input_tfrs_df.iterrows():
        launch_rec_id = tfr_info["LAUNCHES_REC_ID"]
        tfr_notam_rec_id = tfr_info["NOTAM_REC_ID"]
        tfr_notam_df = notams_df[notams_df["NOTAM_REC_ID"] == tfr_notam_rec_id]
        result = query(
            conn, centroid_df, tfr_notam_df, top_pick_param, radius_param, debug_flag
        )
        (ss_matches_df, ts_matches_df, ms_matches_df) = result
        # adding column, reindex results
        ss_matches_df = add_launch_col(tfr_info, ss_matches_df)
        ts_matches_df = add_launch_col(tfr_info, ts_matches_df)
        ms_matches_df = add_launch_col(tfr_info, ms_matches_df)
        ss_matches_df = ss_matches_df.reindex(cols, axis=1)
        ts_matches_df = ts_matches_df.reindex(cols, axis=1)
        ms_matches_df = ms_matches_df.reindex(cols, axis=1)
        results[launch_rec_id] = (tfr_info, ss_matches_df, ts_matches_df, ms_matches_df)

    return results

def predict_related_notams(launch_ids_param,top_pick_param, balltree_radius_param, debug_flag ):
    conn = sqlite3.Connection(root + "/data/svo_db_20201027.db")

    sql = """ SELECT * FROM notams """
    notams_df = pd.read_sql_query(sql, conn)
    sql = """ SELECT * FROM notam_centroids """
    centroid_df = pd.read_sql_query(sql, conn)
    spaceports_dict = get_spaceports_dict(conn)
    input_tfrs_df = pd.read_csv(root + "/data/tfr_notams.0709.csv", engine="python")

    start = time.time()
    results = nlp_match(
        conn,
        centroid_df,
        input_tfrs_df,
        notams_df,
        launch_ids_param,
        top_pick_param,
        balltree_radius_param,
        debug_flag,
    )
    end = time.time()
    print(f"Elapse time: {str(timedelta(seconds=end-start))}")
   
    ##### print and write results to csv, db
    display_cols = [
        "LAUNCHES_REC_ID",
        "NOTAM_REC_ID",
        "SCORE",
        "POSSIBLE_START_DATE",
        "POSSIBLE_END_DATE",
        "E_CODE",
        "LOCATION_CODE",
        "ACCOUNT_ID",
        "TFR_FLAG",
        "MAX_ALT_K",
    ]
    ss_results = []
    tx_results = []
    ms_results = []
    for launch_rec_id, result in results.items():
        (tfr, ss_matches_df, ts_matches_df, ms_matches_df) = result
        ss_results.append(ss_matches_df)
        tx_results.append(ts_matches_df)
        ms_results.append(ms_matches_df)
        launch_rec_id, launch_date, spaceport_rec_id = (
            tfr["LAUNCHES_REC_ID"],
            tfr["LAUNCH_DATE"],
            tfr["SPACEPORT_REC_ID"],
        )
        launch_location, launch_state_location = get_launch_location(
            spaceports_dict, spaceport_rec_id
        )

        print(f"\n---Semantic Search Model")
        print(
            f"Launch_id: {launch_rec_id} date: {launch_date} {launch_location} {launch_state_location}"
        )
        print(
            f"TFR {tfr['NOTAM_REC_ID'], tfr['POSSIBLE_START_DATE'], tfr['POSSIBLE_END_DATE'], '%.100s...' % tfr['E_CODE'] }"
        )
        print(f"Related NOTAMs:")
        print(ss_matches_df[display_cols])

        print(f"\n---Siamese Text Model")
        print(
            f"Launch_id: {launch_rec_id} date: {launch_date} {launch_location} {launch_state_location}"
        )
        print(
            f"TFR {tfr['NOTAM_REC_ID'], tfr['POSSIBLE_START_DATE'], tfr['POSSIBLE_END_DATE'], '%.100s...' % tfr['E_CODE']}"
        )
        print(f"Related NOTAMs:")
        print(ts_matches_df[display_cols])

        print(f"\n---Siamese Mix Model")
        print(
            f"Launch_id: {launch_rec_id} date: {launch_date} {launch_location} {launch_state_location}"
        )
        print(
            f"TFR {tfr['NOTAM_REC_ID'], tfr['POSSIBLE_START_DATE'], tfr['POSSIBLE_END_DATE'], '%.100s...' % tfr['E_CODE'] }"
        )
        print(f"Related NOTAMs:")
        print(ms_matches_df[display_cols])

    # write out the results
    ss_results_df = pd.concat(ss_results)
    ts_results_df = pd.concat(tx_results)
    ms_results_df = pd.concat(ms_results)

    ss_results_df.to_csv(f"./data/team_bravo_semantic_matches.csv", index=False)
    ts_results_df.to_csv(f"./data/team_bravo_siamese1_text_matches.csv", index=False)
    ms_results_df.to_csv(f"./data/team_bravo_siamese2_mix_matches.csv", index=False)

    ss_results_df.to_sql(
        "team_bravo_semantic_matches", conn, if_exists="replace", index=False
    )
    ts_results_df.to_sql(
        "team_bravo_siamese_txt_matches", conn, if_exists="replace", index=False
    )
    ms_results_df.to_sql(
        "team_bravo_siamese_mix_matches", conn, if_exists="replace", index=False
    )

    # TODO take care (sanjiv's step3) - need to justify the low score notam list if they have the start/end date and same account_id with others
    # then they are related notams

    conn.close()

def main():
    pass
    


if __name__ == "__main__":
    main()
