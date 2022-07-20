# info: https://www.theairlinepilots.com/flightplanningforairlinepilots/notamdecode.php
import sys
import os
import pandas as pd
from datetime import datetime

current = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(current)
sys.path.append(dir)

from functions_.create_tfr_train_dataset import create_tfr_train_dataset
from queries_.query  import predict_related_notams

def main():
    # options flags
    CREATE_TRAIN_SET_FLAG = False
    PREDICT_NOTAMS_FLAG = True

    if CREATE_TRAIN_SET_FLAG:
        create_tfr_train_dataset()

    available_launch_ids = pd.read_csv(dir + "/data/tfr_notams.0709.csv", engine="python")
    print(f"Available launch_rec_ids have TFR. Total:{len(available_launch_ids)}")
    available_launch_ids = [row['LAUNCHES_REC_ID'] for i, row in available_launch_ids.iterrows()]
    print(available_launch_ids)

    if PREDICT_NOTAMS_FLAG:
        # specify launch_rec_id you wish to run launch. Empty launch_ids_param array will run all 103 launches
        launch_ids_param = [] # this is a default.  Will run all available launches with tfr
        #launch_ids_param = [391]
        top_pick_param = 12
        balltree_radius_in_miles_param = 200 
        debug_flag = False # turn off console print debug

        (ss_results_df, ts_results_df,ms_results_df) = predict_related_notams(launch_ids_param, top_pick_param, balltree_radius_in_miles_param, debug_flag)

        #####  specify the port output in the file names 
        # ss_results_df.to_csv(f"./data/23_CA_semantic_matches_spaceport_23_Lompocs_CA.csv", index=False)
        # ts_results_df.to_csv(f"./data/23_CA_siamese1_text_matches_spaceport_23_Lompocs_CA.csv", index=False)
        # ms_results_df.to_csv(f"./data/23_CA_siamese2_mix_matches_spaceport_23_Lompocs_CA.csv", index=False)

        ss_results_df.to_csv(f'./data/semantic_matches_spaceport.{datetime.now().strftime("%m%d")}.csv', index=False)
        ts_results_df.to_csv(f'./data/siamese1_text_matches_spaceport.{datetime.now().strftime("%m%d")}.csv', index=False)
        ms_results_df.to_csv(f'./data/siamese2_mix_matches_spaceport.{datetime.now().strftime("%m%d")}.csv', index=False)

    

if __name__ == "__main__":
    main()
