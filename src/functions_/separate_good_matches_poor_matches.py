import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
from datetime import datetime

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)
sys.path.append(parent)




def main():
    # Read 
    #ss_results_df.to_csv(f'./data/team_bravo_semantic_matches_spaceport.{datetime.now().strftime("%m%d")}.csv', index=False)
    # ts_results_df.to_csv(f'./data/team_bravo_siamese1_text_matches_spaceport.{datetime.now().strftime("%m%d")}.csv', index=False)
    # ms_results_df.to_csv(f'./data/team_bravo_siamese2_mix_matches_spaceport.{datetime.now().strftime("%m%d")}.csv', index=False)

    semantic_matches_df  = pd.read_csv('./data/team_bravo_semantic_matches.0719.csv', engine="python")
    semantic_good_matches_df = semantic_matches_df.loc[semantic_matches_df['SCORE'] >= 0.95]
    semantic_poor_matches_df = semantic_matches_df.loc[semantic_matches_df['SCORE'] < 0.95]
    print(f'semantic_good_matches len:{len(semantic_good_matches_df)}')
    print(f'semantic_poor_matches len:{len(semantic_poor_matches_df)}')
    semantic_good_matches_df.to_csv(f'./data/team_bravo_semantic_good_matches{datetime.now().strftime("%m%d")}.csv', index=False)
    semantic_poor_matches_df.to_csv(f'./data/team_bravo_semantic_poor_matches{datetime.now().strftime("%m%d")}.csv', index=False)

    siamese1_matches_df  = pd.read_csv('./data/team_bravo_siamese1_text_matches.0719.csv', engine="python")
    siamese1_good_matches_df =siamese1_matches_df.loc[siamese1_matches_df['SCORE'] >= 0.95]
    siamese1_poor_matches_df = siamese1_matches_df.loc[siamese1_matches_df['SCORE'] < 0.95]
    print(f'siamese1_good_matches len:{len(siamese1_good_matches_df)}')
    print(f'siamese1_poor_matches_df len:{len(siamese1_poor_matches_df)}')
    siamese1_good_matches_df.to_csv(f'./data/team_bravo_siamese1_text_good_matches{datetime.now().strftime("%m%d")}.csv', index=False)
    siamese1_poor_matches_df.to_csv(f'./data/team_bravo_siamese1_text_poor_matches{datetime.now().strftime("%m%d")}.csv', index=False)

    siamese2_mix_matches_df  = pd.read_csv('./data/team_bravo_siamese2_mix_matches.0719.csv', engine="python")
    siamese2_good_matches_df =siamese2_mix_matches_df.loc[siamese2_mix_matches_df['SCORE'] >= 0.95]
    siamese2_poor_matches_df = siamese2_mix_matches_df.loc[siamese2_mix_matches_df['SCORE'] < 0.95]
    print(f'siamese2_good_matches len:{len(siamese2_good_matches_df)}')
    print(f'siamese2_poor_matches_df len:{len(siamese2_poor_matches_df)}')
    siamese2_good_matches_df.to_csv(f'./data/team_bravo_siamese2_mix_good_matches{datetime.now().strftime("%m%d")}.csv', index=False)
    siamese2_poor_matches_df.to_csv(f'./data/team_bravo_siamese2_mix_poor_matches{datetime.now().strftime("%m%d")}.csv', index=False)
    
    

if __name__ == "__main__":
    main()


