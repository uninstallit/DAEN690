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
  
    semantic_matches_df  = pd.read_csv('./data/team_bravo_semantic_search_matches.0729.csv', engine="python")
    semantic_good_matches_df = semantic_matches_df.loc[semantic_matches_df['SCORE'] >= 0.95]
    semantic_poor_matches_df = semantic_matches_df.loc[semantic_matches_df['SCORE'] < 0.95]
    print(f'semantic_good_matches len:{len(semantic_good_matches_df)}')
    print(f'semantic_poor_matches len:{len(semantic_poor_matches_df)}')
    semantic_good_matches_df.to_csv(f'./data/team_bravo_semantic_good_matches.{datetime.now().strftime("%m%d")}.csv', index=False)
    semantic_poor_matches_df.to_csv(f'./data/team_bravo_semantic_poor_matches.{datetime.now().strftime("%m%d")}.csv', index=False)

    text_matches_df  = pd.read_csv('./data/team_bravo_text_matches.0729.csv', engine="python")
    text_good_matches_df =text_matches_df.loc[text_matches_df['SCORE'] >= 0.95]
    text_poor_matches_df = text_matches_df.loc[text_matches_df['SCORE'] < 0.95]
    print(f'tex_good_matches len:{len(text_good_matches_df)}')
    print(f'tex_poor_matches_df len:{len(text_poor_matches_df)}')
    text_good_matches_df.to_csv(f'./data/team_bravo_text_good_matches.{datetime.now().strftime("%m%d")}.csv', index=False)
    text_poor_matches_df.to_csv(f'./data/team_bravo_text_poor_matches.{datetime.now().strftime("%m%d")}.csv', index=False)

    mix_matches_df  = pd.read_csv('./data/team_bravo_mix_matches.0729.csv', engine="python")
    mix_good_matches_df =mix_matches_df.loc[mix_matches_df['SCORE'] >= 0.95]
    mix_poor_matches_df = mix_matches_df.loc[mix_matches_df['SCORE'] < 0.95]
    print(f'mix_good_matches len:{len(mix_good_matches_df)}')
    print(f'mix_poor_matches_df len:{len(mix_poor_matches_df)}')
    mix_good_matches_df.to_csv(f'./data/team_bravo_mix_good_matches.{datetime.now().strftime("%m%d")}.csv', index=False)
    mix_poor_matches_df.to_csv(f'./data/team_bravo_mix_poor_matches.{datetime.now().strftime("%m%d")}.csv', index=False)
    
    

if __name__ == "__main__":
    main()


