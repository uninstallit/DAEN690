import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
import datetime

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)
sys.path.append(parent)


def plot_launches_by_spaceport(conn):
    sql = """ select S.spaceport_rec_id, S.location_1, S.location_2 from launches as L, spaceports as S 
            where L.launches_rec_id in(248, 254, 257, 258, 260, 261, 263, 266, 270, 276, 279, 283, 284, 287, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 307, 308, 310, 312, 316, 319, 324, 330, 332, 333, 335, 339, 340, 341, 342, 345, 347, 349, 350, 351, 353, 354, 355, 358, 360, 361, 371, 373, 379, 382, 383, 384, 386, 387, 391, 394, 395, 399, 400, 402, 404, 407, 410, 412, 413, 419, 420, 430, 431, 434, 435, 436, 437, 438, 442, 449, 452, 456, 457, 463, 466, 470, 474, 475, 476, 479, 494, 497, 499, 500, 501, 510, 511, 512, 514, 515, 516, 517)
            and L.spaceport_rec_id = S.spaceport_rec_id
            group by S.spaceport_rec_id
        """
    active_spaceport_df = pd.read_sql_query(sql, conn)
    active_spaceport_df['LOCATION'] = active_spaceport_df[['LOCATION_1', 'LOCATION_2']].apply(lambda x: f'{x.LOCATION_1} {x.LOCATION_2}', axis=1)
    #print(active_spaceport_df)

    active_spaceport_df['NUM_LAUNCHES'] = 0  
    for index, rec in active_spaceport_df.iterrows():
        sql = """ select count(*)  
                from launches as L, spaceports as S 
                where launches_rec_id in(248, 254, 257, 258, 260, 261, 263, 266, 270, 276, 279, 283, 284, 287, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 307, 308, 310, 312, 316, 319, 324, 330, 332, 333, 335, 339, 340, 341, 342, 345, 347, 349, 350, 351, 353, 354, 355, 358, 360, 361, 371, 373, 379, 382, 383, 384, 386, 387, 391, 394, 395, 399, 400, 402, 404, 407, 410, 412, 413, 419, 420, 430, 431, 434, 435, 436, 437, 438, 442, 449, 452, 456, 457, 463, 466, 470, 474, 475, 476, 479, 494, 497, 499, 500, 501, 510, 511, 512, 514, 515, 516, 517)
                and L.spaceport_rec_id = S.spaceport_rec_id
                and S.spaceport_rec_id ={id}
            """.format( id = rec['SPACEPORT_REC_ID'])
        launches_by_spaceport_df = pd.read_sql_query(sql, conn)
        count = launches_by_spaceport_df.loc[0, 'count(*)']
        active_spaceport_df.loc[index,'NUM_LAUNCHES'] = count

    print(active_spaceport_df)
   
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
   
    sns.barplot(data= active_spaceport_df, x ='NUM_LAUNCHES', y='LOCATION', ax=ax1, palette = "Blues_r")
    ax1.set_title('Count Launch Events having TFR NOTAMs by Spaceport Location')
    ax1.set_xlabel('Count Launch Events')
    ax1.set_ylabel('Spaceport Location')
    fig.tight_layout()
    plt.show()

def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()
    
    plot_launches_by_spaceport(conn)  
    
    conn.close()
    

if __name__ == "__main__":
    main()
