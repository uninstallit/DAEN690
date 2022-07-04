import sqlite3
from jinja2 import Undefined
import pandas as pd
import numpy as np

def get_spaceports_dict(conn = None):
    if conn == None:
        conn = sqlite3.Connection("./data/svo_db_20201027.db")
    
    sql = """ SELECT * from spaceports """
    spaceports_df = pd.read_sql_query(sql, conn)
    conn.close()

    spaceports_dict = {}
    for _, space in spaceports_df.iterrows():
        spaceports_dict[space['SPACEPORT_REC_ID']] = space

    return spaceports_dict

def get_launch_location(spaceports_dict, launch_spaceport_rec_id):
    if launch_spaceport_rec_id in spaceports_dict:
        launch_location = spaceports_dict[launch_spaceport_rec_id]['LOCATION_1'] 
        launch_state_location  = spaceports_dict[launch_spaceport_rec_id]['LOCATION_2'] 
        return (launch_location, launch_state_location)
    return ("","")

def main():
    print('Hello spaceports')

if __name__ == "__main__":
    main()