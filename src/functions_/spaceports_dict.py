import sqlite3
from jinja2 import Undefined
import pandas as pd
import numpy as np

# United States has 22 Air Route Traffic Control Centers (ARTCC): https://en.wikipedia.org/wiki/List_of_U.S._Air_Route_Traffic_Control_Centers#
# Airport Data to find ARTCC for spaceports: https://www.airport-data.com 
# ICAO https://www.faa.gov/air_traffic/publications/atpubs/locid_html/chap1_section_6_paragraph_H.html
# https://en.wikipedia.org/wiki/List_of_U.S._Air_Route_Traffic_Control_Centers

# store artcc as a string for each spaceport ready for sql to use
spaceport_artcc_dict = {} 
spaceport_artcc_dict[2] =  "('ZMA','KZMA','ZJX', 'KZJX')"         # Cape Canaveral	Florida
spaceport_artcc_dict[3] =  "('ZJX', 'KZJX')"                      # Jacksonville	Florida
spaceport_artcc_dict[4] =  "('ZDV','KZDV')"                       # Adams County	Colorado
spaceport_artcc_dict[5] =  "('ZAB','KZAB')"                       # Adams County	Colorado
spaceport_artcc_dict[6] =  "('ZLA', 'KZLA')"                      # Lancaster	California
spaceport_artcc_dict[7] =  "('ZHU','KZHU')"                       # Houston	Texas
spaceport_artcc_dict[8] =  "('ZMA','KZMA','ZJX', 'KZJX')"         # Merritt Island	Florida
spaceport_artcc_dict[9] =  "('ZDC','KZDC')"                       # Wallops Island	Virginia
spaceport_artcc_dict[10] = "('ZFW','KZFW')"                       # Midland	Texas
spaceport_artcc_dict[11] = "('ZLA','KZLA')"                       # Mojave	California
spaceport_artcc_dict[12] = "('ZFW','KZFW','KDFW')"                # Burns Flat	Oklahoma
spaceport_artcc_dict[13] = "('ZHN','PHNL')"                       # Kekaha	Hawaii
spaceport_artcc_dict[14] = "('ZAN','KZAN','PAZA')"                # Kodiak Island	Alaska
spaceport_artcc_dict[15] = "('ZAN', 'KZAN', 'PAZA')"              # Fairbanks	Alaska
spaceport_artcc_dict[16] = "('PKRO','PKWA','ROI')"                # Kwajalein Atoll	Marshall Islands. No ARTCC found far land
spaceport_artcc_dict[17] = "('ZHN', 'PHNL', 'PHZH')"              # Wake Island	Marshall Islands, Honolulu
spaceport_artcc_dict[18] = "('ZMA','KZMA','ZJX','KZJX')"          # Titusville	Florida
spaceport_artcc_dict[19] = "('ZAB', 'KZAB')"                      # Truth or Consequences	New Mexico
spaceport_artcc_dict[20] = "('ZJX','KZJX')"                       # Woodbine	Georgia
spaceport_artcc_dict[21] = "('ZFW','KZFW')"                       # McGregor	Texas
spaceport_artcc_dict[22] = "('ZHU','KZHU')"                       # Brownsville	Texas
spaceport_artcc_dict[23] = "('ZLA','KZLA')"                       # Lompoc	California
spaceport_artcc_dict[24] = "('ZAB', 'KZAB')"                      # Las Cruces	New Mexico

def get_spaceport_artcc(spaceport_rec_id):
    if spaceport_rec_id in spaceport_artcc_dict.keys():
        return spaceport_artcc_dict[spaceport_rec_id]

    return "()"

def get_spaceports_dict(conn = None):
    if conn == None:
        conn = sqlite3.Connection("./data/svo_db_20201027.db")
    
    sql = """ SELECT * from spaceports """
    spaceports_df = pd.read_sql_query(sql, conn)
    
    spaceports_dict = {}
    for _, space in spaceports_df.iterrows():
        spaceports_dict[space['SPACEPORT_REC_ID']] = space

    return spaceports_dict

def get_launch_location(spaceports_dict, launch_spaceport_rec_id):
    if launch_spaceport_rec_id in spaceports_dict.keys():
        launch_location, launch_state_location = spaceports_dict[launch_spaceport_rec_id]['LOCATION_1'], spaceports_dict[launch_spaceport_rec_id]['LOCATION_2'] 
        return (launch_location, launch_state_location)

    return ("","")

def has_launch_spaceport_rec_id(spaceports_dict, launch_spaceport_rec_id):
    if launch_spaceport_rec_id == 9999:
        return False

    return launch_spaceport_rec_id in spaceports_dict.keys()


def main():
    print('Hello spaceports')

if __name__ == "__main__":
    main()