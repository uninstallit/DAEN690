import sqlite3
import pandas as pd

import sys
import os

from pandas import DataFrame
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)

exclusion_words = """ obst* OR fire OR unmanned OR crane OR uas OR aerial OR drill OR installed OR 
                            terminal OR parking OR rwy OR taxi OR twy OR hangar OR chemical OR pavement OR 
                            firing OR out* OR volcan OR turbine OR flare OR wx OR weather OR 
                            aerodrome OR apron OR tower OR hospital OR covid OR medical OR copter OR 
                            disabled OR passenger OR passanger OR arctic OR artic OR defense OR defence OR 
                            helipad OR bird  OR laser  OR heliport OR ordnance OR decommisioned OR decomissioned OR 
                            dropzone OR runway OR wind  OR aerobatic OR airfield OR model  OR 
                            para*  OR parachute OR jumpers OR paradrops OR 
                            glide OR tcas OR accident OR investigation OR training OR 
                            approach OR explosion OR explosive OR demolitions """


def create_virtual_full_text_search_notam_table(conn, cursor):
    print('create_virtual_notam_table')
    cols = """ NOTAM_REC_ID, NOTAM_ID, NOTAM_NUMBER, NOTAM_TYPE, E_CODE, TEXT, POSSIBLE_START_DATE, POSSIBLE_END_DATE, ISSUE_DATE, MIN_ALT, MAX_ALT, AFFECTED_FIR, LOCATION_CODE, LOCATION_NAME  """

    sql = f'select {cols} from notams '
    notams_df = pd.read_sql_query(sql, conn)

    # clean text first
    #create virtual table
    conn_v = sqlite3.connect(':memory:')
    cur_v = conn_v.cursor()
    cur_v.execute('create virtual table virtual_notams using fts5(NOTAM_REC_ID,NOTAM_ID,NOTAM_NUMBER, NOTAM_TYPE,E_CODE,TEXT,POSSIBLE_START_DATE,POSSIBLE_END_DATE,ISSUE_DATE, MIN_ALT,MAX_ALT,AFFECTED_FIR,LOCATION_CODE,LOCATION_NAME, tokenize="porter unicode61");')
    
    notams_df.to_sql('virtual_notams', conn_v, if_exists='append', index = False)
    print('Finished inserting to virtual_notams table')
    return (conn_v, cur_v)


def match_launch(conn_v, cur_v, launch_time, possible_end_time):
    print(f'match  dlaunch_time:{launch_time}')

    search = exclusion_words
   
    #    |2days-------Possible_start_time---|launch_time|------Possible_end_time------2days|
    sql = """ SELECT NOTAM_REC_ID,  POSSIBLE_START_DATE, POSSIBLE_END_DATE, LOCATION_CODE, E_CODE FROM virtual_notams  
                WHERE (DATETIME(POSSIBLE_START_DATE) <= DATETIME('{launch_date}') AND DATETIME(POSSIBLE_START_DATE) > DATETIME('{launch_date}', '-2 days'))
                and (DATETIME(POSSIBLE_END_DATE) > DATETIME('{launch_date}') AND DATETIME(POSSIBLE_END_DATE) < DATETIME('{launch_date}', '+2 days'))
                and (E_CODE not null or TEXT not null) 
                and (E_CODE MATCH "{q}" or TEXT MATCH "{q}") """

    sql1 = sql.format(launch_date=launch_time, poss_end_date=possible_end_time, q=search)
    virt_notams_df = pd.read_sql_query(sql1, conn_v)
    print(f'Found matching notams len:{len(virt_notams_df)}')
    print(virt_notams_df.head(5))
    
def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()
    (conn_v, cur_v) = create_virtual_full_text_search_notam_table(conn, cursor)

    launch_rec_id = 391
    launch_time = '2018-04-02 20:30:38'
    possible_end_time = '2018-04-02 21:08:00'
    match_launch(conn_v, cur_v, launch_time, possible_end_time)

    conn.close()

if __name__ == "__main__":
    main()