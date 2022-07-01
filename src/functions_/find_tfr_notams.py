import sqlite3
import pandas as pd
from datetime import datetime
import sys
import os

from pandas import DataFrame
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)

exclusion_words = """ obst* OR fire OR unmanned OR crane OR uas OR aerial OR drill OR installed OR 
                            terminal OR parking OR rwy OR taxi OR twy OR hangar OR chemical OR pavement OR 
                            firing OR "out of service" OR volcan OR turbine OR flare OR wx OR weather OR 
                            aerodrome OR apron OR tower OR hospital OR covid OR medical OR copter OR 
                            disabled OR passenger OR passanger OR arctic OR artic OR defense OR defence OR 
                            helipad OR bird  OR laser  OR heliport OR ordnance OR decommisioned OR decomissioned OR 
                            dropzone OR runway OR wind  OR aerobatic OR airfield OR model  OR 
                            para*  OR parachute OR jumpers OR paradrops OR 
                            glide OR tcas OR accident OR investigation OR training OR 
                            approach OR explosion OR explosive OR demolitions """

inclusionwords = """ launch OR space  OR 91.143 OR "ATTENTION AIRLINE DISPATCHERS" OR "HAZARD AREA" OR
                            "STNR ALT" OR "STNR ALTITUDE" OR "STATIONARY ALT" OR
                            "TEMPORARY FLIGHT RESTRICTION" """

# combine inclusions and exclusionwords together
search = """launch OR space OR "91*143" OR "ATTENTION AIRLINE DISPATCHERS" OR "HAZARD AREA" OR
            "STNR ALT" OR "STNR ALTITUDE" OR "STATIONARY ALT" OR "TEMPORARY FLIGHT RESTRICTION" 
            NOT obst* NOT fire NOT unmanned NOT crane NOT uas NOT aerial NOT drill NOT installed
            NOT terminal NOT parking NOT rwy NOT taxi NOT twy NOT hangar NOT chemical
            NOT pavement NOT firing NOT "out of service" NOT volcan NOT turbine NOT flare NOT wx 
            NOT weather NOT aerodrome NOT apron NOT tower NOT hospital NOT covid NOT medical 
            NOT copter NOT disabled passenger NOT passanger NOT arctic NOT artic  NOT defense  NOT defence 
            NOT helipad NOT bird NOT laser NOT heliport NOT ordnance NOT decommisioned NOT decomissioned 
            NOT dropzone NOT runway NOT wind NOT aerobatic NOT airfield NOT model NOT para* NOT parachute 
            NOT jumpers NOT paradrops NOT glide NOT tcas NOT accident NOT investigation NOT training 
            NOT approach NOT explosion NOT explosive NOT demolitions"""

def create_negative_notams_dataset(conn_v, cur_v, launch_rec_id, launch_time):

    #    |2days-------Possible_start_time---|launch_time|------Possible_end_time------2days|
    sql = """ SELECT NOTAM_REC_ID,  POSSIBLE_START_DATE, POSSIBLE_END_DATE, LOCATION_CODE, LOCATION_NAME, E_CODE FROM virtual_notams  
                WHERE (DATETIME(POSSIBLE_START_DATE) <= DATETIME('{launch_date}') AND DATETIME(POSSIBLE_START_DATE) > DATETIME('{launch_date}', '-1 days'))
                and (DATETIME(POSSIBLE_END_DATE) > DATETIME('{launch_date}') AND DATETIME(POSSIBLE_END_DATE) < DATETIME('{launch_date}', '+1 days'))
                and (LOCATION_CODE like 'Z%' or LOCATION_CODE like 'K%' or LOCATION_NAME like '%ARTCC%' or AFFECTED_FIR like 'Z%' or AFFECTED_FIR like 'K%')
                and (E_CODE not null or TEXT not null) 
                and (E_CODE MATCH "{q}" or TEXT MATCH "{q}") """

    sql1 = sql.format(launch_date=launch_time, q=exclusion_words)
    neg_notams_df = pd.read_sql_query(sql1, conn_v)
    print(f'launch_rec_id:{launch_rec_id}, launch_time:{launch_time} - Found neg notams len:{len(neg_notams_df)}')
    
    if len(neg_notams_df):
        neg_notams_df['LAUNCHES_REC_ID'] = launch_rec_id
        neg_notams_df['LAUNCH_DATE'] = launch_time

    return  neg_notams_df


def create_virtual_full_text_search_notam_table(conn, cursor):
    print('create_virtual_notam_table')
    cols = """ NOTAM_REC_ID, E_CODE, TEXT, ISSUE_DATE, POSSIBLE_START_DATE, POSSIBLE_END_DATE, MIN_ALT, MAX_ALT, AFFECTED_FIR, LOCATION_CODE, LOCATION_NAME  """

    sql = f'select {cols} from notams '
    notams_df = pd.read_sql_query(sql, conn)

    # TODO  clean text before inserting to the virtual table

    #create virtual table
    conn_v = sqlite3.connect(':memory:')
    cur_v = conn_v.cursor()
    cur_v.execute('create virtual table virtual_notams using fts5(NOTAM_REC_ID,E_CODE,TEXT,ISSUE_DATE,POSSIBLE_START_DATE,POSSIBLE_END_DATE, MIN_ALT,MAX_ALT,AFFECTED_FIR,LOCATION_CODE,LOCATION_NAME, tokenize="porter unicode61");')
    notams_df.to_sql('virtual_notams', conn_v, if_exists='append', index = False)
    print('Finished inserting to virtual_notams table')

    return (conn_v, cur_v)

def convert_str_datetime_unix_datetime(date_time_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    date_time_obj = datetime.strptime(date_time_str, date_format)
    return datetime.timestamp(date_time_obj)

# TFR NOTAM for a launch is the one has a shortest duration, keywords, and USA locations KXXX, ZXX
def find_tfr_for_launch(conn_v, cur_v, launch):
    launch_rec_id, launch_date = launch['LAUNCHES_REC_ID'], launch['LAUNCH_DATE']
    print(f'launch: {launch_rec_id}, {launch_date}')

    #   |2days----Possible_start_time-----|launch_time|------Possible_end_time------2days|
    sql =""" SELECT NOTAM_REC_ID, MIN_ALT, MAX_ALT, ISSUE_DATE, POSSIBLE_START_DATE, POSSIBLE_END_DATE, E_CODE  FROM virtual_notams 
            WHERE (DATETIME(POSSIBLE_START_DATE) <= DATETIME('{launch_date}') AND DATETIME(POSSIBLE_START_DATE) > DATETIME('{launch_date}', '-2 days'))
            and (DATETIME(POSSIBLE_END_DATE) > DATETIME('{launch_date}') AND DATETIME(POSSIBLE_END_DATE) < DATETIME('{launch_date}', '+2 days')) 
            and (LOCATION_CODE like 'Z%' or LOCATION_CODE like 'K%' or LOCATION_NAME like '%ARTCC%')
            and (E_CODE not null or TEXT not null) 
            and (E_CODE MATCH '{q}' or TEXT MATCH '{q}')
            """.format(launch_date=launch_date, q=search)

    notams_df = pd.read_sql_query(sql, conn_v)

    tfr_notam = None
    shortest_duration = float('inf')
    if len(notams_df):
        for idx, notam in notams_df.iterrows():
            start_str, end_str = notam['POSSIBLE_START_DATE'], notam['POSSIBLE_END_DATE']
            duration = convert_str_datetime_unix_datetime(end_str) - convert_str_datetime_unix_datetime(start_str)
            if duration < shortest_duration:
                shortest_duration = duration
                tfr_notam = notam

        return tfr_notam
    
    return None

def find_tfr_notams(conn, cursor):
    (conn_v, cur_v) = create_virtual_full_text_search_notam_table(conn, cursor)

    sql = """ select LAUNCHES_REC_ID, LAUNCH_DATE from launches  """
    launch_df = pd.read_sql_query(sql, conn)

    launches_has_no_tfr = []
    tfr_notams = []
    for index, launch in launch_df.iterrows():
        tfr_notam = find_tfr_for_launch(conn_v, cur_v, launch)
        if tfr_notam is not None:
            # attach a launch_rec_id column to a  TFR
            launch_tfr_notam = pd.concat([pd.Series([launch['LAUNCHES_REC_ID']], index=['LAUNCHES_REC_ID']), tfr_notam])
            df = pd.DataFrame([launch_tfr_notam]) # same as DataFrame(launch_tfr_notam).transpose()
            tfr_notams.append(df)
        else:
            launches_has_no_tfr.append(launch['LAUNCHES_REC_ID'])
   
    results = pd.concat(tfr_notams)
    results.to_csv(f'./data/tft_notams.{datetime.now().strftime("%m%d")}.csv', index=False)
    print(f'Total TFR count:{len(results)}')
    print(f'Launch_rec_id without TFR: {launches_has_no_tfr}')
    return tfr_notams
    
def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()
    results = find_tfr_notams(conn, cursor)
    # for launch=391 ==> TRF = 835580, 2018-04-02 19:52:00 2018-04-02 21:08:00

    conn.close()

    exit()

    (conn_v, cur_v) = create_virtual_full_text_search_notam_table(conn, cursor)
    sql = """ select LAUNCHES_REC_ID, LAUNCH_DATE from launches  """
    launch_df = pd.read_sql_query(sql, conn)

    frames = []
    for index, row in launch_df.iterrows():
        launch_rec_id, launch_date = row['LAUNCHES_REC_ID'], row['LAUNCH_DATE']
        neg_notams_df =  create_negative_notams_dataset(conn_v, cur_v, launch_rec_id, launch_date)
        if len(neg_notams_df):
            frames.append(neg_notams_df)

    final_neg_notams_df = pd.concat(frames)
    print(f'final_neg_notams_df len: {len(final_neg_notams_df)}')
    print(final_neg_notams_df.head(100))

    # final_neg_notams_df.to_csv("./data/neg_notams_usa.csv", index=False)

    

if __name__ == "__main__":
    main()
