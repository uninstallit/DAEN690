import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)

from pipelines_.pipelines import lower_case_column_text_pipeline
from functions_.spaceports_dict import get_launch_location, get_spaceports_dict, has_launch_spaceport_rec_id, get_spaceport_artcc

spaceports_dict = get_spaceports_dict()

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

inclusion_words = """ launch OR space OR 91.143 OR "ATTENTION AIRLINE DISPATCHERS" OR "HAZARD AREA" 
                     OR "STNR ALT" OR "STNR ALTITUDE" OR "STATIONARY ALT*" 
                     OR "TEMPORARY FLIGHT RESTRICTION" OR "TEMPORARY FLT RESTRICTION*" OR "FLIGHT REST*" OR TFR """

# combine inclusions and exclusionwords together
searchXX = """(launch OR space  OR  airspace OR "91*143" OR "attention airline dispatchers" OR "hazard area" 
            OR "stnr alt" OR "stnr altitude" OR "stationary alt*" 
            OR "temporary flight restriction*" OR "temporary flt rest*" OR "flight rest*" OR tfr) 
            NOT obst* NOT fire NOT unmanned NOT crane OR uas NOT aerial NOT drill NOT installed
            NOT terminal NOT parking NOT rwy NOT taxi NOT twy NOT hangar NOT chemical
            NOT pavement NOT firing NOT "out of service" NOT volcan NOT turbine NOT flare NOT wx 
            NOT weather NOT aerodrome NOT apron NOT tower NOT hospital NOT covid NOT medical 
            NOT copter NOT disabled passenger NOT passanger NOT arctic NOT artic NOT defense  NOT defence 
            NOT helipad NOT bird NOT laser NOT heliport NOT ordnance NOT decommisioned NOT decomissioned 
            NOT dropzone NOT runway NOT wind NOT aerobatic NOT airfield NOT model NOT para* NOT parachute 
            NOT jumpers NOT paradrops NOT glide NOT tcas NOT accident NOT investigation NOT training 
            NOT approach NOT explosion NOT explosive NOT demolition* NOT launched """

exclusion_list  =[ 'obst','obstruct','obstn', 'obstruction',
            'fire', 'unmanned', 'crane', 'uas', 'aerial', 'drill', 'installed',
            'terminal','parking', 'rwy', 'taxi', 'twy', 'hangar', 'chemical',
            'pavement', 'firing', "out of service", 'volcan', 'turbine', 'flare', 'wx',
            'weather','NOT','aerodrome', 'apron', 'tower', 'hospital', 'covid', 'medical'
            'copter', 'disabled passenger', 'passanger', 'arctic', 'artic' ,'defense', 'defence'
            'helipad','bird','laser','heliport', 'ordnance','decommisioned','decomissioned'
            'dropzone','runway','wind','aerobatic', 'airfiel', 'model','para', 'parachute' 
            'jumpers','paradrops','glide','tcas','accident','investigation','training' 
            'approach', 'explosion', 'explosive', 'demolition','demolitions']
    
inclusions = """launch OR space OR airspace OR "91*143" OR "attention airline dispatchers" OR "hazard area" 
            OR "stnr alt" OR "stnr altitude" OR "stationary alt*" 
            OR "temporary flight restriction*" OR "temporary flt rest*" OR "flight rest*" OR tfr"""


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

    notams_df["E_CODE"] = notams_df["E_CODE"].fillna("UNKNOWN")
    notams_df["TEXT"] = notams_df["TEXT"].fillna("UNKNOWN")

    # create lowercase columns for better full-text search before inserting to the virtual table
    e_code_lc = lower_case_column_text_pipeline("E_CODE").fit_transform(notams_df)
    e_code_lc = np.squeeze(e_code_lc, axis=1)
    notams_df['E_CODE_LC'] = e_code_lc

    text_lc = lower_case_column_text_pipeline("TEXT").fit_transform(notams_df)
    text_lc = np.squeeze(text_lc, axis=1)
    notams_df['TEXT_LC'] = text_lc

    #print(notams_df[['NOTAM_REC_ID','E_CODE']])

    #create virtual table
    conn_v = sqlite3.connect(':memory:')
    cur_v = conn_v.cursor()
    cur_v.execute('create virtual table virtual_notams using fts5(NOTAM_REC_ID,E_CODE, TEXT, ISSUE_DATE,POSSIBLE_START_DATE,POSSIBLE_END_DATE, MIN_ALT,MAX_ALT,AFFECTED_FIR,LOCATION_CODE,LOCATION_NAME, E_CODE_LC ,tokenize="unicode61");')
    notams_df.to_sql('virtual_notams', conn_v, if_exists='append', index = False)
    print('Finished inserting to virtual_notams table')

    return (conn_v, cur_v)

def convert_str_datetime_unix_datetime(date_time_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    date_time_obj = datetime.strptime(date_time_str, date_format)
    return datetime.timestamp(date_time_obj)


# Find TFR NOTAM condition: shortest duration, keywords, and USA locations KXXX, ZXX
def find_tfr_for_launch(conn_v, cur_v, launch):
    
    launch_rec_id, launch_date  = launch['LAUNCHES_REC_ID'], launch['LAUNCH_DATE']
    spaceport_rec_id = launch['SPACEPORT_REC_ID']
    print(f'launch_spaceport_rec_id: {spaceport_rec_id}')
    launch_pad_location = get_spaceport_artcc(spaceport_rec_id) 
    print(f'** launch_pad_location: {spaceport_rec_id} {launch_pad_location}')

    print(f'launch: {launch_rec_id}, {launch_date}')

    #   |2days----Possible_start_time-----|launch_time|------Possible_end_time------2days|
    # sql =""" SELECT NOTAM_REC_ID, MIN_ALT, MAX_ALT, ISSUE_DATE, POSSIBLE_START_DATE, POSSIBLE_END_DATE, LOCATION_CODE, E_CODE  FROM virtual_notams 
    #         WHERE (DATETIME(POSSIBLE_START_DATE) <= DATETIME('{launch_date}') AND DATETIME(POSSIBLE_START_DATE) > DATETIME('{launch_date}', '-2 days'))
    #         and (DATETIME(POSSIBLE_END_DATE) > DATETIME('{launch_date}') AND DATETIME(POSSIBLE_END_DATE) < DATETIME('{launch_date}', '+2 days')) 
    #         and (LOCATION_CODE like 'Z%' or LOCATION_CODE like 'K%' or LOCATION_NAME like '%ARTCC%' or AFFECTED_FIR like 'Z%' or AFFECTED_FIR like 'K%')
    #         and (E_CODE_LC not null or TEXT_LC not null) 
    #         and (E_CODE_LC MATCH '{q}' or TEXT_LC MATCH '{q}')
    #         """.format(launch_date=launch_date, q=search)

    #   |2days----Possible_start_time-----|launch_time|------Possible_end_time------2days|
    sql =""" SELECT NOTAM_REC_ID, MIN_ALT as MIN_ALT_K, MAX_ALT as MAX_ALT_K, ISSUE_DATE, POSSIBLE_START_DATE, POSSIBLE_END_DATE, LOCATION_CODE, E_CODE, E_CODE_LC FROM virtual_notams 
            WHERE (DATETIME(POSSIBLE_START_DATE) <= DATETIME('{launch_date}') AND DATETIME(POSSIBLE_START_DATE) > DATETIME('{launch_date}', '-2 days'))
            and (DATETIME(POSSIBLE_END_DATE) > DATETIME('{launch_date}') AND DATETIME(POSSIBLE_END_DATE) < DATETIME('{launch_date}', '+2 days')) 
            and (LOCATION_CODE in {location} or LOCATION_NAME in {location} or AFFECTED_FIR in {location})
            and E_CODE_LC not null
            and E_CODE_LC MATCH '{q}' 
            """.format(launch_date=launch_date, location = launch_pad_location, q=inclusions)
    notams_df = pd.read_sql_query(sql, conn_v)
   
    # manually filter out the exclusion key words since fts does not work well when combining inclusions and exclusion together
    good_notam_indx =[]
    for indx, notam in notams_df.iterrows():
        e_code = notam['E_CODE_LC']
        if any(word in e_code for word in exclusion_list):
           continue
        else:
            good_notam_indx.append(indx)
    notams_df = notams_df.iloc[good_notam_indx]

    # select TFR with shortest duration
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

    sql = """ select * from launches  """
    launches_df = pd.read_sql_query(sql, conn)
    launches_df["SPACEPORT_REC_ID"] = launches_df["SPACEPORT_REC_ID"].fillna(9999)
    launches_df["SPACEPORT_REC_ID"] = launches_df["SPACEPORT_REC_ID"].astype('int')
    launches_has_no_tfr = []
    tfr_notams = []
    for index, launch in launches_df.iterrows():
        # if launch['LAUNCHES_REC_ID'] != 286:
        #     continue

        if has_launch_spaceport_rec_id(spaceports_dict, launch['SPACEPORT_REC_ID']) == False:
            continue
        tfr_notam = find_tfr_for_launch(conn_v, cur_v, launch)
        if tfr_notam is not None:
            launch_spaceport_rec_id = int(launch['SPACEPORT_REC_ID'])
            # append a launch_rec_id column to a  TFR
            launch_location, launch_state_location = get_launch_location(spaceports_dict, launch_spaceport_rec_id)
            launch_tfr_notam = pd.concat([pd.Series([launch['LAUNCHES_REC_ID']], index=['LAUNCHES_REC_ID']), tfr_notam])
            launch_tfr_notam = pd.concat([launch_tfr_notam,pd.Series([launch_location], index=['LAUNCH_LOCATION'])])
            launch_tfr_notam = pd.concat([launch_tfr_notam,pd.Series([launch_state_location], index=['LAUNCH_STATE_LOCATION'])])

            df = pd.DataFrame([launch_tfr_notam]) # same as DataFrame(launch_tfr_notam).transpose()
            tfr_notams.append(df)
        else:
            launches_has_no_tfr.append(launch['LAUNCHES_REC_ID'])

    results = pd.concat(tfr_notams)
    results = results.drop(columns=['E_CODE_LC'])
    print(f'Total TFR count:{len(results)}')
    print(f'Launch_rec_id without TFR: {launches_has_no_tfr}')
    return results
    
def main():

    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()

    results = find_tfr_notams(conn, cursor)
    results.to_csv(f'./data/tfr_notams.{datetime.now().strftime("%m%d")}.csv', index=False)
    
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
