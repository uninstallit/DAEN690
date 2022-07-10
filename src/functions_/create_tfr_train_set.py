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

exclusion_words = """obst* OR fire OR unmanned OR crane OR uas OR aerial OR drill OR installed OR 
                            terminal OR parking OR rwy OR taxi OR twy OR hangar OR chemical OR pavement OR 
                            firing OR "out of service" OR volcan OR turbine OR flare OR wx OR weather OR 
                            aerodrome OR apron OR tower OR hospital OR covid OR medical OR copter OR 
                            disabled OR passenger OR passanger OR arctic OR artic OR defense OR defence OR 
                            helipad OR bird OR laser OR heliport OR ordnance OR decommisioned OR decomissioned OR 
                            dropzone OR runway OR wind OR aerobatic OR airfield OR model  OR 
                            para*  OR parachute OR jumpers OR paradrops OR 
                            glide OR tcas OR accident OR investigation OR training OR 
                            approach OR explosion OR explosive OR demolitions 
                            OR baloon* OR balloon* OR hurricane* OR "transfered from ship" OR "troop transport" 
                            OR construction OR "radio unavailable" OR "handled by falcon radio" OR rattlesnake """

exclusion_list  = ['obst','obstruct','obstn', 'obstruction',
            'fire', 'unmanned', 'crane', 'uas', 'aerial', 'drill', 'installed',
            'terminal','parking', 'rwy', 'taxi', 'twy', 'hangar', 'chemical',
            'pavement', 'firing', "out of service", 'volcan', 'turbine', 'flare', 'wx',
            'weather','aerodrome', 'apron', 'tower', 'hospital', 'covid', 'medical'
            'copter', 'disabled passenger', 'passanger', 'arctic', 'artic' ,'defense', 'defence'
            'helipad','bird','laser','heliport', 'ordnance','decommisioned','decomissioned'
            'dropzone','runway','wind','aerobatic', 'airfiel', 'model','para', 'parachute' 
            'jumpers','paradrops','glide','tcas','accident','investigation','training' 
            'approach', 'explosion', 'explosive', 'demolition','demolitions', 'baloon', 'balloon', 'hurricane']

# use in finding TFR notams
exclusions = """ NOT (obst* OR fire OR unmanned OR crane OR uas OR aerial OR drill OR installed OR 
                            terminal OR parking OR rwy OR taxi OR twy OR hangar OR chemical OR pavement OR 
                            firing OR "out of service" OR volcan OR turbine OR flare OR wx OR weather OR 
                            aerodrome OR apron OR tower OR hospital OR covid OR medical OR copter OR 
                            disabled OR passenger OR passanger OR arctic OR artic OR defense OR defence OR 
                            helipad OR bird OR laser OR heliport OR ordnance OR decommisioned OR decomissioned OR 
                            dropzone OR runway OR wind OR aerobatic OR airfield OR model OR 
                            para* OR parachute OR jumpers OR paradrops OR glide OR tcas OR accident OR investigation OR 
                            training OR approach OR explosion OR explosive OR demolitions 
                            OR baloon* OR balloon* OR hurricane OR "transfered from ship" OR "troop transport" 
                            OR construction OR "radio unavailable" OR "handled by falcon radio" OR rattlesnake) """
    
# add vehicle name to the initial inclusion_words for this dataset notams
inclusions = """launch OR space OR "91*143" OR "attention airline dispatchers" OR "hazard area" 
            OR "stnr alt" OR "stnr altitude" OR "stationary alt*" 
            OR "temporary flight restriction*" OR "temporary flt rest*" OR "flight rest*" OR tfr 
            OR rocket OR missile OR canaveral OR kennedy OR nasa OR antares OR orion OR atlas
            OR zenit OR falcon OR dragon OR spaceship OR minuteman OR trident """

# for finding TFR notams
search1 = """(launch {exclusions})
            OR (space {exclusions}) 
            OR (rocket {exclusions}) OR (missile {exclusions}) OR (canaveral {exclusions}) OR (kennedy {exclusions}) OR (nasa {exclusions} )
            OR (antares {exclusions}) OR (orion {exclusions}) OR (atlas {exclusions}) OR (zenit {exclusions}) OR (falcon {exclusions}) 
            OR (dragon {exclusions}) OR (spaceship {exclusions}) OR (minuteman {exclusions}) OR (trident {exclusions}) 
            """.format(exclusions=exclusions)

# for generate good notams
search2 = """(launch {exclusions})
            OR (space {exclusions}) 
            OR ("91*143" {exclusions}) 
            OR ("attention airline dispatchers" {exclusions}) 
            OR ("hazard area" {exclusions} )
            OR ("stnr alt" {exclusions})
            OR ("stnr altitude" {exclusions})
            OR ("stationary alt*" {exclusions})
            OR ("temporary flight restriction*" {exclusions})
            OR ("temporary flt rest*" {exclusions})
            OR ("flight rest*" {exclusions})
            OR (rocket {exclusions}) OR (missile {exclusions}) OR (canaveral {exclusions}) OR (kennedy {exclusions}) OR (nasa {exclusions} )
            OR (antares {exclusions}) OR (orion {exclusions}) OR (atlas {exclusions}) OR (zenit {exclusions}) OR (falcon {exclusions}) 
            OR (dragon {exclusions}) OR (spaceship {exclusions}) OR (minuteman {exclusions}) OR (trident {exclusions}) 
            """.format(exclusions=exclusions)

def check_good_notams(conn, tfr_notams_df,good_notams_df ):
    # check how many tfr notams in the good_notams_df
    tfr_notams = tfr_notams_df['NOTAM_REC_ID'].to_numpy()
    saw_tfr_in_good_notams = []
    tfr_not_in_good_notams = []
    for trf in tfr_notams:
        if (good_notams_df['NOTAM_REC_ID'] == trf).any():
            saw_tfr_in_good_notams.append(trf)
        else:
            tfr_not_in_good_notams.append(trf)
    print(f'saw tfr in good_notams list: {len(saw_tfr_in_good_notams)} // {len(tfr_notams_df)}')
    print(f'trf not in good_notams list:{len(tfr_not_in_good_notams)} // {len(tfr_notams_df)} ')
    
   # compare with human_matches
    sql = """ select * from human_matches  """
    human_matches_df = pd.read_sql_query(sql, conn)
    found_in_human_matches= []
    not_found_in_human_matches = []
    for _, match in human_matches_df.iterrows():
        if (good_notams_df['NOTAM_REC_ID'] == match['NOTAM_REC_ID']).any():
        # if len(good_notams_df[good_notams_df['NOTAM_REC_ID'] == match]):
            found_in_human_matches.append(match)
        else:
            not_found_in_human_matches.append((match['LAUNCHES_REC_ID'],  match['NOTAM_REC_ID']))

    print(f'Found good notams in human matches len:{len(found_in_human_matches)} .Total provided human_matches len:{len(human_matches_df)}')
    
    # compare with human_poor_matches
    sql = """ select * from human_poor_matches  """
    human_poor_matches_df = pd.read_sql_query(sql, conn)
    poor_matches = human_poor_matches_df['NOTAM_REC_ID'].to_numpy()
    found_in_poor_matches= []
    for poor in poor_matches:
        if (good_notams_df['NOTAM_REC_ID'] == poor).any():
            found_in_poor_matches.append(poor)
    print(f'Found in human poor matches len:{len(found_in_poor_matches)}')

def convert_str_datetime_unix_datetime(date_time_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    date_time_obj = datetime.strptime(date_time_str, date_format)
    return datetime.timestamp(date_time_obj)

def create_virtual_full_text_search_notam_table(conn, cursor):
    cols = """ NOTAM_REC_ID, E_CODE, TEXT, ISSUE_DATE, POSSIBLE_START_DATE, POSSIBLE_END_DATE, MIN_ALT, MAX_ALT, AFFECTED_FIR, LOCATION_CODE, LOCATION_NAME, NOTAM_TYPE, ACCOUNT_ID  """

    sql = """ select {cols} from notams  
                WHERE (LOCATION_CODE like 'Z%' or LOCATION_CODE like 'K%' or LOCATION_NAME like '%ARTCC%' or AFFECTED_FIR like 'Z%' or AFFECTED_FIR like 'K%' 
                or LOCATION_CODE in ('PHNL','PHZH') or LOCATION_CODE in ('PHNL','PHZH') or AFFECTED_FIR in ('PHNL','PHZH') ) """
    sql = sql.format(cols=cols)
    notams_df = pd.read_sql_query(sql, conn)
    print(f'Total notams brought to vitual_table len:{len(notams_df)}')

    notams_df["E_CODE"] = notams_df["E_CODE"].fillna("UNKNOWN")
    notams_df["TEXT"] = notams_df["TEXT"].fillna("UNKNOWN")
    notams_df["NOTAM_TYPE"] = notams_df["NOTAM_TYPE"].fillna("UNKNOWN")

    # create lowercase columns for better full-text search before inserting to the virtual table
    e_code_lc = lower_case_column_text_pipeline("E_CODE").fit_transform(notams_df)
    e_code_lc = np.squeeze(e_code_lc, axis=1)
    notams_df['E_CODE_LC'] = e_code_lc

    text_lc = lower_case_column_text_pipeline("TEXT").fit_transform(notams_df)
    text_lc = np.squeeze(text_lc, axis=1)
    notams_df['TEXT_LC'] = text_lc

    #create virtual table
    conn_v = sqlite3.connect(':memory:')
    cur_v = conn_v.cursor()
    cur_v.execute('create virtual table virtual_notams using fts5(NOTAM_REC_ID,E_CODE, TEXT, ISSUE_DATE,POSSIBLE_START_DATE,POSSIBLE_END_DATE, MIN_ALT,MAX_ALT,AFFECTED_FIR,LOCATION_CODE,LOCATION_NAME, NOTAM_TYPE,ACCOUNT_ID, E_CODE_LC, TEXT_LC,tokenize="unicode61");')
    notams_df.to_sql('virtual_notams', conn_v, if_exists='append', index = False)
    print(f'Finished inserting to virtual_notams table len:{len(notams_df)}')
    return (conn_v, cur_v)


# Find TFR NOTAM condition: shortest duration, keywords, and USA locations KXXX, ZXX
def find_initial_tfr(conn_v, cur_v, launch):
    launch_rec_id, launch_date, spaceport_rec_id  = launch['LAUNCHES_REC_ID'], launch['LAUNCH_DATE'], launch['SPACEPORT_REC_ID']
    launch_pad_artcc = get_spaceport_artcc(spaceport_rec_id) 
    (location, state_location ) = get_launch_location(spaceports_dict, spaceport_rec_id)
    print(f'* launch:{launch_rec_id} launch_date:{launch_date}, spaceport_rec_id:{spaceport_rec_id} artcc:{launch_pad_artcc} location:{location}, state:{state_location} ')


    #   |-30days----Possible_start_time-----|launch_time|------Possible_end_time------+30days|
    sql =""" SELECT NOTAM_REC_ID, MIN_ALT as MIN_ALT_K, MAX_ALT as MAX_ALT_K, ISSUE_DATE, POSSIBLE_START_DATE, POSSIBLE_END_DATE, LOCATION_CODE, E_CODE, E_CODE_LC, NOTAM_TYPE FROM virtual_notams 
            WHERE (DATETIME(POSSIBLE_START_DATE) <= DATETIME('{launch_date}') AND DATETIME(POSSIBLE_START_DATE) > DATETIME('{launch_date}', '-30 days'))
            and (DATETIME(POSSIBLE_END_DATE) >= DATETIME('{launch_date}') AND DATETIME(POSSIBLE_END_DATE) < DATETIME('{launch_date}', '+30 days')) 
            and (LOCATION_CODE in {spaceport_location} or LOCATION_NAME in {spaceport_location} or AFFECTED_FIR in {spaceport_location})
            and NOTAM_TYPE != 'NOTAMC'
            and MAX_ALT > 15.0
            and E_CODE_LC not null
            and E_CODE_LC MATCH '{q}' 
            """.format(launch_date=launch_date, spaceport_location = launch_pad_artcc, q=search1)
    notams_df = pd.read_sql_query(sql, conn_v)
   
    # select TFR with shortest duration
    tfr_notam = None
    shortest_duration = float('inf')
    if len(notams_df):
        for _, notam in notams_df.iterrows():
            # if notam['NOTAM_TYPE'] == 'NOTAMC':
            #     continue
            start_str, end_str = notam['POSSIBLE_START_DATE'], notam['POSSIBLE_END_DATE']
            duration = convert_str_datetime_unix_datetime(end_str) - convert_str_datetime_unix_datetime(start_str)
            if duration < shortest_duration:
                shortest_duration = duration
                tfr_notam = notam

    return tfr_notam

def create_tfr_notams(conn_v, cur_v, launches_df):
    launches_has_no_tfr = []
    tfr_notams = []
    for index, launch in launches_df.iterrows():
        if has_launch_spaceport_rec_id(spaceports_dict, launch['SPACEPORT_REC_ID']) == False:
            continue
        tfr_notam = find_initial_tfr(conn_v, cur_v, launch)
        if tfr_notam is not None:
            launch_spaceport_rec_id = launch['SPACEPORT_REC_ID']
            # append a launch_rec_id column to a  TFR
            launch_location, launch_state_location = get_launch_location(spaceports_dict, launch_spaceport_rec_id)
            launch_tfr_notam = pd.concat([pd.Series([launch['LAUNCHES_REC_ID']], index=['LAUNCHES_REC_ID']), tfr_notam])
            launch_tfr_notam = pd.concat([launch_tfr_notam,pd.Series([launch_location], index=['LAUNCH_LOCATION'])])
            launch_tfr_notam = pd.concat([launch_tfr_notam,pd.Series([launch_state_location], index=['LAUNCH_STATE_LOCATION'])])
            launch_tfr_notam = pd.concat([launch_tfr_notam,pd.Series([launch['LAUNCH_DATE']], index=['LAUNCH_DATE'])])
            launch_tfr_notam = pd.concat([launch_tfr_notam,pd.Series([launch['SPACEPORT_REC_ID']], index=['SPACEPORT_REC_ID'])])

            df = pd.DataFrame([launch_tfr_notam]) 
            tfr_notams.append(df)
        else:
            launches_has_no_tfr.append(launch['LAUNCHES_REC_ID'])

    results_df = pd.concat(tfr_notams)
    results_df = results_df.drop(columns=['E_CODE_LC'])
    print(f'Total TFR count:{len(results_df)}')
    print(f'Launch_rec_id without TFR: {launches_has_no_tfr}')
    return results_df

# select NOTAMs matching the TFR exactly start or stop time, inclusion and exclusion words
def create_good_notams(conn_v):
    print(f'create_good_notams')
    sql ="""SELECT NOTAM_REC_ID, MIN_ALT as MIN_ALT_K, MAX_ALT as MAX_ALT_K, 
        POSSIBLE_START_DATE, POSSIBLE_END_DATE, LOCATION_CODE, E_CODE, E_CODE_LC, ACCOUNT_ID, NOTAM_TYPE  FROM virtual_notams 
        WHERE NOTAM_TYPE != 'NOTAMC'
        and MAX_ALT > 15.0
        and E_CODE_LC not null
        and E_CODE_LC MATCH '{q}' 
        """.format( q=search2)
    good_notams_df = pd.read_sql_query(sql, conn_v)
    good_notams_df = good_notams_df.drop_duplicates(subset=['NOTAM_REC_ID'])
    good_notams_df = good_notams_df.drop_duplicates(subset=['E_CODE'])
    good_notams_df = good_notams_df.drop(columns=['E_CODE_LC'])   


    print(f'good_notams len: {len(good_notams_df)}')
    return good_notams_df
    
def create_bad_notams(conn_v, cur_v):
    print(f'create_bad_notams')
    sql ="""SELECT NOTAM_REC_ID, MIN_ALT as MIN_ALT_K, MAX_ALT as MAX_ALT_K, 
        POSSIBLE_START_DATE, POSSIBLE_END_DATE, LOCATION_CODE, E_CODE, E_CODE_LC, ACCOUNT_ID, NOTAM_TYPE  FROM virtual_notams 
        WHERE NOTAM_TYPE != 'NOTAMC'
        and E_CODE_LC not null
        and E_CODE_LC MATCH '{q}' 
        """.format( q=exclusion_words)
    bad_notams_df = pd.read_sql_query(sql, conn_v)
    bad_notams_df = bad_notams_df.drop_duplicates(subset=['NOTAM_REC_ID'])
    bad_notams_df = bad_notams_df.drop_duplicates(subset=['E_CODE'])
    bad_notams_df = bad_notams_df.drop(columns=['E_CODE_LC'])    
    print(f'bad_notams len: {len(bad_notams_df)}')
    print(bad_notams_df)
    return bad_notams_df
    

def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()
    (conn_v, cur_v) = create_virtual_full_text_search_notam_table(conn, cursor)

    sql = """ select * from launches  """
    launches_df = pd.read_sql_query(sql, conn)
    launches_df["SPACEPORT_REC_ID"] = launches_df["SPACEPORT_REC_ID"].fillna(9999)
    launches_df["SPACEPORT_REC_ID"] = launches_df["SPACEPORT_REC_ID"].astype('int')

    ## create TFR notams
    # tfr_notams = create_tfr_notams(conn_v, cur_v, launches_df)
    # tfr_notams.to_csv(f'./data/tfr_notams.{datetime.now().strftime("%m%d")}.csv', index=False)

    good_notams_df = create_good_notams(conn_v)
    good_notams_df.to_csv(f'./data/possitive_unique_notams.{datetime.now().strftime("%m%d")}.csv', index=False)
    tfr_notams_df = pd.read_csv('./data/tfr_notams.csv', engine="python" )
    check_good_notams(conn, tfr_notams_df,good_notams_df)
       
    # bad_notams_df = create_bad_notams(conn_v, cur_v)
    # bad_notams_df.to_csv(f'./data/negative_unique_notams.{datetime.now().strftime("%m%d")}.csv', index=False)

    conn.close()
    

if __name__ == "__main__":
    main()
