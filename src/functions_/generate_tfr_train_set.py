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

##### Original provided keywords
exclusion_words = """ (obst* OR fire OR unmanned OR crane OR uas OR aerial OR drill OR installed OR 
                            terminal OR parking OR rwy OR taxi OR twy OR hangar OR chemical OR pavement OR 
                            firing OR "out of service" OR volcan OR turbine OR flare OR wx OR weather OR 
                            aerodrome OR apron OR tower OR hospital OR covid OR medical OR copter OR 
                            disabled OR passenger OR passanger OR arctic OR artic OR defense OR defence OR 
                            helipad OR bird OR laser OR heliport OR ordnance OR decommisioned OR decomissioned OR 
                            dropzone OR runway OR wind OR aerobatic OR airfield OR model  OR 
                            para*  OR parachute OR jumpers OR paradrops OR 
                            glide OR tcas OR accident OR investigation OR training OR 
                            approach OR explosion OR explosive OR demolitions)"""

# inclusion_words = """ launch OR space OR 91.143 OR "ATTENTION AIRLINE DISPATCHERS" OR "HAZARD AREA" 
#                      OR "STNR ALT" OR "STNR ALTITUDE" OR "STATIONARY ALT*" 
#                      OR "TEMPORARY FLIGHT RESTRICTION" OR "TEMPORARY FLT RESTRICTION*" OR "FLIGHT REST*" OR TFR """
####################

exclusion_list  =[ 'obst','obstruct','obstn', 'obstruction',
            'fire', 'unmanned', 'crane', 'uas', 'aerial', 'drill', 'installed',
            'terminal','parking', 'rwy', 'taxi', 'twy', 'hangar', 'chemical',
            'pavement', 'firing', "out of service", 'volcan', 'turbine', 'flare', 'wx',
            'weather','aerodrome', 'apron', 'tower', 'hospital', 'covid', 'medical'
            'copter', 'disabled passenger', 'passanger', 'arctic', 'artic' ,'defense', 'defence'
            'helipad','bird','laser','heliport', 'ordnance','decommisioned','decomissioned'
            'dropzone','runway','wind','aerobatic', 'airfiel', 'model','para', 'parachute' 
            'jumpers','paradrops','glide','tcas','accident','investigation','training' 
            'approach', 'explosion', 'explosive', 'demolition','demolitions']

exclusions = """ NOT (obst* OR fire OR unmanned OR crane OR uas OR aerial OR drill OR installed OR 
                            terminal OR parking OR rwy OR taxi OR twy OR hangar OR chemical OR pavement OR 
                            firing OR "out of service" OR volcan OR turbine OR flare OR wx OR weather OR 
                            aerodrome OR apron OR tower OR hospital OR covid OR medical OR copter OR 
                            disabled OR passenger OR passanger OR arctic OR artic OR defense OR defence OR 
                            helipad OR bird OR laser OR heliport OR ordnance OR decommisioned OR decomissioned OR 
                            dropzone OR runway OR wind OR aerobatic OR airfield OR model  OR 
                            para*  OR parachute OR jumpers OR paradrops OR 
                            glide OR tcas OR accident OR investigation OR training OR 
                            approach OR explosion OR explosive OR demolitions) """
    
# add vehicle name to the initial inclusion_words for this dataset notams
inclusions = """launch OR space OR "91*143" OR "attention airline dispatchers" OR "hazard area" 
            OR "stnr alt" OR "stnr altitude" OR "stationary alt*" 
            OR "temporary flight restriction*" OR "temporary flt rest*" OR "flight rest*" OR tfr 
            OR rocket OR missile OR canaveral OR kennedy OR nasa OR antares OR orion OR atlas
            OR zenit OR falcon OR dragon OR spaceship OR minuteman OR trident """

search1 = """(launch {exclusions})
            OR (space {exclusions}) 
            OR (rocket {exclusions}) OR (missile {exclusions}) OR (canaveral {exclusions}) OR (kennedy {exclusions}) OR (nasa {exclusions} )
            OR (antares {exclusions}) OR (orion {exclusions}) OR (atlas {exclusions}) OR (zenit {exclusions}) OR (falcon {exclusions}) 
            OR (dragon {exclusions}) OR (spaceship {exclusions}) OR (minuteman {exclusions}) OR (trident {exclusions}) 
            """.format(exclusions=exclusions)

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
            OR (tfr {exclusions} ) 
            OR (rocket {exclusions}) OR (missile {exclusions}) OR (canaveral {exclusions}) OR (kennedy {exclusions}) OR (nasa {exclusions} )
            OR (antares {exclusions}) OR (orion {exclusions}) OR (atlas {exclusions}) OR (zenit {exclusions}) OR (falcon {exclusions}) 
            OR (dragon {exclusions}) OR (spaceship {exclusions}) OR (minuteman {exclusions}) OR (trident {exclusions}) 
            """.format(exclusions=exclusions)

def check_good_notams(conn, tfr_notams_df,good_notams_df ):
    # check how many tfr notams in the good_notams_df
    tfr_notams = tfr_notams_df['NOTAM_REC_ID'].to_numpy()
    trf_in_notams = []
    tfr_not_in = []
    for trf in tfr_notams:
        if (good_notams_df['NOTAM_REC_ID'] == trf).any():
            trf_in_notams.append(trf)
        else:
            tfr_not_in.append(trf)
    print(f'trf in  notams len:{len(trf_in_notams)} // {len(tfr_notams_df)}')
    print(f'trf not in len:{len(tfr_not_in)} // {len(tfr_notams_df)} ')
    
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

def filter_out_exclusion_words(notams_df):
    keep_notam_indx =[]
    for indx, notam in notams_df.iterrows():
        e_code = notam['E_CODE_LC']
        e_code_list = e_code.split(' ')
        if any(word in e_code_list for word in exclusion_list):
            continue
        else:
            keep_notam_indx.append(indx)

    # notams_df = notams_df.iloc[keep_notam_indx]
    notams_df = notams_df.filter(items = keep_notam_indx, axis=0)
    return notams_df

def select_notams_with_exclusion_words(notams_df):
    keep_notam_indx =[]
    for indx, notam in notams_df.iterrows():
        e_code = notam['E_CODE_LC']
        e_code_list = e_code.split(' ')
        if any(word in e_code_list for word in exclusion_list):
            keep_notam_indx.append(indx)   

    notams_df = notams_df.filter(items = keep_notam_indx, axis=0)
    return notams_df

def get_active_period(notams_df):
    start = float('inf')
    end = float('-inf')
    start_date_str = ''
    end_date_str = ''
    for _, notam in notams_df.iterrows():
        start_date = notam['POSSIBLE_START_DATE']
        end_date = notam['POSSIBLE_END_DATE']
        start_date_unix = convert_str_datetime_unix_datetime(start_date)
        end_time_unix = convert_str_datetime_unix_datetime(end_date)
        if start_date_unix < start:
            start = start_date_unix
            start_date_str = start_date
        if end_time_unix > end:
            end = end_time_unix
            end_date_str = end_date

    return (start_date_str, end_date_str)

def filter_active_period_notams(notams_step3_df):
    keep_notams_idx = []
    active_start_dates = notams_step3_df['POSSIBLE_START_DATE'].to_numpy()
    active_end_dates = notams_step3_df['POSSIBLE_END_DATE'].to_numpy()
    for indx, notam in notams_step3_df.iterrows():
        if (notam['POSSIBLE_START_DATE'] in active_start_dates) or (notam['POSSIBLE_END_DATE'] in active_end_dates):
            keep_notams_idx.append(indx)

    notams_step3_df= notams_step3_df.filter(items = keep_notams_idx, axis=0)
    return notams_step3_df

def filter_active_notams_during_launch_time(notams_step3_df, launch_time_str):
    launch_time = convert_str_datetime_unix_datetime(launch_time_str)
    keep_notams_idx = []
    for indx, notam in notams_step3_df.iterrows():
        start = convert_str_datetime_unix_datetime(notam['POSSIBLE_START_DATE'])
        end = convert_str_datetime_unix_datetime(notam['POSSIBLE_END_DATE'])
        if start <= launch_time and end >= launch_time:
            keep_notams_idx.append(indx)

    notams_step3_df= notams_step3_df.filter(items = keep_notams_idx, axis=0)
    return notams_step3_df
    
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
def create_good_notams(conn_v, cur_v, tfr_notams_df):
    print(f'create_good_notams')
    results = []
    for _, tfr in tfr_notams_df.iterrows():
        good_notams = []
        launch_rec_id = tfr['LAUNCHES_REC_ID']
        # if launch_rec_id != 347:  # TODO Remove
        #     continue
        start = tfr['POSSIBLE_START_DATE']
        stop = tfr['POSSIBLE_END_DATE']
        print(f"start stop:{start, stop} launch_rec_id:{tfr['LAUNCHES_REC_ID']}")
        
        # step 2.
        sql ="""SELECT NOTAM_REC_ID, MIN_ALT as MIN_ALT_K, MAX_ALT as MAX_ALT_K, 
            POSSIBLE_START_DATE, POSSIBLE_END_DATE, LOCATION_CODE, E_CODE, E_CODE_LC, ACCOUNT_ID, NOTAM_TYPE  FROM virtual_notams 
            WHERE (DATETIME(POSSIBLE_START_DATE) = DATETIME('{start}') 
            or DATETIME(POSSIBLE_END_DATE) = DATETIME('{stop}') )
            and NOTAM_TYPE != 'NOTAMC'
            and E_CODE_LC not null
            and E_CODE_LC MATCH '{q}' 
            """.format(start=start, stop=stop, q=search2)
        notams_step2_df = pd.read_sql_query(sql, conn_v)
        notams_step2_df.insert(0, "LAUNCHES_REC_ID", notams_step2_df.apply(lambda row : launch_rec_id, axis = 1))
        good_notams.append(notams_step2_df)
       
        # step 3. capture the notam active period from_start to_end
        (from_start, to_end) = get_active_period(notams_step2_df)
        print(f'active period:{from_start, to_end}')
        sql ="""SELECT NOTAM_REC_ID, MIN_ALT as MIN_ALT_K, MAX_ALT as MAX_ALT_K, 
            POSSIBLE_START_DATE, POSSIBLE_END_DATE, LOCATION_CODE,ACCOUNT_ID, E_CODE, E_CODE_LC, NOTAM_TYPE FROM virtual_notams 
            WHERE DATETIME(POSSIBLE_START_DATE) >= DATETIME('{start}') 
            and DATETIME(POSSIBLE_END_DATE) <= DATETIME('{stop}')
            and NOTAM_TYPE != 'NOTAMC'
            and E_CODE not null
            and E_CODE not like '%"out of service"%'
            """.format(start=from_start, stop =to_end)

        notams_step3_df = pd.read_sql_query(sql, conn_v)
        notams_step3_df = filter_out_exclusion_words(notams_step3_df)
        notams_step3_df = filter_active_period_notams(notams_step3_df)
        notams_step3_df = filter_active_notams_during_launch_time(notams_step3_df, tfr['LAUNCH_DATE'])
        
        #select the notams published by the same facility ACCOUNT_ID in step2
        account_id_step2_list = notams_step2_df['ACCOUNT_ID'].to_numpy()
        account_id_step2_set = set(account_id_step2_list)
        print(f'account_id_step2_set:{account_id_step2_set}')
        same_account_notams_indx =[]
        for indx, notam in notams_step3_df.iterrows():
            account_id = notam['ACCOUNT_ID']
            if account_id in account_id_step2_set:
                same_account_notams_indx.append(indx)
        notams_step3_df = notams_step3_df.filter(items = same_account_notams_indx, axis=0)

        notams_step3_df.insert(0, "LAUNCHES_REC_ID", notams_step3_df.apply(lambda row : launch_rec_id, axis = 1))
        good_notams.append(notams_step3_df)

        good_notams_df = pd.concat(good_notams)
        good_notams_df = good_notams_df.drop_duplicates(subset=['NOTAM_REC_ID'])
        results.append(good_notams_df)
        
    results_df = pd.concat(results)
    results_df = results_df.drop(columns=['E_CODE_LC'])   
    
    print(f'good_notams len: {len(results_df)}')
    return results_df
    
def create_bad_notams(conn_v, cur_v, tfr_notams_df):
    print(f'create_bad_notams')
    results = []
    for _, tfr in tfr_notams_df.iterrows():
        launch_rec_id = tfr['LAUNCHES_REC_ID']
        start = tfr['POSSIBLE_START_DATE']
        stop = tfr['POSSIBLE_END_DATE']
        print(f"start stop:{start, stop} launch_rec_id:{tfr['LAUNCHES_REC_ID']}")
        
        # step 2.
        sql ="""SELECT NOTAM_REC_ID, MIN_ALT as MIN_ALT_K, MAX_ALT as MAX_ALT_K, 
            POSSIBLE_START_DATE, POSSIBLE_END_DATE, LOCATION_CODE, E_CODE, E_CODE_LC, ACCOUNT_ID, NOTAM_TYPE  FROM virtual_notams 
            WHERE (DATETIME(POSSIBLE_START_DATE) = DATETIME('{start}') 
            or DATETIME(POSSIBLE_END_DATE) = DATETIME('{stop}') )
            and E_CODE_LC not null
            and E_CODE_LC MATCH '{q}' 
            """.format(start=start, stop=stop, q=exclusion_words)
        bad_notams_df = pd.read_sql_query(sql, conn_v)
        bad_notams_df['LAUNCHES_REC_ID'] = launch_rec_id
        new_column  = bad_notams_df.pop('LAUNCHES_REC_ID')
        bad_notams_df.insert(0, 'LAUNCHES_REC_ID', new_column)
        bad_notams_df = bad_notams_df.drop_duplicates(subset=['NOTAM_REC_ID'])
        results.append(bad_notams_df)
        
    results_df = pd.concat(results)
    results_df = results_df.drop(columns=['E_CODE_LC'])    
    
    print(f' bad_notams len:{len(results_df)}')
    print(results_df)
    return results_df
    

def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()
    (conn_v, cur_v) = create_virtual_full_text_search_notam_table(conn, cursor)

    sql = """ select * from launches  """
    launches_df = pd.read_sql_query(sql, conn)
    launches_df["SPACEPORT_REC_ID"] = launches_df["SPACEPORT_REC_ID"].fillna(9999)
    launches_df["SPACEPORT_REC_ID"] = launches_df["SPACEPORT_REC_ID"].astype('int')

    ## Finding TFR
    # tfr_notams = create_tfr_notams(conn_v, cur_v, launches_df)
    # tfr_notams.to_csv(f'./data/tfr_notams.{datetime.now().strftime("%m%d")}.csv', index=False)
    
    tfr_notams_df = pd.read_csv('./data/tfr_notams.csv', engine="python" )

    good_notams_df = create_good_notams(conn_v, cur_v, tfr_notams_df)
    good_notams_df.to_csv(f'./data/possitive_notams.{datetime.now().strftime("%m%d")}.csv', index=False)
    check_good_notams(conn, tfr_notams_df, good_notams_df )
       
    bad_notams_df = create_bad_notams(conn_v, cur_v, tfr_notams_df)
    bad_notams_df.to_csv(f'./data/negative_notams.{datetime.now().strftime("%m%d")}.csv', index=False)
    conn.close()
    

if __name__ == "__main__":
    main()
