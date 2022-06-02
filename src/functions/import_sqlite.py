import pandas as pd
import sqlite3
from pathlib import Path
from traceback import format_exc
import csv

def import_launches_table (conn, file_name):
    df = pd.read_csv(file_name)
    df.to_sql('launches', conn, if_exists='replace', index = False)
    
def import_spaceports_table (conn, file_name):
    df = pd.read_csv(file_name)
    df.to_sql('spaceports', conn, if_exists='replace', index = False)
    
def import_notams_table (conn,  file_name):
    
    try:
        with open(file_name, 'r', encoding='UTF-16') as fn:
            lines = fn.readlines()
        
        # bad_line_count = 0
        # for line in lines:
        #     if len(line.split(','))> 37:
        #         bad_line_count += 1
        # print(f'bad_line_count:{bad_line_count}')
        
        df = pd.read_csv(file_name, encoding ='UTF-16', error_bad_lines=False,  quoting=csv.QUOTE_NONE,  engine="python", delimiter="|")
        print('Original record count:', len(lines))
        print('After process, count:', df.shape[0])
        print(f'Original record count:{len(lines)}, After process, count:{df.shape[0]}. Threw away count: {len(lines)-df.shape[0]-1}')
        df.to_sql('notams', conn, if_exists='replace', index = False)
    except:
        print(format_exc())
        
def import_sample_data():
    sample_db = '../../sample_data/sample.db'
    file = Path(sample_db)
    file.touch(exist_ok=True)
    conn = sqlite3.connect(sample_db)
    import_launches_table(conn, '../../sample_data/LAUNCHES_sample.csv')
    import_spaceports_table(conn, '../../sample_data/SPACEPORTS_sample.csv')
    import_notams_table(conn, '../../sample_data/NOTAM_sample.csv')
    conn.close()

def import_svo_data():
    svo_db = '../../data/svo_db_20200901.db'
    file = Path(svo_db)
    file.touch(exist_ok=True)
    conn = sqlite3.connect(svo_db)
    import_launches_table(conn, '../../data/launches.csv')
    import_spaceports_table(conn, '../../data/spaceports.csv')
    import_notams_table(conn, '../../data/notams_v3.csv')
    conn.close()

# import_sample_data()
import_svo_data()																																																																																																																							

