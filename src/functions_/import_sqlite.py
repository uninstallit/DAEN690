import os
import pandas as pd
import sqlite3
from pathlib import Path
from traceback import format_exc
import csv

def import_human_annotation_matching_table(conn, file_name):
    try:
        with open(file_name, 'r') as fn:
            lines = fn.readlines()

        df = pd.read_csv(file_name,  encoding='UTF-8', error_bad_lines=False, engine="python", delimiter='|' )
        df.to_sql('human_matches', conn, if_exists='replace', index = False)
        print(f'HumanMatches {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
    except:
        print(format_exc())

def import_polygon_table (conn, file_name):
    try:
        with open(file_name, 'r') as fn:
            lines = fn.readlines()

        df = pd.read_csv(file_name, engine="python" )
        df.to_sql('polygon', conn, if_exists='replace', index = False)
        print(f'Polygon {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
    except:
        print(format_exc())

def import_vertices_table (conn, file_name):
    try:
        with open(file_name, 'r') as fn:
            lines = fn.readlines()

        df = pd.read_csv(file_name, engine="python" )
        df.to_sql('vertices', conn, if_exists='replace', index = False)
        print(f'Vertices {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
    except:
        print(format_exc())


def import_launches_table (conn, file_name):
    try:
        with open(file_name, 'r') as fn:
            lines = fn.readlines()

        df = pd.read_csv(file_name, engine="python" )
        df.to_sql('launches', conn, if_exists='replace', index = False)
        print(f'LAUNCHES {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
    except:
        print(format_exc())

def import_spaceports_table (conn, file_name):
    try:
        with open(file_name, 'r') as fn:
            lines = fn.readlines()

        df = pd.read_csv(file_name)
        df.to_sql('spaceports', conn, if_exists='replace', index = False)
        print(f'SPACEPORTS {file_name}- original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
    except:
        print(format_exc())
    
def import_notams_table (conn,  file_name):
    try:
        with open(file_name, 'r', encoding='UTF-16') as fn:
            lines = fn.readlines()
        
        # Noquotes, delimiter="|" file
        df = pd.read_csv(file_name, encoding ='UTF-16', error_bad_lines=False, quoting=csv.QUOTE_NONE,  engine="python", delimiter="|")
        
        # delimiter="," file
        #df = pd.read_csv(file_name, encoding ='UTF-16', on_bad_lines='skip',  engine="python", delimiter=",")
        print(f'NOTAM {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
        df.to_sql('notams', conn, if_exists='replace', index = False)
    except:
        print(format_exc())

def import_artcc_boundary_table(conn, file_name):
    try:
        with open(file_name, 'r') as fn:
            lines = fn.readlines()

        df = pd.read_csv(file_name, engine="python" )
        df.to_sql('artcc_boundary', conn, if_exists='replace', index = False)
        print(f'ARTCC_BOUNDARY {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
    except:
        print(format_exc())

def import_airports_table(conn, file_name):
    try:
        with open(file_name, 'r') as fn:
            lines = fn.readlines()

        df = pd.read_csv(file_name, engine="python" )
        df.to_sql('airports', conn, if_exists='replace', index = False)
        print(f'AIRPORTS {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
    except:
        print(format_exc())

def import_sample_data():
    dir = '../../sample_data/'
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
    sample_db = '../../sample_data/sample.db'
    file = Path(sample_db)
    file.touch(exist_ok=True)
    conn = sqlite3.connect(sample_db)
    import_launches_table(conn, '../../sample_data/LAUNCHES_sample.csv')
    import_spaceports_table(conn, '../../sample_data/SPACEPORTS_sample.csv')
    import_notams_table(conn, '../../sample_data/NOTAM_sample.csv')
    conn.close()

def import_svo_data():
    dir = '../../data/'
    svo_db = '../../data/svo_db_20201027.db'
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
    file = Path(svo_db)
    file.touch(exist_ok=True)
    conn = sqlite3.connect(svo_db)
    import_human_annotation_matching_table(conn, '../../data/HumanAnnotatedMatches_SVO_DB_20200127_pipes_noquotes.csv')
    import_vertices_table(conn, '../../data/vertices_20201027.csv')
    import_polygon_table(conn, '../../data/polygon_20201027.csv')
    import_launches_table(conn, '../../data/launches_20201027.csv')
    import_spaceports_table(conn, '../../data/spaceports_20201027.csv')
    import_notams_table(conn, '../../data/notam_20201027_pipes_noquotes.csv')
    import_artcc_boundary_table(conn, '../../data/artcc_boundary.csv')
    import_airports_table(conn, '../../data/airports.csv')
    conn.close()

def main():
    import_sample_data()
    import_svo_data()

if __name__ == "__main__":
    main()

