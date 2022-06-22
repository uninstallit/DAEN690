import pandas as pd
import sqlite3
from pathlib import Path
from traceback import format_exc
import chardet
import csv

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)


def import_human_annotation_matches_table(conn, file_name):
    try:
        with open(file_name, 'r') as fn:
            lines = fn.readlines()

        df = pd.read_csv(file_name,  encoding='UTF-8', error_bad_lines=False, engine="python", delimiter='|' )
        df.to_sql('human_matches', conn, if_exists='replace', index = False)
        print(f'HumanMatches {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
    except:
        print(format_exc())

def import_human_annotation_poor_matches_table(conn, file_name):
    try:
        with open(file_name, 'r', encoding='latin-1') as fn:
            lines = fn.readlines()
        print(f'count:{len(lines)}')

        df = pd.read_csv(file_name,  encoding='latin-1', error_bad_lines=False, engine="python", sep='\t')
        df.to_sql('human_poor_matches', conn, if_exists='replace', index = False)
        print(f'HumanPoorMatches {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
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
        df.to_sql('external_artcc_boundary', conn, if_exists='replace', index = False)
        print(f'external_artcc_boundary {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
    except:
        print(format_exc())

def import_external_airports_artcc_table(conn, file_name):
    try:
        with open(file_name, 'r') as fn:
            lines = fn.readlines()

        df = pd.read_csv(file_name, engine="python" )
        df.to_sql('external_airports_artcc', conn, if_exists='replace', index = False)
        print(f'external_airports_artcc {file_name} - original count:{len(lines)}, after process count:{df.shape[0]}. Ignored count: {len(lines)-df.shape[0]-1}')
    except:
        print(format_exc())

def import_external_airports_icao_table(conn, file_name):
    try:
        with open(file_name, 'r') as fn:
            lines = fn.readlines()

        df = pd.read_csv(file_name, engine="python" )
        df.to_sql('external_airports_icao', conn, if_exists='replace', index = False)
        print(f'external_airports_icao {file_name} - original count:{len(lines)}, after process count:{df.shape[0]} Ignored count: {len(lines)-df.shape[0]-1}')
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
    data_dir = root + '/data/'
    svo_db = root + '/data/svo_db_20201027.db'
    isExist = os.path.exists(data_dir)
    if not isExist:
        os.makedirs(data_dir)
    file = Path(svo_db)

    file.touch(exist_ok=True)

    conn = sqlite3.connect(svo_db)
    #import_human_annotation_matches_table(conn, data_dir + 'HumanAnnotatedMatches_SVO_DB_20200127_pipes_noquotes.csv')
    import_human_annotation_poor_matches_table(conn, data_dir + 'HumanAnnotatedMatches_poormatches_SVO_DB_20201027.csv')
    # import_vertices_table(conn, data_dir +  'vertices_20220621.csv')
    # import_polygon_table(conn, data_dir + 'polygon_20201027.csv')
    # import_launches_table(conn, data_dir + 'launches_20201027.csv')
    # import_spaceports_table(conn, data_dir + 'spaceports_20201027.csv')
    # import_notams_table(conn, data_dir + 'notam_20201027_pipes_noquotes.csv')
    # import_artcc_boundary_table(conn, data_dir + 'external_artcc_boundary.csv')
    # import_external_airports_artcc_table(conn, data_dir + 'external_airports.csv')
    # import_external_airports_icao_table(conn, data_dir + 'external_airports_database.csv')

    conn.close()

def main():
    #import_sample_data()
    import_svo_data()

if __name__ == "__main__":
    main()

