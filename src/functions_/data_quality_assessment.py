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
sys.path.append(parent)
#print('sys.path:', sys.path)

def inspect_csv_notams_bad_records():
    print('\nInspect_csv_notams_bad_records')

    file_name= './data/notam_20201027_pipes_noquotes.csv'
    with open(file_name, 'r', encoding='UTF-16') as fn:
        lines = fn.readlines()
    #print(f'line count:', len(lines))
    bab_count =0
    for line in lines:
        if len(line.split('|')) > 37:
            print(f"NOTAM_REC_ID:{line.split('|')[0]}, FNS_ID:{line.split('|')[1]}, NOTAM_TYPE:{line.split('|')[4]}, NOTAM_ID:{line.split('|')[3]}") 
            bab_count +=1
    
def inspect_time_frame_notams_launches_tables(cursor):
    print('\nInspect_time_frame_notams_launches_tables')
    def getTimeFramDates(data, col):
        # parse str date
        date_format = "%Y-%m-%d %H:%M:%S"
        x = [
            datetime.strptime(date_time_str, date_format)
            for date_time_str in data if date_time_str is not None
        ]
        # convert to unix time
        x = [datetime.timestamp(date_time_obj) for date_time_obj in x]
        pd.Series(x, name=col)
   
        min = float('inf')
        max =float('-inf')
        for t in x:
            if t > max:
                max = t
            if t < min:
                min = t
    
        return (datetime.fromtimestamp(min), datetime.fromtimestamp(max), x)

    sql = """ SELECT "ISSUE_DATE" FROM notams"""
    data = [issue_date[0] for issue_date in cursor.execute(sql).fetchall()]
    (start, end, notams) = getTimeFramDates(data, "ISSUE_DATE")
    print(f'notams dataset issued between: {start}, {end}')

    sql = """ SELECT "LAUNCH_DATE" FROM launches"""
    data = [issue_date[0] for issue_date in cursor.execute(sql).fetchall()]
    (start, end, lauches) = getTimeFramDates(data, "LAUNCH_DATE")
    print(f'lauches dataset issued between: {start}, {end}')
    
    # find number notams that has issued date irrelavent to launches 
    date_format = "%Y-%m-%d %H:%M:%S"
    launch_from ='2013-01-05 00:00:00'
    launch_time = datetime.strptime(launch_from, date_format)
    launch_time = datetime.timestamp(launch_time)
    count_notam_irrelevent_to_launches = 0
    for n in notams:
        if n < launch_time:
            count_notam_irrelevent_to_launches +=1

    print(f'count_notam issued that is irrelevant to launches:{count_notam_irrelevent_to_launches}')

    print(f'Inspect Possible Start >  End Date')
    sql = """ SELECT POSSIBLE_START_DATE, POSSIBLE_END_DATE, NOTAM_REC_ID FROM notams WHERE NOTAM_REC_ID between 1 and 5 """ 
    data = cursor.execute(sql).fetchall()
    date_format = "%Y-%m-%d %H:%M:%S"
    found_bad_notam = []
    for d in data:
        start = datetime.strptime(d[0], date_format)
        end = datetime.strptime(d[1], date_format)
        if start > end:
            found_bad_notam.append(d[2])

    print(f'Number notam has possible start date > possible end date: {len(found_bad_notam)}')

def inspect_notams_polygons_tables(cursor):
    print('\nInspect_notams_polygons_vertices')
    sql = """ SELECT NOTAM_REC_ID FROM notams"""
    notams = [notam_rec_id[0] for notam_rec_id in cursor.execute(sql).fetchall()]

    sql = """ SELECT NOTAM_REC_ID, POLYGON_ID FROM polygon"""
    polygons = cursor.execute(sql).fetchall()
   
    notam_polygons = set()
    for d in polygons:
        notam_id = d[0]
        notam_polygons.add(notam_id)
        
    print(f'set of notam_polygons: {len(notam_polygons)}')
    count_notam_with_no_polygons = 0
    notam_with_no_polygons = []
    for notam in notams:
        if notam not in notam_polygons:
            count_notam_with_no_polygons +=1
            notam_with_no_polygons.append(notam)

    print(f'Found count_notam_with_no_polygons: {count_notam_with_no_polygons},e.i. {notam_with_no_polygons[0], notam_with_no_polygons[1]} ')
    
def inspect_polygons_vertices_tables(cursor):
    print('\nInspect_polygons_vertices')
    sql = """ SELECT POLYGON_ID FROM polygon """
    polygons = [polygons[0] for polygons in cursor.execute(sql).fetchall()]

    sql = """ SELECT POLYGON_ID, VERTICES_REC_ID FROM vertices """
    vertices = cursor.execute(sql).fetchall()

    vertices_polygons = set()
    for v in vertices:
        pol_id= v[0]
        vertices_polygons.add(pol_id)
    print(f'set of vertices_polygons: {len(vertices_polygons)}')

    count_polygon_with_no_vertices = 0
    polygon_with_no_vertices = []
    for p in polygons:
        if p not in vertices_polygons:
            count_polygon_with_no_vertices +=1
            polygon_with_no_vertices.append(p)

    print(f'Found count_polygon_with_no_vertices: {count_polygon_with_no_vertices}')
    

def main():

    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()
    
    inspect_csv_notams_bad_records()
    inspect_time_frame_notams_launches_tables(cursor)
    inspect_notams_polygons_tables(cursor)
    inspect_polygons_vertices_tables(cursor)
    conn.close()
    

if __name__ == "__main__":
    main()
