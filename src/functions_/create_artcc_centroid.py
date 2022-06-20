import sqlite3
import pandas as pd

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)
sys.path.append(parent)  # parent = /Users/tranbre/gmu/01-DAEN-690/DAEN690/src
from functions_.coordinates_utils import calculate_centroid_and_radius, get_DDcoords

def make_artcc_centroid(cursor):
    print('make_centroid_from_artcc')
    sql = """ SELECT facilityId, latitude, longitude FROM external_artcc_boundary """
    artccs = [(row[0], row[1], row[2]) for row in cursor.execute(sql).fetchall()]

    # table artccc stores lat lon as str, we need to convert to DD format
    artcc_vertices = {}
    for (facility_id, lat, lon) in artccs:
        latlon_str = lat + ' ' + lon 
        coord = get_DDcoords(latlon_str)
        if facility_id not in artcc_vertices:
            artcc_vertices[facility_id] = [(coord['lat'], coord['lon'])]
        else:
            artcc_vertices[facility_id].append((coord['lat'], coord['lon']))

    print(f'len artcc_vertices:{len(artcc_vertices)}')
    items =list(artcc_vertices.items())

    df = pd.DataFrame(columns = ['FacilityId','Latitude','Longitude','Radius_nm']) 
    for facility_id, vertices in items:
        (lat, lon,  radius) = calculate_centroid_and_radius(vertices)
        df.loc[len(df.index)] = [facility_id, lat, lon, radius] 

    return df

def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()

    df  =  make_artcc_centroid(cursor)
    df.to_sql('artcc_centroids', conn, if_exists='replace', index = False)
    print(f'Successfully wrote {len(df)} rows  to artcc_centroids table')

    conn.close()
    
if __name__ == "__main__":
    main()
