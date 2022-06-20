import sqlite3
import pandas as pd


import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)
sys.path.append(parent)  # parent = /Users/tranbre/gmu/01-DAEN-690/DAEN690/src
from functions_.coordinates_utils import calculate_centroid_and_radius

def make_centroid_from_provided_polygons(conn, cursor):
    print(f'make_centroid from the provided polygon and vertices tables')

    sql = """ SELECT polygon_id, notam_rec_id from polygon"""
    polygons = cursor.execute(sql).fetchall()
    polygon_notam = {}
    for p in polygons:
        polygon_notam[p[0]] = p[1]

    sql = """ SELECT POLYGON_ID, LATITUDE, LONGITUDE  from vertices """
    vertices = cursor.execute(sql).fetchall()
    
    polygon_vertices = {}
    for (pol_id, lat, lon) in vertices:
        if pol_id not in polygon_vertices:
            polygon_vertices[pol_id] = [(lat, lon)]
        else:
            polygon_vertices[pol_id].append((lat,lon))

    print(f'len polygon_vertices:{len(polygon_vertices)}')

    items =list(polygon_vertices.items()) #[1:3]   
    notam_centroids = {}
    for pol_id, vertices in items:
        (lat, lon, radius) = calculate_centroid_and_radius(vertices)
        notam_rec_id = polygon_notam[pol_id]
        notam_centroids[notam_rec_id] = (lat, lon, radius)

    lats =[]
    lons =[]
    radiuses =[]
    notam_rec_ids =[]
    for notam_rec_id, (lat, lon, radius) in notam_centroids.items():
       notam_rec_ids.append(notam_rec_id)
       radiuses.append(radius)
       lats.append(lat)
       lons.append(lon)

    assert(len(notam_rec_ids) == len(lats) == len(lons) == len(radiuses))

    df = pd.DataFrame({'NOTAM_REC_ID': notam_rec_ids,
                       'LATITUDE': lats,
                       'LONGITUDE': lons,
                       'RADIUS_NM': radiuses})   # in nautical miles 
    return df

def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()

    df =  make_centroid_from_provided_polygons(conn, cursor)
    df.to_sql('notam_centroids', conn, if_exists='replace', index = False)
    
    conn.close()

if __name__ == "__main__":
    main()