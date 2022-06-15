import sqlite3
import pandas as pd
from geopandas import GeoSeries
from shapely.geometry import Polygon, Point, LineString
from haversine import haversine
from geographiclib.geodesic import Geodesic


def calculateCentroidAndRadius(pol_id, vertices):
    geod = Geodesic.WGS84 

    if len(vertices) == 1:
        return (Point(vertices), 0) # just a point so radius is 0
    
    if len(vertices) == 2:
        l1 = LineString(vertices)
        centroid = l1.centroid
        x_lat = centroid.x
        x_lon = centroid.y
        radius = 0
        for x, y in l1.coords:
            # TODO need to fix distance in NM unit
            g = geod.Inverse(x_lat, x_lon, x, y ) 
            dis = g['s12']/1852.344  # 1852.344 meter = 1NM
            if dis > radius:
                radius = dis
        return (centroid, radius)

    p1 = Polygon(vertices )
    centroid = p1.centroid
    x_lat = centroid.x
    x_lon = centroid.y
    radius = 0
    for x, y in p1.exterior.coords:
        g = geod.Inverse(x_lat, x_lon, x, y ) 
        dis = g['s12']/1852.344  # 1852.344 meter = 1NM
        if dis > radius:
            radius = dis

    return (centroid, radius)

def make_centroid(conn, cursor):
    print(f'make_centroid')

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
        (centroid, radius) = calculateCentroidAndRadius(pol_id, vertices)
        notam_rec_id = polygon_notam[pol_id]
        notam_centroids[notam_rec_id] = (centroid, radius)

    lats =[]
    lons =[]
    radiuses =[]
    notam_rec_ids =[]
    for notam_rec_id, (centroid, radius) in notam_centroids.items():
       notam_rec_ids.append(notam_rec_id)
       radiuses.append(radius)
       lats.append(centroid.x)
       lons.append(centroid.y)

    assert(len(notam_rec_ids) == len(lats) == len(lons) == len(radiuses))

    df = pd.DataFrame({'NOTAM_REC_ID': notam_rec_ids,
                       'LATITUDE': lats,
                       'LONGITUE': lons,
                       'RADIUS_NM': radiuses})   
    print(f"df: {df.head()}")
    df.to_sql('notam_centroids', conn, if_exists='replace', index = False)

def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()
    
    make_centroid(conn, cursor)
    
    conn.close()

if __name__ == "__main__":
    main()