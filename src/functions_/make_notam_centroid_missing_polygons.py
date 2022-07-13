import sqlite3
import pandas as pd

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)
sys.path.append(parent)  # parent = /Users/tranbre/gmu/01-DAEN-690/DAEN690/src
from functions_.coordinates_utils import get_DDcoords

def find_notams_missing_polygons(cursor):
    sql = """ SELECT NOTAM_REC_ID FROM notams WHERE 
              (NOTAM_ID is not null or 
              NOTAM_NUMBER is not null) and
              NOTAM_REC_ID not in (select NOTAM_REC_ID from Polygon )"""
    notam_ids_without_polygon = [notam_rec_id[0] for notam_rec_id in cursor.execute(sql).fetchall()]
    return notam_ids_without_polygon

def select_notam_q_code(cursor):
    sql = """ SELECT NOTAM_REC_ID, Q_CODE FROM notams WHERE 
              NOTAM_ID is not null and 
              NOTAM_NUMBER is not null and 
              Q_CODE is not null and 
              NOTAM_REC_ID not in (select NOTAM_REC_ID from Polygon )"""
    notams = [(notam[0],notam[1]) for notam in cursor.execute(sql).fetchall()]
    return notams

def decode_centroit_from_Qcode(notam_rec_id,q):
    coordinates = ''
    cols = q.split('/')
    if len(cols):
        coordinates = cols[-1]
        coord = get_DDcoords(coordinates)
        return (notam_rec_id, coord)
    return (notam_rec_id,{})

def make_notam_centroids_from_Qcode(conn, cursor):
    notams_q_code = select_notam_q_code(cursor)
    count_valid_q = 0
    invalid_q_code_notams = []
    df = pd.DataFrame(columns = ['NOTAM_REC_ID','LATITUDE','LONGITUDE','RADIUS_NM']) 
    for (notam_rec_id, q) in notams_q_code:
        (notam_rec_id, coord) = decode_centroit_from_Qcode(notam_rec_id, q)  
        if len(coord):
            count_valid_q +=1
            df.loc[len(df.index)] = [notam_rec_id, coord['lat'], coord['lon'], coord['radius'] if coord['radius'] else 0] 
        else:
            invalid_q_code_notams.append(notam_rec_id)
        
    print(f'Found centroids in Q_CODE:{count_valid_q}, not found centroids {len(notams_q_code) -count_valid_q }')

    df.to_sql('notam_centroids', conn, if_exists='append', index = False)
    print(f'Successful wrote Q_CODE NOTAMs centroids to notam_centroids db: {len(df)}')

def make_notam_centroids_from_location_ARTCC(conn, cursor):
    sql =""" select notam_rec_id, location_code, location_name from notams where  
        location_name like '%ARTCC%'  and
        NOTAM_REC_ID not in (select NOTAM_REC_ID from Polygon) and
        NOTAM_REC_ID not in (select NOTAM_REC_ID from notam_centroids)
        """ 
    artcc_notams = [(notam[0], notam[1], notam[2]) for notam in cursor.execute(sql).fetchall()]
    print(f'found artcc notams len: {len(artcc_notams)}')

    sql =""" select facilityId, latitude, longitude, radius_nm from artcc_centroids """
    artcc_centroids = [(artcc[0], artcc[1], artcc[2], artcc[3]) for artcc in cursor.execute(sql).fetchall()]
    
    artcc_centroid_dict = {}
    for (facilityId, lat, lon , radius) in artcc_centroids:
        artcc_centroid_dict[facilityId] = (lat, lon , radius)

    df = pd.DataFrame(columns = ['NOTAM_REC_ID','LATITUDE','LONGITUDE','RADIUS_NM'] )
    location_code_not_found = set()
    found_location_code = 0
    for notam in artcc_notams:
        notam_rec_id, location_name = notam[0],notam[2]
        location_code = location_name[0:3] # ZMP ARTCC
        if location_code in artcc_centroid_dict:
            (lat, lon , radius) = artcc_centroid_dict[location_code]
            df.loc[len(df.index)] = [notam_rec_id, lat, lon, radius] 
            found_location_code +=1
        else:
            location_code_not_found.add(location_code)
    
    print(f'found_location_code:{found_location_code}')
    print(f'location_code_not_found:{len(location_code_not_found)}, {location_code_not_found}')
    df.to_sql('notam_centroids', conn, if_exists='append', index = False)
    print(f'Successful wrote ARTCC NOTAMs centroids to notam_centroids db: {len(df)}')

def make_notam_centroids_from_affected_FIR_US(conn, cursor):
    print(f'make_notam_centroids_from_affected_FIR_US')
    # KZMA, ZMA ==> US ICAO starts with 'K'
    sql = """ select notam_rec_id, AFFECTED_FIR from notams where  
        (AFFECTED_FIR like 'K%' or
        AFFECTED_FIR like 'Z%') and
        NOTAM_REC_ID not in (select NOTAM_REC_ID from Polygon) and
        NOTAM_REC_ID not in (select NOTAM_REC_ID from notam_centroids) """

    affected_fir_notams = [(notam[0], notam[1]) for notam in cursor.execute(sql).fetchall()]
    print(f'found affected_fir KZ notams len: {len(affected_fir_notams)}')

    sql =""" select facilityId, latitude, longitude, radius_nm from artcc_centroids """
    artcc_centroids = [(artcc[0], artcc[1], artcc[2], artcc[3]) for artcc in cursor.execute(sql).fetchall()]
    artcc_centroid_dict = {}
    for (facilityId, lat, lon , radius) in artcc_centroids:
        artcc_centroid_dict[facilityId] = (lat, lon , radius)

    df = pd.DataFrame(columns = ['NOTAM_REC_ID','LATITUDE','LONGITUDE','RADIUS_NM'] )
    location_code_not_found = set()
    found_location_code = 0
    for notam in affected_fir_notams:
        notam_rec_id, affected_fir = notam[0],notam[1]
        location_code = affected_fir
        if affected_fir[0] == 'K':
            location_code = affected_fir[1:4]
        
        if location_code in artcc_centroid_dict:
            (lat, lon , radius) = artcc_centroid_dict[location_code]
            df.loc[len(df.index)] = [notam_rec_id, lat, lon, radius] 
            found_location_code +=1
        else:
            location_code_not_found.add(location_code)
    
    print(f'found_location_code:{found_location_code}')
    print(f'location_code_not_found:{len(location_code_not_found)}, {location_code_not_found}')
    df.to_sql('notam_centroids', conn, if_exists='append', index = False)
    print(f'Successful wrote AFFECTED_FIR(US) NOTAMs centroids to notam_centroids db: {len(df)}')

def make_notam_centroids_from_location_code_FDC(conn, cursor):
    print('make_notam_centroids_from_location_code_FDC')
    # FDC FAA 800 Independence Avenue, SW, Washington, DC 20591
    # 38.886924937762174, -77.02280003895204
    # default FDC location lat/lon
    LAT = 38.886925
    LON = -77.022800
    RADIUS = 0
    sql =""" select notam_rec_id, location_code, location_name from notams where  
        location_code = 'FDC' or location_name = 'FDC' and
        NOTAM_REC_ID not in (select NOTAM_REC_ID from Polygon) and
        NOTAM_REC_ID not in (select NOTAM_REC_ID from notam_centroids) """

    fdc_notams = [(notam[0], notam[1], notam[2]) for notam in cursor.execute(sql).fetchall()]
    print(f'found location_code FDC notams len: {len(fdc_notams)}')      
    df = pd.DataFrame(columns = ['NOTAM_REC_ID','LATITUDE','LONGITUDE','RADIUS_NM'] )
    for notam in fdc_notams:
        notam_rec_id= notam[0]
        df.loc[len(df.index)] = [notam_rec_id, LAT, LON, RADIUS] 

    df.to_sql('notam_centroids', conn, if_exists='append', index = False)
    print(f'Successful wrote location_code FDC NOTAMs centroids to notam_centroids db: {len(df)}')

def make_notam_centroids_from_location_code_Airport(conn, cursor):
    print('make_notam_centroids_from_location_code_Airport')

    sql = """ select notams.NOTAM_REC_ID, notams.location_code, notams.location_name from notams where 
        notams.NOTAM_REC_ID not in (select NOTAM_REC_ID from Polygon) and
        notams.NOTAM_REC_ID not in (select NOTAM_REC_ID from notam_centroids) and 
        (notams.location_code in (select icao_code from external_airports_icao) or 
        notams.location_code in (select iata_code from external_airports_icao)) """

    airport_notams = [(notam[0], notam[1], notam[2]) for notam in cursor.execute(sql).fetchall()]
    print(f'found airport notams len: {len(airport_notams)}')    

    sql=""" select icao_code, iata_code, lat_decimal, lon_decimal from external_airports_icao """
    icao_airports = [(a[0], a[1], a[2], a[3]) for a in cursor.execute(sql).fetchall()]

    icao_code_airport_dict = {}
    iata_code_airport_dict = {}
    for a in icao_airports:
        icao_code, iata_code, lat_decimal, lon_decimal = a[0], a[1], a[2], a[3]
        if icao_code != None and len(icao_code):
            icao_code_airport_dict[icao_code] = (lat_decimal, lon_decimal)
        if iata_code != None and len(iata_code):
            iata_code_airport_dict[iata_code] =  (lat_decimal, lon_decimal)

    location_code_not_found = set()
    found_location_code = 0
    df = pd.DataFrame(columns = ['NOTAM_REC_ID','LATITUDE','LONGITUDE','RADIUS_NM'] )
    for notam in airport_notams:
        notam_rec_id, location_code = notam[0],notam[1]
        if location_code in icao_code_airport_dict:
            (lat, lon) = icao_code_airport_dict[location_code]
            df.loc[len(df.index)] = [notam_rec_id, lat, lon, 0] 
            found_location_code +=1
        elif location_code in iata_code_airport_dict:
            (lat, lon) = iata_code_airport_dict[location_code]
            df.loc[len(df.index)] = [notam_rec_id, lat, lon, 0] 
            found_location_code +=1         
        else:
            location_code_not_found.add(location_code)
    
    assert len(location_code_not_found) == 0
    df.to_sql('notam_centroids', conn, if_exists='append', index = False)
    print(f'Successful wrote airport location_code to notam_centroids db: {len(df)}')
   

def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()
    
    # find centroids for those missing polygon NOTAMs
    # notam_id_missing_polygons = find_notams_missing_polygons(cursor)
    # print(f'Found Notams missing polygons:{len(notam_id_missing_polygons)}')

    # # search rules: Q_Code->AFFECTED_FIR->FDC->AIRPORT->ARTCC
    make_notam_centroids_from_Qcode(conn, cursor)
    make_notam_centroids_from_affected_FIR_US(conn, cursor)
    make_notam_centroids_from_location_code_FDC(conn, cursor)
    make_notam_centroids_from_location_code_Airport(conn, cursor)
    
    # NOTAM REC_ID= 1126008 location_code='ZAN' is a TFR notam but doesn't have a centroid causing balltree failed.
    # Mnually add it to the notam_centroids table
    insert_query = """insert into notam_centroids (NOTAM_REC_ID, LATITUDE , LONGITUDE, RADIUS_NM) VALUES (1126008, 61.1744, -149.996,0) """
    cursor.execute(insert_query)
    conn.commit()

    conn.close()

if __name__ == "__main__":
    main()


