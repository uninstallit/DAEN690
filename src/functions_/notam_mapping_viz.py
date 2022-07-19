# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 21:20:46 2022

@author: Andrew
"""
import csv
import sqlite3
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly
import datetime

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)

my_connect = sqlite3.Connection('./data/svo_db_20201027.db')
my_cursor = my_connect.cursor()

good_notam = []
with open('./data/good_notams_list_2022.06.28.csv', 'r') as my_file:
    my_reader = csv.reader(my_file)
    for row in my_reader:
        good_notam.append(row)

good = """ 
    CREATE TABLE IF NOT EXISTS good_notams (
    LAUNCH_REC_ID           INTEGER,
    NOTAM_REC_ID            INTEGER,
    SPACEPORT_REC_ID        INTEGER);
    """
    
my_cursor.execute(good)
my_connect.commit()

my_cursor.execute("DELETE FROM good_notams")
my_cursor.executemany("INSERT INTO good_notams VALUES(?,?,?);", good_notam)
my_connect.commit()

tfr_notams = []
with open('./data/tfr_notams.0704.csv', 'r') as my_file:
    my_reader = csv.reader(my_file)
    next(my_reader)
    for row in my_reader:
        tfr_notams.append(row)

tfr = """ 
    CREATE TABLE IF NOT EXISTS tfr_notams (
    LAUNCH_REC_ID           INTEGER,
    NOTAM_REC_ID            INTEGER,
    MIN_ALT                 INTEGER,
    MAX_ALT                 INTEGER,
    ISSUE_DATE              DATETIME,
    START_DATE              DATETIME,
    END_DATE                DATETIME,
    LOCATION_CODE           TEXT,
    E_CODE                  TEXT,
    LAUNCH_CITY             TEXT,
    LAUNCH_STATE            TEXT);
    """
    
my_cursor.execute(tfr)
my_connect.commit()

my_cursor.execute("DELETE FROM tfr_notams")
my_cursor.executemany("INSERT INTO tfr_notams VALUES(?,?,?,?,?,?,?,?,?,?,?);", tfr_notams)
my_connect.commit()

semantic_matches = []
with open('./data/2_FL_semantic_matches_spaceport_2_Cape_FL.csv', 'r') as my_file:
    my_reader = csv.reader(my_file)
    next(my_reader)
    for row in my_reader:
        semantic_matches.append(row)

matches = """ 
    CREATE TABLE IF NOT EXISTS matches (
    LAUNCH_REC_ID           INTEGER,
    NOTAM_REC_ID            INTEGER,
    SCORE                   REAL,
    LAUNCH_DATE             DATETIME,
    START_DATE              DATETIME,
    END_DATE                DATETIME,
    E_CODE                  TEXT,
    LOCATION_CODE           TEXT,
    ACCOUNT_ID              TEXT,
    TFR_FLAG                INTEGER,
    MIN_ALT                 INTEGER,
    MAX_ALT                 INTEGER);
    """
    
my_cursor.execute(matches)
my_connect.commit()

my_cursor.execute("DELETE FROM matches")
my_cursor.executemany("INSERT INTO matches VALUES(?,?,?,?,?,?,?,?,?,?,?,?);", semantic_matches)
my_connect.commit()

def world_map(W):

    spaceport = my_cursor.execute("""SELECT SPACEPORT_REC_ID, LATITUDE, LONGITUDE FROM spaceports WHERE LATITUDE > 0""").fetchall()
    space_df = pd.DataFrame(spaceport)
    space_df.columns = ['REC_ID', 'LATITUDE', 'LONGITUDE']
    space_df['CLASSIFICATION'] = 'SPACEPORT'
    space_df['RADIUS_NM'] = 1
    # print(space_df)

    centriods = my_cursor.execute("""SELECT t1.NOTAM_REC_ID, t1.LATITUDE, t1.LONGITUDE, t2.CLASSIFICATION, t1.RADIUS_NM FROM notam_centroids AS t1 CROSS JOIN notams AS t2 WHERE t1.NOTAM_REC_ID = t2.NOTAM_REC_ID AND LATITUDE < 90 AND LATITUDE > -90 AND LONGITUDE < 180 AND LONGITUDE > -180 LIMIT """ + str(W) + """; """).fetchall()
    # print(centriods)
    # print(len(centriods))

    df = pd.DataFrame(centriods)
    df.columns = ['REC_ID', 'LATITUDE', 'LONGITUDE', 'CLASSIFICATION', 'RADIUS_NM']

    frames = [df, space_df]
    w_df = pd.concat(frames)

    w_geometry = [Point(xy) for xy in zip(w_df['LONGITUDE'], w_df['LATITUDE'])]
    w_gdf = GeoDataFrame(w_df, geometry=w_geometry)
    print(w_df.shape[0])

    fig = px.scatter_geo(w_gdf,
                        lat=w_gdf.geometry.y,
                        lon=w_gdf.geometry.x,
                        hover_name="REC_ID",
                        color="CLASSIFICATION", # which column to use to set the color of markers
                        # size="RADIUS_NM", # size of markers
                        projection="natural earth",
                        title = "NOTAM & Spaceport Locations<br>(Hover for ID)")
    fig.write_html('./data/world_notam_map.html')
    return fig.show()

def all_us_map(U):

    spaceport = my_cursor.execute("""SELECT SPACEPORT_REC_ID, LATITUDE, LONGITUDE FROM spaceports WHERE LATITUDE > 0""").fetchall()
    space_df = pd.DataFrame(spaceport)
    space_df.columns = ['REC_ID', 'LATITUDE', 'LONGITUDE']
    space_df['CLASSIFICATION'] = 'SPACEPORT'
    space_df['RADIUS_NM'] = 1
    # print(space_df)

    us_centriods = my_cursor.execute("""SELECT t1.NOTAM_REC_ID, t1.LATITUDE, t1.LONGITUDE, t2.CLASSIFICATION, t1.RADIUS_NM FROM notam_centroids AS t1 CROSS JOIN notams AS t2 WHERE t1.NOTAM_REC_ID = t2.NOTAM_REC_ID AND LATITUDE < 80 AND LATITUDE > 10 AND LONGITUDE < -50 AND LONGITUDE > -180 LIMIT """ + str(U) + """; """).fetchall()

    usdf = pd.DataFrame(us_centriods)
    usdf.columns = ['REC_ID', 'LATITUDE', 'LONGITUDE', 'CLASSIFICATION', 'RADIUS_NM']
    
    frames2 = [usdf, space_df]
    us_df = pd.concat(frames2)
    print((us_df.shape[0]))

    us_geometry = [Point(xy) for xy in zip(us_df['LONGITUDE'], us_df['LATITUDE'])]
    us_gdf = GeoDataFrame(us_df, geometry=us_geometry)

    fig2 = px.scatter_geo(us_gdf,
                        lat=us_gdf.geometry.y,
                        lon=us_gdf.geometry.x,
                        hover_name="REC_ID",
                        color="CLASSIFICATION", # which column to use to set the color of markers
                        # size="RADIUS_NM", # size of markers
                        projection="natural earth",
                        title = "NOTAM & Spaceport Locations<br>(Hover for ID)")

    fig2.update_layout(
                    title = "NOTAM & Spaceport Locations<br>(Hover for ID)",
                    geo = dict(
                        scope='usa',
                        projection_type='albers usa',
                        showland = True,
                        landcolor = "rgb(250, 250, 250)",
                        subunitcolor = "rgb(217, 217, 217)",
                        countrycolor = "rgb(217, 217, 217)",
                        countrywidth = 0.5,
                        subunitwidth = 0.5))
    fig2.write_html('./data/us_notam_map.html')
    return fig2.show()

def good_notam_map(G):

    spaceport = my_cursor.execute("""SELECT SPACEPORT_REC_ID, SPACEPORT_NAME, LATITUDE, LONGITUDE FROM spaceports WHERE LATITUDE > 0""").fetchall()
    space_df = pd.DataFrame(spaceport)
    space_df.columns = ['REC_ID_1', 'REC_ID_2', 'LATITUDE', 'LONGITUDE']
    space_df['CLASSIFICATION'] = 'SPACEPORT'
    space_df['RADIUS_NM'] = 1
    space_df['REC_ID'] = list(zip(space_df.REC_ID_1, space_df.REC_ID_2))
    # print(space_df)

    good_centriods = my_cursor.execute("""SELECT t3.LAUNCH_REC_ID, t1.NOTAM_REC_ID, t3.SPACEPORT_REC_ID, t1.LATITUDE, t1.LONGITUDE, t2.CLASSIFICATION, t1.RADIUS_NM FROM notam_centroids AS t1 CROSS JOIN notams AS t2 CROSS JOIN good_notams AS t3 WHERE t1.NOTAM_REC_ID = t2.NOTAM_REC_ID AND t1.NOTAM_REC_ID = t3.NOTAM_REC_ID AND t1.LATITUDE < 80 AND t1.LATITUDE > 10 AND t1.LONGITUDE < -50 AND t1.LONGITUDE > -180 LIMIT """ + str(G) + """; """).fetchall()

    gooddf = pd.DataFrame(good_centriods)
    gooddf.columns = ['REC_ID_1', 'REC_ID_2', 'REC_ID_3', 'LATITUDE', 'LONGITUDE', 'CLASSIFICATION', 'RADIUS_NM']
    gooddf['REC_ID'] = list(zip(gooddf.REC_ID_1, gooddf.REC_ID_2, gooddf.REC_ID_3))
    frames2 = [gooddf, space_df]
    good_df = pd.concat(frames2)
    print(good_df.shape[0])
    print(good_df.head(5))

    good_geometry = [Point(xy) for xy in zip(good_df['LONGITUDE'], good_df['LATITUDE'])]
    good_gdf = GeoDataFrame(good_df, geometry=good_geometry)

    fig3 = px.scatter_geo(good_gdf,
                        lat=good_gdf.geometry.y,
                        lon=good_gdf.geometry.x,
                        hover_name='REC_ID',
                        color="CLASSIFICATION", # which column to use to set the color of markers
                        # size="RADIUS_NM", # size of markers
                        projection="natural earth",
                        title = "Launch & NOTAM Match Locations<br>(Hover for IDs (Launch_ID, NOTAM_ID, Spaceport_ID))")

    fig3.update_layout(
                    title = "Launch & NOTAM Match Locations<br>(Hover for IDs (Launch_ID, NOTAM_ID, Spaceport_ID))",
                    geo = dict(
                        scope='usa',
                        projection_type='albers usa',
                        showland = True,
                        landcolor = "rgb(250, 250, 250)",
                        subunitcolor = "rgb(217, 217, 217)",
                        countrycolor = "rgb(217, 217, 217)",
                        countrywidth = 0.5,
                        subunitwidth = 0.5))
    fig3.write_html('./data/good_notam_map.html')
    return fig3.show()

def tfr_notam_map(T):
    
    spaceport = my_cursor.execute("""SELECT SPACEPORT_REC_ID, SPACEPORT_NAME, LATITUDE, LONGITUDE FROM spaceports WHERE LATITUDE > 0""").fetchall()
    space_df = pd.DataFrame(spaceport)
    space_df.columns = ['REC_ID_1', 'SPACEPORT', 'LATITUDE', 'LONGITUDE']
    space_df['CLASSIFICATION'] = 'SPACEPORT'
    space_df['RADIUS_NM'] = 1
    space_df['REC_ID'] = list(zip(space_df.REC_ID_1, space_df.SPACEPORT))
    # print(space_df.head(3))

    tfr_query = my_cursor.execute("""SELECT t3.LAUNCH_REC_ID, t3.NOTAM_REC_ID, t4.SPACEPORT_REC_ID, t1.LATITUDE, t1.LONGITUDE, t2.CLASSIFICATION, t1.RADIUS_NM, t4.SPACEPORT_NAME, t2.E_CODE FROM notam_centroids AS t1 CROSS JOIN notams AS t2 CROSS JOIN tfr_notams AS t3 CROSS JOIN spaceports AS t4 WHERE t3.NOTAM_REC_ID = t1.NOTAM_REC_ID AND t3.NOTAM_REC_ID = t2.NOTAM_REC_ID AND t1.NOTAM_REC_ID = t2.NOTAM_REC_ID AND t3.LAUNCH_CITY = t4.LOCATION_1 AND t1.LATITUDE < 80 AND t1.LATITUDE > 10 AND t1.LONGITUDE < -50 AND t1.LONGITUDE > -180 LIMIT """ + str(T) + """; """).fetchall()

    tfrdf = pd.DataFrame(tfr_query)
    tfrdf.columns = ['REC_ID_1', 'REC_ID_2', 'REC_ID_3', 'LATITUDE', 'LONGITUDE', 'CLASSIFICATION', 'RADIUS_NM', 'SPACEPORT', 'E_CODE']
    tfrdf['REC_ID'] = list(zip(tfrdf.REC_ID_1, tfrdf.REC_ID_2, tfrdf.REC_ID_3))
    tfrdf['MATCHES'] = tfrdf['E_CODE'].apply(lambda text: text[:100])
    frames2 = [tfrdf, space_df]
    tfr_df = pd.concat(frames2)
    print(tfr_df.shape[0])
    # print(tfr_df.head(3))

    tfr_geometry = [Point(xy) for xy in zip(tfr_df['LONGITUDE'], tfr_df['LATITUDE'])]
    tfr_gdf = GeoDataFrame(tfr_df, geometry=tfr_geometry)

    fig4 = px.scatter_geo(tfr_gdf,
                        lat=tfr_gdf.geometry.y,
                        lon=tfr_gdf.geometry.x,
                        hover_name='REC_ID',
                        color="MATCHES", # which column to use to set the color of markers
                        # size="RADIUS_NM", # size of markers
                        projection="natural earth",
                        title = "<b>Temporary Flight Restrictions by Spaceport</b><br>Hover for IDs (Launch_ID, NOTAM_ID, Spaceport_ID)")

    fig4.update_layout(
                    title = "<b>Temporary Flight Restrictions by Spaceport</b><br>Hover for IDs (Launch_ID, NOTAM_ID, Spaceport_ID)",
                    geo = dict(
                        scope='usa',
                        projection_type='albers usa',
                        showland = True,
                        landcolor = "rgb(250, 250, 250)",
                        subunitcolor = "rgb(217, 217, 217)",
                        countrycolor = "rgb(217, 217, 217)",
                        countrywidth = 0.5,
                        subunitwidth = 0.5))
    fig4.write_html('./data/tfr_notam_map.html')
    return fig4.show()

def tfr_year_map(my_connect,  my_cursor):
    sql = """ select t1.LAUNCH_REC_ID, t1.NOTAM_REC_ID, t1.START_DATE, t2.LATITUDE, t2.LONGITUDE FROM tfr_notams AS t1 CROSS JOIN notam_centroids AS t2 WHERE t1.NOTAM_REC_ID = t2.NOTAM_REC_ID """
    launches_df = pd.read_sql_query(sql, my_connect)
    launches_df.columns = ['REC_ID_1', 'REC_ID_2', 'START_DATE', 'LATITUDE', 'LONGITUDE']

    # reformat Launch datetime format 2013-01-26 22:00:00 to Year 2013
    launches_df['LAUNCH_YEAR'] = [ datetime.datetime.strptime(x, "%m/%d/%y %H:%M").strftime("%Y") for x in launches_df['START_DATE'] ]
    launches_df['REC_ID'] = list(zip(launches_df.REC_ID_1, launches_df.REC_ID_2))

    launch_tfr_geometry = [Point(xy) for xy in zip(launches_df['LONGITUDE'], launches_df['LATITUDE'])]
    launch_tfr_gdf = GeoDataFrame(launches_df, geometry=launch_tfr_geometry)
    
    fig5 = px.scatter_geo(launch_tfr_gdf,
                        lat=launch_tfr_gdf.geometry.y,
                        lon=launch_tfr_gdf.geometry.x,
                        hover_name='REC_ID',
                        color="LAUNCH_YEAR", # which column to use to set the color of markers
                        # size="RADIUS_NM", # size of markers
                        projection="natural earth",
                        title = "Launch & NOTAM TFR Match by Year<br>(Hover for IDs (Launch_ID, NOTAM_ID))")

    fig5.update_layout(
                    title = "Launch & NOTAM TFR Match by Year<br>(Hover for IDs (Launch_ID, NOTAM_ID))",
                    geo = dict(
                        scope='usa',
                        projection_type='albers usa',
                        showland = True,
                        landcolor = "rgb(250, 250, 250)",
                        subunitcolor = "rgb(217, 217, 217)",
                        countrycolor = "rgb(217, 217, 217)",
                        countrywidth = 0.5,
                        subunitwidth = 0.5))
    fig5.write_html('./data/tfr_year_map.html')
    return fig5.show()

def notam_matches_map():

    index_query = my_cursor.execute(""" SELECT DISTINCT LAUNCH_REC_ID from matches """).fetchall()
    index = [x[0] for x in index_query]
    # print(index)
    
    for i in index:
        if i != 391:
            continue
        launch_query = my_cursor.execute(""" SELECT t1.LAUNCHES_REC_ID, t1.VEHICLE_NAME, t2.LATITUDE, t2.LONGITUDE from launches AS t1 CROSS JOIN spaceports AS t2 WHERE t1.SPACEPORT_REC_ID = t2.SPACEPORT_REC_ID AND t1.LAUNCHES_REC_ID = """ + str(i) + """; """).fetchall()

        launchdf = pd.DataFrame(launch_query)
        launchdf.columns = ['REC_ID', 'VEHICLE_NAME', 'LATITUDE', 'LONGITUDE']
        launchdf['CLASSIFICATION'] = 'LAUNCH'
        launchdf['HOVER'] = list(zip(launchdf.REC_ID, launchdf.VEHICLE_NAME))

        matches_query = my_cursor.execute(""" select t1.LAUNCH_REC_ID, t1.NOTAM_REC_ID, t1.SCORE, t2.LATITUDE, t2.LONGITUDE, t1.E_CODE, t1.TFR_FLAG FROM matches AS t1 CROSS JOIN notam_centroids AS t2 WHERE t1.NOTAM_REC_ID = t2.NOTAM_REC_ID AND t1.LAUNCH_REC_ID = """ + str(i) + """; """)
        matchdf = pd.DataFrame(matches_query)
        matchdf.columns = ['LAUNCH_REC_ID', 'NOTAM_REC_ID', 'SCORE', 'LATITUDE', 'LONGITUDE', 'E_CODE', 'TFR_FLAG']
        matchdf['E_CODE'] = matchdf['E_CODE'].apply(lambda text: text[:100])
        matchdf['HOVER'] = list(zip(matchdf.NOTAM_REC_ID, matchdf.SCORE, matchdf.E_CODE))
        matchdf['CLASSIFICATION'] = 'MATCH'

        tfr_row_index = 999
        for index, row in matchdf.iterrows():
            if row['TFR_FLAG'] == 1:
                tfr_row_index = index
                matchdf.loc[index, 'CLASSIFICATION'] = 'TFR'
            else:
                matchdf.loc[index, 'CLASSIFICATION'] = 'MATCH'
        
        for index, row in matchdf.iterrows():
            if index != tfr_row_index:
                if row['SCORE'] >= .9 and row['CLASSIFICATION'] == 'MATCH':
                    matchdf.loc[index, 'CLASSIFICATION'] = 'GOOD MATCH'
                else:
                    matchdf.loc[index, 'CLASSIFICATION'] = 'POOR MATCH'


        frames2 = [matchdf, launchdf]
        match_df = pd.concat(frames2)
        # print(match_df.shape[0])
        # print(match_df)

        match_geometry = [Point(xy) for xy in zip(match_df['LONGITUDE'], match_df['LATITUDE'])]
        match_gdf = GeoDataFrame(match_df, geometry=match_geometry)

        fig6 = px.scatter_geo(match_gdf,
                            lat=match_gdf.geometry.y,
                            lon=match_gdf.geometry.x,
                            hover_name='HOVER',
                            color="CLASSIFICATION", # which column to use to set the color of markers
                            # size="RADIUS_NM", # size of markers
                            projection="natural earth",
                            title = "<b>NOTAM Matches for Launch {0}</b><br>Hover for NOTAM information".format(i)
                            )

        fig6.update_layout(
                        title = "<b>NOTAM Matches for Launch {0}</b><br>Hover for NOTAM information".format(i),
                        geo = dict(
                            scope='usa',
                            projection_type='albers usa',
                            showland = True,
                            landcolor = "rgb(250, 250, 250)",
                            subunitcolor = "rgb(217, 217, 217)",
                            countrycolor = "rgb(217, 217, 217)",
                            countrywidth = 0.5,
                            subunitwidth = 0.5))
        
        plotly.offline.plot(fig6, filename='./maps/semantic_match_for_launch_{0}.html'.format(i)) 
    return

def main():
    my_connect = sqlite3.Connection("./data/svo_db_20201027.db")
    my_cursor = my_connect.cursor()
    
    W = 500000
    # world_map(W)

    U = 1000000
    # all_us_map(U)

    G = 1000
    # good_notam_map(G)

    T = 1000
    # tfr_notam_map(T)

    notam_matches_map()

    # tfr_year_map(my_connect,  my_cursor)
    
    my_connect.close()

if __name__ == "__main__":
    main()