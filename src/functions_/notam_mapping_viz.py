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

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)
my_connect = sqlite3.Connection('./data/svo_db_20201027.db')
my_cursor = my_connect.cursor()

good_notam = []
with open('./data/good_notams_list.csv', 'r') as my_file:
    my_reader = csv.reader(my_file)
    for row in my_reader:
        good_notam.append(row)

good = """ 
    CREATE TABLE IF NOT EXISTS good_notams (
    NOTAM_REC_ID           INTEGER);
    """
    
my_cursor.execute(good)
my_connect.commit()

my_cursor.execute("DELETE FROM good_notams")
my_cursor.executemany("INSERT INTO good_notams VALUES(?);", good_notam)
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

    spaceport = my_cursor.execute("""SELECT SPACEPORT_REC_ID, LATITUDE, LONGITUDE FROM spaceports WHERE LATITUDE > 0""").fetchall()
    space_df = pd.DataFrame(spaceport)
    space_df.columns = ['REC_ID', 'LATITUDE', 'LONGITUDE']
    space_df['CLASSIFICATION'] = 'SPACEPORT'
    space_df['RADIUS_NM'] = 1
    # print(space_df)

    good_centriods = my_cursor.execute("""SELECT t1.NOTAM_REC_ID, t1.LATITUDE, t1.LONGITUDE, t2.CLASSIFICATION, t1.RADIUS_NM FROM notam_centroids AS t1 CROSS JOIN notams AS t2 CROSS JOIN good_notams AS t3 WHERE t1.NOTAM_REC_ID = t2.NOTAM_REC_ID AND t1.NOTAM_REC_ID = t3.NOTAM_REC_ID AND t1.LATITUDE < 80 AND t1.LATITUDE > 10 AND t1.LONGITUDE < -50 AND t1.LONGITUDE > -180 LIMIT """ + str(G) + """; """).fetchall()

    gooddf = pd.DataFrame(good_centriods)
    gooddf.columns = ['REC_ID', 'LATITUDE', 'LONGITUDE', 'CLASSIFICATION', 'RADIUS_NM']
    
    frames2 = [gooddf, space_df]
    good_df = pd.concat(frames2)
    print(good_df.shape[0])
    print(good_df.head(5))

    good_geometry = [Point(xy) for xy in zip(good_df['LONGITUDE'], good_df['LATITUDE'])]
    good_gdf = GeoDataFrame(good_df, geometry=good_geometry)

    fig3 = px.scatter_geo(good_gdf,
                        lat=good_gdf.geometry.y,
                        lon=good_gdf.geometry.x,
                        hover_name="REC_ID",
                        color="CLASSIFICATION", # which column to use to set the color of markers
                        # size="RADIUS_NM", # size of markers
                        projection="natural earth",
                        title = "NOTAM & Spaceport Locations<br>(Hover for ID)")

    fig3.update_layout(
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
    fig3.write_html('./data/good_notam_map.html')
    return fig3.show()

W = 500000
world_map(W)

U = 1000000
# all_us_map(U)

G = 1000
# good_notam_map(G)


# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# gdf.plot(ax=world[world.continent == 'North America'].plot(figsize=(25, 25)), marker='o', color='b', markersize=1)
# plt.show()