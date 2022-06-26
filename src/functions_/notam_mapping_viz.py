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

my_connect = sqlite3.Connection('svo_db_20201027.db')
my_cursor = my_connect.cursor()

# centriods = my_cursor.execute("SELECT t1.NOTAM_REC_ID, t1.LATITUDE, t1.LONGITUDE, t1.RADIUS_NM, t2.CLASSIFICATION FROM notam_centroids AS t1 CROSS JOIN notams AS t2 WHERE t1.NOTAM_REC_ID = t2.NOTAM_REC_ID AND LATITUDE < 80 AND LATITUDE > 10 AND LONGITUDE < -50 AND LONGITUDE > -180 LIMIT 250000").fetchall()
centriods = my_cursor.execute("SELECT t1.NOTAM_REC_ID, t1.LATITUDE, t1.LONGITUDE, t1.RADIUS_NM, t2.CLASSIFICATION FROM notam_centroids AS t1 CROSS JOIN notams AS t2 WHERE t1.NOTAM_REC_ID = t2.NOTAM_REC_ID AND LATITUDE < 90 AND LATITUDE > -90 AND LONGITUDE < 180 AND LONGITUDE > -180 LIMIT 250000").fetchall()
# print(centriods)
# print(len(centriods))

df = pd.DataFrame(centriods)
df.columns = ['NOTAM_REC_ID', 'LATITUDE', 'LONGITUDE', 'RADIUS_NM', 'CLASSIFICATION']
# print(df)

geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]
gdf = GeoDataFrame(df, geometry=geometry)

# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# gdf.plot(ax=world[world.continent == 'North America'].plot(figsize=(25, 25)), marker='o', color='b', markersize=1)
# plt.show()

# [world.continent == 'North America']

fig = px.scatter_geo(gdf,
                    # locationmode = 'USA-states',
                    lat=gdf.geometry.y,
                    lon=gdf.geometry.x,
                    hover_name="NOTAM_REC_ID",
                    color="CLASSIFICATION", # which column to use to set the color of markers
                    # size="RADIUS_NM", # size of markers
                    projection="natural earth",
                    title="NOTAM Locations<br>(Hover for NOTAM ID)")
fig.show()