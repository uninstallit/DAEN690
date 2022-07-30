import sqlite3
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(root)

conn = sqlite3.Connection(root + "/data/svo_db_20201027.db")
sql = """ SELECT * FROM notam_centroids"""
centroid_df = pd.read_sql_query(sql, conn)
conn.close()

matches_df = pd.read_csv("./data/team_bravo_mix_matches.0729.csv")
matches_df = pd.merge(matches_df, centroid_df, on="NOTAM_REC_ID")
matches_df["E_CODE"] = matches_df["E_CODE"].apply(lambda text: text[:80])

mapbox_access_token = open("./data/.mapbox_token").read()

# add showcase notams
launches = [284, 391, 466]

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "mapbox"}, {"type": "table"}]],
    horizontal_spacing=0.1,
)

for launch in launches:
    fig.add_trace(
        go.Scattermapbox(
            lat=matches_df[matches_df["LAUNCHES_REC_ID"] == launch]["LATITUDE"],
            lon=matches_df[matches_df["LAUNCHES_REC_ID"] == launch]["LONGITUDE"],
            mode="markers+text",
            marker=go.scattermapbox.Marker(
                size=10,
                color=matches_df[matches_df["LAUNCHES_REC_ID"] == launch]["SCORE"],
                showscale=True,
                cmin=-1,
                cmax=1,
                colorbar_x=0.45,
            ),
            text=matches_df[matches_df["LAUNCHES_REC_ID"] == launch][
                "NOTAM_REC_ID"
            ].to_list(),
            textposition="bottom center",
            name="map",
        ),
        row=1,
        col=1,
    )

for launch in launches:
    fig.add_trace(
        go.Table(
            columnwidth = [23,23, 100],
            header=dict(
                values=["NOTAM_REC_ID", "SCORE", "TEXT"],
                fill_color="#cad2d3",
                align=['center', 'center', 'left']
            ),
            cells=dict(
                values=[
                    matches_df[matches_df["LAUNCHES_REC_ID"] == launch]["NOTAM_REC_ID"],
                    matches_df[matches_df["LAUNCHES_REC_ID"] == launch]["SCORE"],
                    matches_df[matches_df["LAUNCHES_REC_ID"] == launch]["E_CODE"],
                ],
                fill_color="#f3f3f1",
                align=['center', 'center', 'left'],
            ),
        ),
        row=1,
        col=2,
    )


fig.update_layout(
    autosize=True,
    hovermode="closest",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=matches_df["LATITUDE"].mean(),
            lon=matches_df["LONGITUDE"].mean(),
        ),
        pitch=0,
        zoom=5,
    ),
    title_text=f"Matches by Launch Id",
    title_x=0.5,
    showlegend=False,
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            buttons=list(
                [
                    dict(
                        label=launch,
                        method="update",
                        args=[
                            {
                                "visible": 2
                                * [
                                    True if lnch == launch else False
                                    for lnch in launches
                                ]
                            }
                        ],
                    )
                    for launch in launches
                ]
            ),
            pad={"t": 30},
            showactive=True,
            x=0.4,
            xanchor="left",
            y=0,
            yanchor="top",
        ),
    ],
)

fig.update_traces(name="")


fig.show()
