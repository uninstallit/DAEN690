import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
import datetime


import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)
sys.path.append(parent)

def plot_classification_and_type(cursor):
    print(f'plot_classification_and_type')
    sql = """ SELECT CLASSIFICATION, NOTAM_TYPE FROM notams 
            WHERE (LOCATION_CODE like 'Z%' or LOCATION_CODE like 'K%' or LOCATION_NAME like '%ARTCC%' or AFFECTED_FIR like 'Z%' or AFFECTED_FIR like 'K%' 
            or LOCATION_CODE in ('PHNL','PHZH') or LOCATION_CODE in ('PHNL','PHZH') or AFFECTED_FIR in ('PHNL','PHZH') )"""
    data = cursor.execute(sql).fetchall()
    df = pd.DataFrame({ 
    'classification': [d[0] for d in data], 
    'notam_type': [d[1] for d in data]})


    # plotting four plots in one tile 2 x 2
    # fig, [[ax1, ax2],[ax3,ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    sns.countplot(data= df, x='classification', ax=ax1)
    ax1.set_title('NOTAM by Classification')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('NOTAM by Classification')

    sns.countplot(data= df, x='notam_type', ax=ax2)
    ax2.set_title('NOTAM by Type')
    ax2.set_ylabel('Percent Count')
    ax2.set_xlabel('NOTAM by Type')

    fig.tight_layout()
    plt.show()

def plot_us_classification(cursor):
    print(f'plot_us_classification')
    sql = """ SELECT CLASSIFICATION, NOTAM_TYPE FROM notams
              WHERE (LOCATION_CODE like 'Z%' or LOCATION_CODE like 'K%' or LOCATION_NAME like '%ARTCC%' or AFFECTED_FIR like 'Z%' or AFFECTED_FIR like 'K%' 
                or LOCATION_CODE in ('PHNL','PHZH') or LOCATION_CODE in ('PHNL','PHZH') or AFFECTED_FIR in ('PHNL','PHZH') ) """
    data = cursor.execute(sql).fetchall()
    df = pd.DataFrame({ 
    'classification': [d[0] for d in data], 
    'notam_type': [d[1] for d in data]})


    # plotting four plots in one tile 2 x 2
    # fig, [[ax1, ax2],[ax3,ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    sns.countplot(data= df, x='classification', ax=ax1)
    ax1.set_title('Count US NOTAM by Classification')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Classification')
    fig.tight_layout()
    plt.show()

def plot_us_notam_type(cursor):
    print(f'plot_us_notam_type')
    sql = """ SELECT CLASSIFICATION, NOTAM_TYPE FROM notams """
    data = cursor.execute(sql).fetchall()
    df = pd.DataFrame({ 
    'classification': [d[0] for d in data], 
    'notam_type': [d[1] for d in data]})

    # plotting four plots in one tile 2 x 2
    # fig, [[ax1, ax2],[ax3,ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    sns.countplot(data= df, x='notam_type', ax=ax1)
    ax1.set_title('Percentage of US NOTAM by Type')
    ax1.set_ylabel('Percent')
    ax1.set_xlabel('Type')
    fig.tight_layout()
    plt.show()

def plot_notam_missing_polygons(cursor):

    sql = """ SELECT NOTAM_REC_ID, CLASSIFICATION, NOTAM_TYPE FROM notams"""
    notams = cursor.execute(sql).fetchall()

    sql = """ SELECT NOTAM_REC_ID, POLYGON_ID FROM polygon"""
    polygons = cursor.execute(sql).fetchall()
    
    notam_polygons = set()
    for (notam_id, polygon_id) in polygons:
        notam_polygons.add(notam_id)
        
    print(f'set of notam_polygons: {len(notam_polygons)}')

    count_notam_with_no_polygons = 0
    notam_with_no_polygons = []
    for notam_rec_id, classification, notam_type in notams:
        if notam_rec_id not in notam_polygons:
            count_notam_with_no_polygons +=1
            notam_with_no_polygons.append((notam_rec_id,classification, notam_type))

    print(f'Found count_notam_with_no_polygons: {count_notam_with_no_polygons}')
    df = pd.DataFrame({ 
    'classification': [d[1] for d in notam_with_no_polygons], 
    'notam_type': [d[2] for d in notam_with_no_polygons]})


    # plotting four plots in one tile 2 x 2
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    sns.countplot(data= df, x='classification', ax=ax1)
    ax1.set_title('Polygons Not Available by Classification')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('NOTAM by Classification')

    sns.countplot(data= df, x='notam_type', ax=ax2)
    ax2.set_title('Polygons Not Available by Type')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('NOTAM by Type')

    fig.tight_layout()
    plt.show()
 
def plot_launch_data(conn,  cursor):
    sql = """ select LAUNCHES_REC_ID, LAUNCH_DATE from launches """
    launches_df = pd.read_sql_query(sql, conn)

    # reformat Launch datetime format 2013-01-26 22:00:00 to Year 2013
    launches_df['LAUNCH_YEAR'] = [ datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%Y") for x in launches_df['LAUNCH_DATE'] ]

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    sns.countplot(data= launches_df, x='LAUNCH_YEAR', ax=ax1)
    ax1.set_title('LAUNCHES by YEAR')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('LAUNCHES')
    fig.tight_layout()
    plt.show()

def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()
    
    #plot_notam_data(cursor)
    #plot_notam_missing_polygons(cursor)
    # plot_classification_and_type(cursor)
    plot_us_classification(cursor)
    #plot_us_notam_type(cursor)
    #plot_launch_data(conn,  cursor)
    
    conn.close()
    

if __name__ == "__main__":
    main()
