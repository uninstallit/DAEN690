import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import seaborn as sns

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)
sys.path.append(parent)

def plot_notam_data(cursor):
    print(f'plot_classification')
    sql = """ SELECT CLASSIFICATION, NOTAM_TYPE FROM notams """
    data = cursor.execute(sql).fetchall()
    df = pd.DataFrame({ 
    'classification': [d[0] for d in data], 
    'notam_type': [d[1] for d in data]})


    # plotting four plots in one tile 2 x 2
    fig, [[ax1, ax2],[ax3,ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    sns.countplot(data= df, x='classification', ax=ax1)
    ax1.set_title('NOTAM - Classification')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Classification')

    sns.countplot(data= df, x='notam_type', ax=ax2)
    ax2.set_title('NOTAM - Type')
    ax2.set_ylabel('Percent Count')
    ax2.set_xlabel('Type')

    fig.tight_layout()
    plt.show()


def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()
    
    plot_notam_data(cursor)
    
    conn.close()
    

if __name__ == "__main__":
    main()
