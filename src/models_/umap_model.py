import sqlite3
import pandas as pd
import numpy as np
import umap
import umap.plot

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(root)

from pipelines_.pipelines import clean_column_text_pipeline
from functions_.functions import get_matches_index_dict


# source: https://umap-learn.readthedocs.io/en/latest/document_embedding.html#using-tf-idf

# Potential applications
# Now that we have an embedding, what can we do with it?

# Explore/visualize your corpus to identify topics/trends
# Cluster the embedding to find groups of related documents
# Look for nearest neighbours to find related documents
# Look for anomalous documents


def main():

    conn = sqlite3.Connection("./data/svo_db_20201027.db")

    sql = """ SELECT * FROM notams"""
    notam_df = pd.read_sql_query(sql, conn)

    sql = """ SELECT * FROM human_matches"""
    matches_df = pd.read_sql_query(sql, conn)

    sql = """ SELECT * FROM launches"""
    launches_df = pd.read_sql_query(sql, conn)

    # uncomment if less is desired
    # notam_df = notam_df.head(100000)

    matching_notam_rec_ids_dict = get_matches_index_dict(matches_df, launches_df)
    matching_notam_groups = list(matching_notam_rec_ids_dict.values())
    matching_notam_rec_ids = [item for group in matching_notam_groups for item in group]

    matching_notam_idx = notam_df[
        notam_df["NOTAM_REC_ID"].isin(matching_notam_rec_ids)
    ].index

    classes = notam_df["CLASSIFICATION"].unique().tolist()

    clean_text = clean_column_text_pipeline("E_CODE").fit_transform(notam_df)
    clean_text = np.squeeze(clean_text, axis=1)
    notam_df["E_CODE"] = clean_text

    tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words="english")
    tfidf_word_doc_matrix = tfidf_vectorizer.fit_transform(notam_df["E_CODE"].tolist())
    tfidf_embedding = umap.UMAP(metric="hellinger", n_neighbors=15).fit(
        tfidf_word_doc_matrix
    )

    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)

    n = len(classes)
    bounds = np.linspace(0, n, n + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    x = tfidf_embedding.embedding_[:, 0]
    y = tfidf_embedding.embedding_[:, 1]

    x_prime = np.take(x, matching_notam_idx, axis=0).astype(np.int32)
    y_prime = np.take(y, matching_notam_idx, axis=0).astype(np.int32)

    labels = notam_df["CLASSIFICATION"].tolist()
    label_map = {cls: index for index, cls in enumerate(classes)}
    tag = [label_map[label] for label in labels]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    scat1 = ax.scatter(x, y, c=tag, s=1, cmap=cmap, norm=norm)
    scat2 = ax.scatter(x_prime, y_prime, c="red", s=2)

    cb = plt.colorbar(scat1, spacing="proportional", ticks=bounds)
    cb.set_label("Classification")
    cb.set_ticks(np.arange(0.5, len(classes) + 0.5, 1))
    cb.set_ticklabels(classes)
    ax.set_title("Text variation by class")

    plt.show()

    # native plot
    # fig = umap.plot.points(tfidf_embedding, labels=notam_df["classification_column"])
    # umap.plot.plt.show()


if __name__ == "__main__":
    main()
