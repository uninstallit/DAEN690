import sqlite3
import pandas as pd
import numpy as np
import umap
import umap.plot

# Used to get the data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(parent)

from pipelines_.pipelines import clean_column_text_pipeline


# source: https://umap-learn.readthedocs.io/en/latest/document_embedding.html#using-tf-idf

# Potential applications
# Now that we have an embedding, what can we do with it?

# Explore/visualize your corpus to identify topics/trends
# Cluster the embedding to find groups of related documents
# Look for nearest neighbours to find related documents
# Look for anomalous documents


def main():

    conn = sqlite3.Connection("./data/svo_db_20200901.db")
    cursor = conn.cursor()

    sql = """ SELECT "TEXT" FROM notams"""
    text = [text[0] for text in cursor.execute(sql).fetchall()]

    sql = """ SELECT CLASSIFICATION FROM notams"""
    classification = [class_[0] for class_ in cursor.execute(sql).fetchall()]

    notam_df = pd.DataFrame(
        {"text_column": text, "classification_column": classification}
    )
    notam_df = notam_df.dropna()
    notam_df = notam_df.sample(4000)

    # print(notam_df['text_column'].isnull().values.any())
    # print(notam_df['classification_column'].isnull().values.any())

    clean_text = clean_column_text_pipeline("text_column").fit_transform(notam_df)
    clean_text = np.squeeze(clean_text, axis=1)
    notam_df["text_column"] = clean_text

    tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words="english")
    tfidf_word_doc_matrix = tfidf_vectorizer.fit_transform(
        notam_df["text_column"].tolist()
    )
    tfidf_embedding = umap.UMAP(metric="hellinger", n_neighbors=15).fit(
        tfidf_word_doc_matrix
    )

    fig = umap.plot.points(tfidf_embedding, labels=notam_df["classification_column"])
    umap.plot.plt.show()


if __name__ == "__main__":
    main()
