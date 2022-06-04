import sqlite3
import re
import string
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from transformers_.transformers import (
    SplitAlphaNumericTransformer,
    RemovePunctuationTransformer,
    RemoveDigitsTransformer,
    DecodeAbbrevTransformer,
    JoinStrListTransformer,
    SeriesToDataframeTransformer,
)


clean_text_pipeline = Pipeline(
    [
        ("split_sentns", SplitAlphaNumericTransformer()),
        ("remove_punkt", RemovePunctuationTransformer()),
        ("remove_digit", RemoveDigitsTransformer()),
        ("decode_abbrev", DecodeAbbrevTransformer()),
        ("strjoin_words", JoinStrListTransformer()),
        ("to_dataframe", SeriesToDataframeTransformer()),
    ]
)

clean_all_columns_pipeline = Pipeline(
    [
        (
            "clean_up_columns",
            ColumnTransformer(
                [
                    (
                        "clean_text",
                        clean_text_pipeline,
                        "text_column",
                    )
                ]
            ),
        )
    ]
)


def clean_column_text_pipeline(column_name):
    pipeline = Pipeline(
        [
            (
                "clean_up_columns",
                ColumnTransformer(
                    [
                        (
                            "clean_text",
                            clean_text_pipeline,
                            column_name,
                        )
                    ]
                ),
            )
        ]
    )
    return pipeline


def main():

    conn = sqlite3.Connection("./data/svo_db_20200901.db")
    cursor = conn.cursor()
    sql = """ SELECT "TEXT" FROM notams"""

    corpus = pd.DataFrame(
        {"text_column": [text for text in cursor.execute(sql).fetchall()]}
    )


if __name__ == "__main__":
    main()
