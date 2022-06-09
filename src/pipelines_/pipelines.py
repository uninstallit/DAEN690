import sqlite3
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(parent)

from transformers_.transformers import (
    SplitAlphaNumericTransformer,
    RemovePunctuationTransformer,
    RemoveDigitsTransformer,
    DecodeAbbrevTransformer,
    JoinStrListTransformer,
    SeriesToDataframeTransformer,
    NotamDateToUnixTimeTransformer,
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

conv_notam_date_pipeline = Pipeline(
    [
        ("to_unix_time", NotamDateToUnixTimeTransformer()),
        ("to_dataframe", SeriesToDataframeTransformer()),
    ]
)


clean_all_notam_columns_pipeline = Pipeline(
    [
        (
            "clean_up_columns",
            ColumnTransformer(
                [
                    (
                        "clean_text_idx_0",
                        clean_text_pipeline,
                        "TEXT",
                    ),
                    (
                        "clean_simple_text_idx_1",
                        clean_text_pipeline,
                        "SIMPLE_TEXT",
                    ),
                    (
                        "poss_start_timestamp_idx_2",
                        conv_notam_date_pipeline,
                        "POSSIBLE_START_DATE",
                    ),
                    (
                        "poss_end_timestamp_idx_3",
                        conv_notam_date_pipeline,
                        "POSSIBLE_END_DATE",
                    ),
                    (
                        "issue_timestamp_idx_4",
                        conv_notam_date_pipeline,
                        "ISSUE_DATE",
                    ),
                    (
                        "canceled_timestamp_idx_5",
                        conv_notam_date_pipeline,
                        "CANCELED_DATE",
                    ),
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


def conv_column_date_pipeline(column_name):
    pipeline = Pipeline(
        [
            (
                "clean_up_columns",
                ColumnTransformer(
                    [
                        (
                            "conv_date",
                            conv_notam_date_pipeline,
                            column_name,
                        )
                    ]
                ),
            )
        ]
    )
    return pipeline


def main():

    # example

    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    sql = """ SELECT issueTime, startTime, stopTime FROM human_matches"""
    df = pd.read_sql_query(sql, conn)

    column = "IssueTime"
    df = df[column].to_frame()
    df = df.dropna()
    print(df.head())

    conn.close()

    result = conv_column_date_pipeline(column).fit_transform(df)
    print(result)
    print(type(result))


if __name__ == "__main__":
    main()
