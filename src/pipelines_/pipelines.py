from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(root)

params_path = root + "/sample_data/params.txt"

from transformers_.transformers import (
    SplitAlphaNumericTransformer,
    RemovePunctuationTransformer,
    RemoveDigitsTransformer,
    DecodeAbbrevTransformer,
    JoinStrListTransformer,
    SeriesToDataframeTransformer,
    NotamDateToUnixTimeTransformer,
    DummyEncoderTransformer,
    DeltaTimeTransformer,
    StandardScalerTransformer,
    OrdinalEncoderAndStandardScalerTransformer,
    SentenceEmbedderTransformer,
    MinMaxScalerTransformer,
    OrdinalEncoderAndMinMaxScalerTransformer,
    NormalizeCaseTransformer,
    LowerCaseTransformer
)


clean_text_pipeline = Pipeline(
    [
        ("split_sentns", SplitAlphaNumericTransformer()),
        ("remove_digit", RemoveDigitsTransformer()),
        ("decode_abbrev", DecodeAbbrevTransformer()),
        ("remove_punkt", RemovePunctuationTransformer()),
        ("strjoin_words", JoinStrListTransformer()),
        ("to_dataframe", SeriesToDataframeTransformer()),
    ]
)

lower_case_text_pipeline = Pipeline(
    [
        ("lower_case_sentns", LowerCaseTransformer()),
        ("to_dataframe", SeriesToDataframeTransformer()),
    ]
)

dummy_encoder_pipeline = Pipeline(
    [
        ("one_hot", DummyEncoderTransformer()),
        ("to_dataframe", SeriesToDataframeTransformer()),
    ]
)

location_code_pipeline = Pipeline(
    [
        (
            "location_code_normalize",
            OrdinalEncoderAndMinMaxScalerTransformer(
                name="location_code", filepath=params_path
            ),
        ),
        ("location_code_to_dataframe", SeriesToDataframeTransformer()),
    ]
)

account_id_pipeline = Pipeline(
    [
        (
            "account_id_normalize",
            OrdinalEncoderAndMinMaxScalerTransformer(
                name="account_id", filepath=params_path
            ),
        ),
        ("account_id_to_dataframe", SeriesToDataframeTransformer()),
    ]
)

start_date_pipeline = Pipeline(
    [
        ("start_date_unix_time", NotamDateToUnixTimeTransformer()),
        (
            "start_date_normalize",
            MinMaxScalerTransformer(name="start_date", filepath=params_path),
        ),
        ("start_date_to_dataframe", SeriesToDataframeTransformer()),
    ]
)

issue_date_pipeline = Pipeline(
    [
        ("issue_date_unix_time", NotamDateToUnixTimeTransformer()),
        (
            "issue_date_normalize",
            MinMaxScalerTransformer(name="issue_date", filepath=params_path),
        ),
        ("issue_date_to_dataframe", SeriesToDataframeTransformer()),
    ]
)

classification_pipeline = Pipeline(
    [
        (
            "classification_normalize",
            OrdinalEncoderAndMinMaxScalerTransformer(
                name="classification", filepath=params_path
            ),
        ),
        ("classification_to_dataframe", SeriesToDataframeTransformer()),
    ]
)


preprocess_pipeline = Pipeline(
    [
        (
            "columns",
            ColumnTransformer(
                [
                    ("idx_0", SeriesToDataframeTransformer(), "NOTAM_REC_ID"),
                    ("idx_1", clean_text_pipeline, "E_CODE"),
                    ("idx_2", start_date_pipeline, "POSSIBLE_START_DATE"),
                    ("idx_3", issue_date_pipeline, "ISSUE_DATE"),
                    ("idx_4", location_code_pipeline, "LOCATION_CODE"),
                    ("idx_5", classification_pipeline, "CLASSIFICATION"),
                    ("idx_6", account_id_pipeline, "ACCOUNT_ID"),
                ]
            ),
        ),
    ]
)


features_pipeline = Pipeline(
    [
        ("preprocess", preprocess_pipeline),
        ("idx_7", DeltaTimeTransformer()),
        ("idx_8", SentenceEmbedderTransformer()),
    ]
)


def clean_column_text_pipeline(col_name):
    _pipeline = Pipeline(
        [
            (
                "columns",
                ColumnTransformer(
                    [
                        ("clean_text", clean_text_pipeline, col_name),
                    ]
                ),
            ),
        ]
    )
    return _pipeline

def lower_case_column_text_pipeline(col_name):
    _pipeline = Pipeline(
        [
            (
                "columns",
                ColumnTransformer(
                    [
                        ("lower_case_text", lower_case_text_pipeline, col_name),
                    ]
                ),
            ),
        ]
    )
    return _pipeline



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
    pass


if __name__ == "__main__":
    main()
