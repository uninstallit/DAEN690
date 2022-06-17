from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(parent)

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
            OrdinalEncoderAndStandardScalerTransformer(
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
            OrdinalEncoderAndStandardScalerTransformer(
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
            StandardScalerTransformer(name="start_date", filepath=params_path),
        ),
        ("start_date_to_dataframe", SeriesToDataframeTransformer()),
    ]
)

issue_date_pipeline = Pipeline(
    [
        ("issue_date_unix_time", NotamDateToUnixTimeTransformer()),
        (
            "issue_date_normalize",
            StandardScalerTransformer(name="issue_date", filepath=params_path),
        ),
        ("issue_date_to_dataframe", SeriesToDataframeTransformer()),
    ]
)

classification_pipeline = Pipeline(
    [
        (
            "classification_normalize",
            OrdinalEncoderAndStandardScalerTransformer(
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
                    ("idx_0", clean_text_pipeline, "TEXT"),
                    ("idx_1", start_date_pipeline, "POSSIBLE_START_DATE"),
                    ("idx_2", issue_date_pipeline, "ISSUE_DATE"),
                    ("idx_3", location_code_pipeline, "LOCATION_CODE"),
                    ("idx_4", classification_pipeline, "CLASSIFICATION"),
                    ("idx_5", account_id_pipeline, "ACCOUNT_ID"),
                ]
            ),
        ),
    ]
)


features_pipeline = Pipeline(
    [
        ("preprocess", preprocess_pipeline),
        ("idx_6", DeltaTimeTransformer()),
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
    pass


if __name__ == "__main__":
    main()
