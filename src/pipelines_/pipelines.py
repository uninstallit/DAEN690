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
    DummyEncoderTransformer,
    CatBoostTransformer,
    DeltaTimeTransformer,
    LabelTransformer,
    MostFrequenInputerTransformer,
    # SentenceEmbedderTransformer,
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

cat_boost_encoder_pipeline = Pipeline(
    [
        ("cat_boost", CatBoostTransformer()),
        ("to_dataframe", SeriesToDataframeTransformer()),
    ]
)


conv_notam_date_pipeline = Pipeline(
    [
        ("to_unix_time", NotamDateToUnixTimeTransformer()),
        ("to_dataframe", SeriesToDataframeTransformer()),
    ]
)

label_encoder_pipeline = Pipeline(
    [
        ("to_numbers", LabelTransformer()),
        ("to_dataframe", SeriesToDataframeTransformer()),
    ]
)


preprocess_pipeline = Pipeline(
    [
        (
            "columns",
            ColumnTransformer(
                [
                    (
                        "text_idx_0",
                        clean_text_pipeline,
                        "TEXT",
                    ),
                    (
                        "poss_start_timestamp_idx_1",
                        conv_notam_date_pipeline,
                        "POSSIBLE_START_DATE",
                    ),
                    (
                        "issue_timestamp_idx_2",
                        conv_notam_date_pipeline,
                        "ISSUE_DATE",
                    ),
                    (
                        "location_code_idx_3",
                        cat_boost_encoder_pipeline,
                        "LOCATION_CODE",
                    ),
                    (
                        "classification_idx_4",
                        label_encoder_pipeline,
                        "CLASSIFICATION",
                    ),
                    ("account_id_idx_5", SeriesToDataframeTransformer(), "ACCOUNT_ID"),
                ]
            ),
        ),
    ]
)


features_pipeline = Pipeline(
    [
        ("preprocess", preprocess_pipeline),
        ("add_delta_time_feature_idx_6", DeltaTimeTransformer()),
        # too slow - done separately
        # ("add_text_embedder_feature_idx_5", SentenceEmbedderTransformer()),
    ]
)


def clean_column_text_pipeline(column_name):
    pipeline = Pipeline(
        [
            (
                "process_columns",
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
                "process_columns",
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


def dummy_encode_ordinal_pipeline(column_name):
    pipeline = Pipeline(
        [
            (
                "process_columns",
                ColumnTransformer(
                    [
                        (
                            "dummy_encode",
                            dummy_encoder_pipeline,
                            column_name,
                        )
                    ]
                ),
            )
        ]
    )
    return pipeline


def cat_boost_encode_pipeline(column_name):
    pipeline = Pipeline(
        [
            (
                "process_columns",
                ColumnTransformer(
                    [
                        (
                            "cat_boost_encode",
                            cat_boost_encoder_pipeline,
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
