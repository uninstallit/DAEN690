import re
import string
import pandas as pd
import numpy as np
import category_encoders as ce
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(root)

from libs.PyNotam.notam import Notam


class SplitAlphaNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        _x = [re.split(r"\W+", str(text)) for text in x.tolist()]
        return pd.Series(_x, name=series_name)


class RemovePunctuationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        table = str.maketrans("", "", string.punctuation)
        _x = [
            [word.translate(table) for word in words if word != ""]
            for words in x.tolist()
        ]
        return pd.Series(_x, name=series_name)


class RemoveDigitsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        table = str.maketrans("", "", string.digits)
        _x = [
            [word.translate(table) for word in words if word != ""]
            for words in x.tolist()
        ]
        return pd.Series(_x, name=series_name)


class DecodeAbbrevTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        _x = [
            [Notam.decode_abbr(word) for word in words if word != ""]
            for words in x.tolist()
        ]
        return pd.Series(_x, name=series_name)


class NormalizeCaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        _x = [[word.lower() for word in words] for words in x.tolist()]
        return pd.Series(_x, name=series_name)


class JoinStrListTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        _x = [" ".join([word for word in words]) for words in x.tolist()]
        return pd.Series(_x, name=series_name)


class SeriesToDataframeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return x.to_frame()


class NotamDateToUnixTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        date_format = "%Y-%m-%d %H:%M:%S"
        # parse str date

        _x = [
            datetime.strptime(date_time_str, date_format)
            if date_time_str != None
            else datetime.strptime(x.value_counts().idxmax(), date_format)
            for date_time_str in x.tolist()
        ]
        # convert to unix time
        _x = [datetime.timestamp(date_time_obj) for date_time_obj in _x]
        return pd.Series(_x, name=series_name)


class DummyEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        self.x_ = x.apply(lambda x: [x]).tolist()
        return self

    def transform(self, x, y=None):
        series_name = x.name
        encoder = OneHotEncoder(drop="first", sparse=False)
        _x = encoder.fit_transform(self.x_)
        return pd.Series(_x, name=series_name)


class CatBoostTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target=None):
        self.target = target

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        cat_encoder = ce.cat_boost.CatBoostEncoder()
        _x = cat_encoder.fit_transform(x, self.target, return_df=False).to_numpy()
        _x = np.squeeze(_x)
        return pd.Series(_x, name=series_name)


class LabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        label_encoder = LabelEncoder()
        _x = label_encoder.fit_transform(x)
        return pd.Series(_x, name=series_name)


class DeltaTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_indexes=None):
        self.column_indexes = column_indexes

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        _x = np.absolute(x[:, self.column_indexes[0]] - x[:, self.column_indexes[1]])
        _x = np.expand_dims(_x, -1)
        x = np.hstack((x, _x))
        return x


class SentenceEmbedderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_index=None):
        self.column_index = column_index
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        _x = self.embedder.encode(x[:, self.column_index], convert_to_numpy=True)
        # use np.fromstring('1 2', dtype=int, sep=' ') to decode
        _x = np.apply_along_axis(str, 1, _x)
        _x = np.expand_dims(_x, -1)
        x = np.hstack((x, _x))
        return x


class MostFrequenInputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        inputer = SimpleImputer(strategy="most_frequent")
        _x = inputer.fit_transform(x)
        return pd.Series(_x, name=series_name).to_frame()
