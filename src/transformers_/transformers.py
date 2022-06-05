import re
import string
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(parent)


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


class NotamDateToUNixTimeTransformer(BaseEstimator, TransformerMixin):
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
            for date_time_str in x.tolist()
        ]
        # convert to unix time
        _x = [datetime.timestamp(date_time_obj) for date_time_obj in _x]
        return pd.Series(_x, name=series_name)
