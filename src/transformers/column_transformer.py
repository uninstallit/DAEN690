import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from libs.PyNotam.notam import Notam
from sklearn.preprocessing import Normalizer

launch_df = pd.read_csv("./data/launch.csv")
print(launch_df.head())


class NotamLocationDecoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        _x = x  # some transformation
        return pd.DataFrame({series_name: _x})


class NotamMessageDecoder(BaseEstimator, TransformerMixin):
    def __init__(self, params={}):
        self.params = params

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        series_name = x.name
        _x = x  # some transformation
        return pd.DataFrame({series_name: _x})


# columns
location_column = "Location"
full_msg_column = "NOTAM Condition/LTA subject/Construction graphic title"

# column transformer will combine data
column_transformer = ColumnTransformer(
    # name, fitted_transformer, column
    [
        ("loc", NotamLocationDecoder(), location_column),
        ("msg", NotamMessageDecoder(), full_msg_column),
    ]
)

column_transformer.set_params(**{"msg__params": "hi"})
result = column_transformer.fit_transform(launch_df)

# inspect
print(result.shape) # note 2 dims
print(result[0][0]) # note location value