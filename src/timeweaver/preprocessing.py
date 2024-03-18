from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.interpolate import interp1d, Akima1DInterpolator
import pandas as pd
import numpy as np

class DropColumnTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_drop: list = []):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1)

class CharactersToNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, value):
        self.value = value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        def to_numeric_or_nan(val):
            if isinstance(val, (int, float)):
                return val
            return self.value
        
        X_transformed = X.map(to_numeric_or_nan)
        return X_transformed
    
class EdgeNaNFillerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for column_name in X_transformed.columns:
            first_valid = X_transformed[column_name].first_valid_index()
            last_valid = X_transformed[column_name].last_valid_index()
            
            if first_valid is not None:
                X_transformed.loc[:first_valid, column_name] = X_transformed.loc[:first_valid, column_name].ffill()
            if last_valid is not None:
                X_transformed.loc[last_valid:, column_name] = X_transformed.loc[last_valid:, column_name].bfill()

        return X_transformed


class ContinuityReindexTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tracking_column: str, frequency: float):
        self.tracking_column = tracking_column
        self.frequency = frequency
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X.sort_values(by=self.tracking_column, inplace=True)
        start = X[self.tracking_column].min()
        end = X[self.tracking_column].max()
        time_range = np.arange(start, end + self.frequency, self.frequency)
        full_range_df = pd.DataFrame(time_range, columns=[self.tracking_column])
        X_reindexed = pd.merge(full_range_df, X, on=self.tracking_column, how="left", sort=True)
        return X_reindexed


class ColumnInterpolateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, interpolation_methods):
        """
        interpolation_methods: A dictionary mapping column names to interpolation methods.
        """
        self.interpolation_methods = interpolation_methods
    
    def fit(self, X, y=None):
        return self
    
    def interpolate_column(self, y, method):
        method_parts = method.split('_')
        if len(method_parts) > 1 and method_parts[1] == 'order':
            order = int(method_parts[-1])
            interpolated = y.interpolate(method='polynomial', order=order)
        else:
            interpolated = y.interpolate(method=method)
        return interpolated
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col, method in self.interpolation_methods:  # Iterate directly over the tuple
            if col in X.columns:
                X_transformed[col] = self.interpolate_column(X_transformed[col], method)
        return X_transformed
    
class PreProcessor:
    def __init__(self, pipeline):
        """
        Initializes the Preprocessor object with a pipeline.

        :param pipeline: A sklearn Pipeline object configured for preprocessing.
        """
        self.pipeline = pipeline

    def transform(self, df):
        """
        Processes the input DataFrame using the pipeline and returns the processed DataFrame.

        :param df: The DataFrame to process.
        :return: The processed DataFrame.
        """
        return self.pipeline.transform(df)