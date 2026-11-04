"""Serializable custom missing-value handling and the notebook encoder."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class UnknownImputer(TransformerMixin, BaseEstimator):
    """Notebook's constant fills; stringify only mixed boolean/Unknown columns."""

    def __init__(self, stringify=False):
        self.stringify = stringify

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        data = X.copy().astype(object).where(X.notna(), "Unknown")
        return data.astype(str) if self.stringify else data

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


class DataPreprocessor(TransformerMixin, BaseEstimator):
    """Fit categories on training only; preserve numerical NaNs for XGBoost."""

    impute_columns = ("funder", "installer", "public_meeting",
                      "scheme_management", "scheme_name", "permit")

    def __init__(self, min_frequency=20, max_categories=50,
                 handle_unknown="infrequent_if_exist"):
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.handle_unknown = handle_unknown

    def _encoder(self):
        return OneHotEncoder(min_frequency=self.min_frequency,
                             max_categories=self.max_categories,
                             handle_unknown=self.handle_unknown, sparse_output=True)

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        transformers = []
        for column in self.impute_columns:
            if column in X:
                transformers.append((column, Pipeline([
                    ("impute", UnknownImputer(column in ("public_meeting", "permit"))),
                    ("encode", self._encoder()),
                ]), [column]))
        other = X.select_dtypes(include=["object", "category", "bool", "string"]).columns
        other = sorted(set(other) - set(self.impute_columns))
        numbers = X.select_dtypes(include="number").columns.tolist()
        if other:
            transformers.append(("other_categorical", self._encoder(), other))
        if numbers:
            transformers.append(("numerical", "passthrough", numbers))
        self.column_transformer_ = ColumnTransformer(transformers, remainder="drop")
        self.column_transformer_.fit(X, y)
        return self

    def transform(self, X):
        return self.column_transformer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.column_transformer_.get_feature_names_out(input_features)
