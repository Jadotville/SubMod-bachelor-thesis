"""Encode mixed tabular features for sklearn/LightGBM while keeping raw search columns."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


#: Levels rarer than this share of the global training rows are pooled into a single
#: "infrequent" column. Without it, the ICD-9 diagnosis codes of `Diabetes130US`
#: (three columns, 2249 levels between them) expand to a 2479-column matrix, and a
#: local model fitted on a 2000-row subgroup would have more columns than rows — it
#: would overfit rather than reveal a region-specific pattern, so Q would measure
#: variance instead of adaptability. At 1 % of a 50 000-row training half a level
#: needs roughly 500 occurrences to get its own column, which is also about the point
#: where a local fit could estimate a coefficient for it at all.
MIN_CATEGORY_FREQUENCY = 0.01


def make_tabular_encoder(X_sample: pd.DataFrame) -> ColumnTransformer:
    """One-hot categoricals + imputed/scaled numerics (fixed recipe for all datasets)."""
    numeric, categorical = _split_columns(X_sample)
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=MIN_CATEGORY_FREQUENCY,
                                sparse_output=False,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns for encoding.")
    return ColumnTransformer(transformers, remainder="drop")


class EncodedModelHooks:
    """
    Fit a shared encoder on the global training matrix; reuse for local fits.

    TabPFN and other raw-data models should not use this wrapper.
    """

    def __init__(self, model_builder):
        self.model_builder = model_builder
        self.encoder: Optional[ColumnTransformer] = None

    def build(self):
        return self.model_builder()

    def _transform(self, X: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        if fit:
            self.encoder = make_tabular_encoder(X)
            arr = self.encoder.fit_transform(X)
        else:
            if self.encoder is None:
                raise RuntimeError("Encoder not fitted; train global model first.")
            arr = self.encoder.transform(X)
        return np.asarray(arr, dtype=np.float64)

    def train_global(self, model, X: pd.DataFrame, y: pd.Series) -> None:
        X_enc = self._transform(X, fit=True)
        model.fit(X_enc, np.asarray(y))

    def train_local(self, model, X: pd.DataFrame, y: pd.Series) -> None:
        X_enc = self._transform(X, fit=False)
        model.fit(X_enc, np.asarray(y))

    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        X_enc = self._transform(X, fit=False)
        proba = model.predict_proba(X_enc)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
