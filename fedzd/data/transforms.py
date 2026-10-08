"""Fitted preprocessing shared by every model.

Fit on `universal_train` (seen families only -- no zero-day leakage), then apply
unchanged to every other split/zone. Categorical retained features are
ordinal-encoded; the full retained vector is standard-scaled. Produces a fixed
width float32 matrix suitable for the 1D-CNN backbone and the autoencoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .schemas import DatasetSchema


@dataclass
class Preprocessor:
    schema: DatasetSchema
    features: List[str]
    _encoders: Dict[str, OrdinalEncoder]
    _scaler: StandardScaler

    @property
    def n_features(self) -> int:
        return len(self.features)

    @classmethod
    def fit(cls, df: pd.DataFrame, schema: DatasetSchema, features: List[str]) -> "Preprocessor":
        encoders: Dict[str, OrdinalEncoder] = {}
        cols = []
        for c in features:
            if c in schema.categorical_columns:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                enc.fit(df[[c]].fillna("missing").astype(str))
                encoders[c] = enc
                cols.append(enc.transform(df[[c]].fillna("missing").astype(str)).ravel())
            else:
                cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy())
        M = np.nan_to_num(np.column_stack(cols).astype(np.float64), posinf=0.0, neginf=0.0)
        scaler = StandardScaler().fit(M)
        return cls(schema=schema, features=list(features), _encoders=encoders, _scaler=scaler)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.features:
            if c in self._encoders:
                cols.append(self._encoders[c].transform(df[[c]].fillna("missing").astype(str)).ravel())
            else:
                cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy())
        M = np.nan_to_num(np.column_stack(cols).astype(np.float64), posinf=0.0, neginf=0.0)
        return self._scaler.transform(M).astype(np.float32)

    def labels(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.schema.label_column].to_numpy().astype(np.float32)

    def families(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.schema.family_column].to_numpy()

    def xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.transform(df), self.labels(df)
