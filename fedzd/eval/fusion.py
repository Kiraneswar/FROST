"""Weighted-score fusion of the three ZAID abnormality signals.

    AS = w_A * lambda_A  +  w_C * lambda_C  +  w_U * lambda_U            (ZAID Eq.)

  lambda_U : universal-model abnormality probability
  lambda_C : zone-local (personalized) model abnormality probability
  lambda_A : normalized autoencoder reconstruction-error anomaly signal

A sample is flagged as an attack when AS >= threshold. lambda_A is calibrated on
*benign* validation reconstruction errors so its scale matches the [0,1]
classifier probabilities before weighting (ZAID's "normalized" signal).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ReconNormalizer:
    q_lo: float
    q_hi: float

    @classmethod
    def fit(cls, benign_errors: np.ndarray, lo_pct: float = 5, hi_pct: float = 99):
        return cls(q_lo=float(np.percentile(benign_errors, lo_pct)),
                   q_hi=float(np.percentile(benign_errors, hi_pct)))

    def transform(self, errors: np.ndarray) -> np.ndarray:
        denom = max(self.q_hi - self.q_lo, 1e-8)
        return np.clip((errors - self.q_lo) / denom, 0.0, 1.0)


def fuse(lambda_A: np.ndarray, lambda_C: np.ndarray, lambda_U: np.ndarray,
         w_A: float, w_C: float, w_U: float) -> np.ndarray:
    return w_A * lambda_A + w_C * lambda_C + w_U * lambda_U


def predict(anomaly_score: np.ndarray, threshold: float) -> np.ndarray:
    return (anomaly_score >= threshold).astype(int)
