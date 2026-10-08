"""Schema-faithful synthetic surrogate for the ToN_IoT Network / UNSW-NB15 CSVs.

WHY THIS EXISTS
---------------
The real datasets are multi-GB and live on the team's GPU box; this cloud
sandbox cannot host them. To *validate the code end-to-end* (that every stage
runs, logs, and behaves sensibly) we generate data with the exact column layout
and dtypes of the real files, with family-conditional structure so classifiers
and the autoencoder have genuine signal to learn.

IMPORTANT -- READ THIS
----------------------
Numbers produced on synthetic data are **not** experimental results and must
never appear in the paper. They exist only to prove the machinery is correct.
Every run made on synthetic data is tagged `data.source: synthetic` in its
config so it can never be confused with a real-data run. The real numbers come
from running the identical code on the real CSVs (`data.source: csv`).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schemas import (DatasetSchema, TONIOT_NETWORK_FEATURES, TONIOT_SCHEMA,
                      UNSW_FEATURES, UNSW_SCHEMA)

# ZAID-reported ToN_IoT family mix after withholding (used only to make the
# synthetic surrogate's proportions realistic; supervised split proportions).
TONIOT_FAMILY_MIX = {
    "normal": 0.55,      # benign dominates real flow data
    "xss": 0.145,
    "injection": 0.055,
    "password": 0.055,
    "scanning": 0.02,
    "backdoor": 0.015,
    "ransomware": 0.005,
    # withheld zero-day families present in the raw pool (protocol removes them
    # from *supervised training* later; they must exist in the source data).
    "ddos": 0.06,
    "dos": 0.03,
    "mitm": 0.01,
}

UNSW_FAMILY_MIX = {
    "normal": 0.55, "generic": 0.15, "exploits": 0.10, "fuzzers": 0.06,
    "dos": 0.05, "scanning": 0.05, "analysis": 0.02, "backdoor": 0.01,
    "shellcode": 0.008, "worms": 0.002,
}


def _numeric_feature_names(schema: DatasetSchema, all_features: List[str]) -> List[str]:
    return [c for c in all_features if c not in schema.categorical_columns]


def _family_means(rng: np.random.Generator, families: List[str], n_num: int,
                  separability: float) -> Dict[str, np.ndarray]:
    """One mean vector per family. `separability` controls how far apart family
    clusters are => how learnable the task is. Benign sits near origin."""
    means = {}
    for fam in families:
        if fam == "normal":
            means[fam] = np.zeros(n_num)
        else:
            means[fam] = rng.normal(0.0, separability, size=n_num)
    return means


def generate(
    schema: DatasetSchema,
    n_samples: int,
    seed: int,
    family_mix: Optional[Dict[str, float]] = None,
    separability: float = 1.1,
    n_categories: int = 6,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if schema.name == TONIOT_SCHEMA.name:
        all_features = TONIOT_NETWORK_FEATURES
        family_mix = family_mix or TONIOT_FAMILY_MIX
    elif schema.name == UNSW_SCHEMA.name:
        all_features = UNSW_FEATURES
        family_mix = family_mix or UNSW_FAMILY_MIX
    else:
        raise ValueError(f"no synthetic generator for schema {schema.name}")

    families = list(family_mix.keys())
    probs = np.array([family_mix[f] for f in families], dtype=float)
    probs = probs / probs.sum()
    fam_idx = rng.choice(len(families), size=n_samples, p=probs)
    fam_col = np.array([families[i] for i in fam_idx])

    num_cols = _numeric_feature_names(schema, all_features)
    means = _family_means(rng, families, len(num_cols), separability)

    # numeric block: family-conditional Gaussian + heavy-tailed noise on a few
    # "byte/pkt count" style columns to mimic network-flow skew.
    X = np.zeros((n_samples, len(num_cols)), dtype=np.float64)
    for i, fam in enumerate(families):
        mask = fam_idx == i
        if not mask.any():
            continue
        X[mask] = rng.normal(means[fam], 1.0, size=(mask.sum(), len(num_cols)))
    # exponentiate a subset to emulate positive, skewed byte/packet counts
    skew_cols = [j for j, c in enumerate(num_cols)
                 if any(k in c for k in ("bytes", "pkts", "duration", "len"))]
    for j in skew_cols:
        X[:, j] = np.expm1(np.clip(X[:, j], -3, 6)).clip(min=0)

    data = {c: X[:, j] for j, c in enumerate(num_cols)}

    # categorical block: family-conditional categorical distribution
    for c in schema.categorical_columns:
        if c not in all_features:
            continue
        cat_means = {fam: rng.dirichlet(np.ones(n_categories) * 0.5) for fam in families}
        vals = np.empty(n_samples, dtype=object)
        for i, fam in enumerate(families):
            mask = fam_idx == i
            if not mask.any():
                continue
            picks = rng.choice(n_categories, size=mask.sum(), p=cat_means[fam])
            vals[mask] = [f"{c}_v{p}" for p in picks]
        data[c] = vals

    # identifier columns (to be dropped) -- exercise the drop logic
    for idc in schema.id_columns:
        if idc in ("ts",):
            data[idc] = rng.integers(1_500_000_000, 1_600_000_000, size=n_samples)
        elif "ip" in idc:
            data[idc] = [f"10.0.{rng.integers(0,255)}.{rng.integers(0,255)}" for _ in range(n_samples)]
        elif "port" in idc:
            data[idc] = rng.integers(1, 65535, size=n_samples)
        else:
            data[idc] = rng.integers(0, n_samples, size=n_samples)

    data[schema.label_column] = (fam_col != schema.benign_family).astype(int)
    data[schema.family_column] = fam_col
    df = pd.DataFrame(data)
    return df
