"""Load a dataset as a raw DataFrame, from real CSV(s) or the synthetic surrogate.

The returned frame always carries the schema's `label_column` and
`family_column` (families normalized to canonical lowercase). Downstream stages
never care whether the data was real or synthetic -- only the config records it.
"""
from __future__ import annotations

import glob
from typing import List, Optional

import numpy as np
import pandas as pd

from . import synthetic
from .schemas import SCHEMAS, DatasetSchema


def _read_csvs(patterns: List[str]) -> pd.DataFrame:
    files: List[str] = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise FileNotFoundError(f"no CSVs matched: {patterns}")
    frames = [pd.read_csv(f, low_memory=False, encoding="utf-8-sig") for f in files]
    df = pd.concat(frames, ignore_index=True)
    # strip BOM / stray whitespace from column names (real CSVs often carry a BOM
    # on the first header, e.g. ToN_IoT's "ts") so id-column drops match by name
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df


def load_raw(cfg_data: dict, seed: int) -> tuple[pd.DataFrame, DatasetSchema]:
    """cfg_data keys:
      schema: 'toniot_network' | 'unsw_nb15'
      source: 'csv' | 'synthetic'
      csv_globs: [..]                 (source == csv)
      synthetic: {n_samples, separability}   (source == synthetic)
      max_rows: optional cap (for fast iteration on real data)
    """
    schema = SCHEMAS[cfg_data["schema"]]
    source = cfg_data.get("source", "synthetic")

    if source == "csv":
        df = _read_csvs(cfg_data["csv_globs"])
    elif source == "synthetic":
        s = cfg_data.get("synthetic", {})
        df = synthetic.generate(
            schema,
            n_samples=int(s.get("n_samples", 40000)),
            seed=seed,
            separability=float(s.get("separability", 1.1)),
        )
    else:
        raise ValueError(f"unknown data.source: {source}")

    # ToN_IoT full CSVs sometimes use '-' for missing; normalize.
    df = df.replace({"-": np.nan})

    # normalize family strings to canonical names
    fam = schema.family_column
    if fam not in df.columns:
        raise KeyError(f"family column '{fam}' not in data; columns={list(df.columns)[:20]}")
    df[fam] = df[fam].map(schema.normalize_family)

    # ensure a binary label exists / is consistent with family
    lbl = schema.label_column
    if lbl not in df.columns:
        df[lbl] = (df[fam] != schema.benign_family).astype(int)
    else:
        df[lbl] = pd.to_numeric(df[lbl], errors="coerce").fillna(0).astype(int)

    max_rows = cfg_data.get("max_rows")
    if max_rows and len(df) > int(max_rows):
        df = df.sample(n=int(max_rows), random_state=seed).reset_index(drop=True)

    return df, schema


def family_counts(df: pd.DataFrame, schema: DatasetSchema) -> dict:
    return df[schema.family_column].value_counts().sort_index().to_dict()
