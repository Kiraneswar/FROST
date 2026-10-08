"""Withheld-attack-family zero-day protocol (ZAID-aligned).

MITM, DDoS and DoS are withheld from *supervised* training and introduced only
at zone-level evaluation. This module partitions the raw pool into named splits
and records exact per-family class counts for every split (Stage 1 requirement:
"Document exact class counts per split").

Splits produced
----------------
  universal_train : seen families only -- trains the universal backbone
  universal_val   : seen families only -- validation / early signal
  zone_pool       : seen families only -- distributed across tactical zones
  test_seen       : seen families only -- in-distribution test
  zeroday_pool    : withheld families only -- injected at zone evaluation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .schemas import DatasetSchema


@dataclass
class ZeroDaySplits:
    universal_train: pd.DataFrame
    universal_val: pd.DataFrame
    zone_pool: pd.DataFrame
    test_seen: pd.DataFrame
    zeroday_pool: pd.DataFrame
    withheld_families: List[str]

    def as_dict(self) -> Dict[str, pd.DataFrame]:
        return {
            "universal_train": self.universal_train,
            "universal_val": self.universal_val,
            "zone_pool": self.zone_pool,
            "test_seen": self.test_seen,
            "zeroday_pool": self.zeroday_pool,
        }


def _stratified_partition(df: pd.DataFrame, by: str, fractions: Dict[str, float],
                          seed: int) -> Dict[str, pd.DataFrame]:
    """Split df into named parts with given fractions, stratified by `by`."""
    assert abs(sum(fractions.values()) - 1.0) < 1e-6, "fractions must sum to 1"
    rng = np.random.default_rng(seed)
    parts: Dict[str, List[pd.DataFrame]] = {k: [] for k in fractions}
    for _, grp in df.groupby(by, sort=True):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        boundaries = np.cumsum([int(round(f * n)) for f in fractions.values()])
        boundaries[-1] = n  # absorb rounding into last split
        start = 0
        for name, end in zip(fractions.keys(), boundaries):
            parts[name].append(grp.loc[idx[start:end]])
            start = end
    return {k: (pd.concat(v).reset_index(drop=True) if v else df.iloc[0:0].copy())
            for k, v in parts.items()}


def make_splits(df: pd.DataFrame, schema: DatasetSchema, cfg_zeroday: dict,
                seed: int) -> ZeroDaySplits:
    """cfg_zeroday keys:
      withheld_families: [..]
      seen_fractions: {universal_train, universal_val, zone_pool, test_seen}
    """
    withheld = [schema.normalize_family(f) for f in cfg_zeroday["withheld_families"]]
    fam = schema.family_column

    is_withheld = df[fam].isin(withheld)
    seen_df = df[~is_withheld].reset_index(drop=True)
    zeroday_df = df[is_withheld].reset_index(drop=True)

    fr = cfg_zeroday["seen_fractions"]
    parts = _stratified_partition(seen_df, by=fam, fractions=fr, seed=seed)

    return ZeroDaySplits(
        universal_train=parts["universal_train"],
        universal_val=parts["universal_val"],
        zone_pool=parts["zone_pool"],
        test_seen=parts["test_seen"],
        zeroday_pool=zeroday_df,
        withheld_families=withheld,
    )


def class_counts_report(splits: ZeroDaySplits, schema: DatasetSchema) -> List[dict]:
    """Tidy rows: one per (split, family) with count, for CSV logging."""
    rows = []
    for split_name, sdf in splits.as_dict().items():
        vc = sdf[schema.family_column].value_counts().to_dict()
        for family, count in sorted(vc.items()):
            rows.append({
                "split": split_name, "family": family, "count": int(count),
                "is_withheld": family in splits.withheld_families,
            })
    return rows
