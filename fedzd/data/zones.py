"""Partition data into simulated tactical zones with non-IID class imbalance.

Two operations:
  1. `partition_zones`  -- split the *seen-family* zone pool across N zones with
     Dirichlet label skew (standard FL non-IID mechanism; `alpha` controls
     severity -- small alpha => each family concentrated in a few zones).
  2. `build_zone_eval`  -- assemble each zone's evaluation set: a slice of the
     in-distribution test data plus a small injected number of withheld
     zero-day samples per family per zone (ZAID reports ~12-29/zone/family).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .schemas import DatasetSchema


@dataclass
class ZonePartition:
    zone_train: List[pd.DataFrame]      # seen-family training data per zone
    zone_eval: List[pd.DataFrame]       # eval data (seen + injected zero-day) per zone
    n_zones: int


def partition_zones(pool: pd.DataFrame, schema: DatasetSchema, n_zones: int,
                    alpha: float, seed: int) -> List[pd.DataFrame]:
    """Dirichlet label-skew partition. Returns a list of DataFrames, one per zone."""
    rng = np.random.default_rng(seed)
    fam = schema.family_column
    zone_rows: List[List[np.ndarray]] = [[] for _ in range(n_zones)]
    for _, grp in pool.groupby(fam, sort=True):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * n_zones)
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        chunks = np.split(idx, cuts)
        for z in range(n_zones):
            zone_rows[z].append(chunks[z])
    zones = []
    for z in range(n_zones):
        sel = np.concatenate(zone_rows[z]) if zone_rows[z] else np.array([], dtype=int)
        zones.append(pool.loc[sel].reset_index(drop=True))
    return zones


def build_zone_eval(test_seen: pd.DataFrame, zeroday_pool: pd.DataFrame,
                    schema: DatasetSchema, n_zones: int,
                    zeroday_per_zone_per_family: int, seed: int) -> List[pd.DataFrame]:
    """Each zone gets an even slice of in-distribution test data plus a small,
    fixed number of injected samples of each withheld family."""
    rng = np.random.default_rng(seed + 777)
    fam = schema.family_column

    # even split of the seen test set across zones (stratified by family)
    seen_parts: List[List[pd.DataFrame]] = [[] for _ in range(n_zones)]
    for _, grp in test_seen.groupby(fam, sort=True):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        for z, chunk in enumerate(np.array_split(idx, n_zones)):
            seen_parts[z].append(grp.loc[chunk])

    # injected zero-day, per family per zone
    zd_parts: List[List[pd.DataFrame]] = [[] for _ in range(n_zones)]
    for _, grp in zeroday_pool.groupby(fam, sort=True):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        take = min(len(idx), zeroday_per_zone_per_family * n_zones)
        idx = idx[:take]
        for z, chunk in enumerate(np.array_split(idx, n_zones)):
            zd_parts[z].append(grp.loc[chunk])

    zone_eval = []
    for z in range(n_zones):
        frames = seen_parts[z] + zd_parts[z]
        zone_eval.append(pd.concat(frames).sample(frac=1.0, random_state=seed + z)
                         .reset_index(drop=True))
    return zone_eval


def make_zones(splits, schema: DatasetSchema, cfg_zones: dict, seed: int) -> ZonePartition:
    """cfg_zones keys: n_zones, dirichlet_alpha, zeroday_per_zone_per_family."""
    n = int(cfg_zones["n_zones"])
    zone_train = partition_zones(splits.zone_pool, schema, n,
                                 float(cfg_zones["dirichlet_alpha"]), seed)
    zone_eval = build_zone_eval(splits.test_seen, splits.zeroday_pool, schema, n,
                                int(cfg_zones["zeroday_per_zone_per_family"]), seed)
    return ZonePartition(zone_train=zone_train, zone_eval=zone_eval, n_zones=n)


def zone_distribution_report(zp: ZonePartition, schema: DatasetSchema) -> List[dict]:
    rows = []
    fam = schema.family_column
    for z in range(zp.n_zones):
        for split_name, frames in (("train", zp.zone_train), ("eval", zp.zone_eval)):
            vc = frames[z][fam].value_counts().to_dict()
            for family, count in sorted(vc.items()):
                rows.append({"zone": z, "split": split_name, "family": family,
                             "count": int(count)})
    return rows
