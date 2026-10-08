"""Stage 1 orchestration: raw load -> zero-day protocol -> zones -> feature
selection -> fitted preprocessor, with every split's class counts logged.

Produces a `Stage1Bundle` (consumed by all later stages) and writes:
  class_counts.csv          exact per-family counts for every split
  zone_distribution.csv     per-zone per-family counts (train + eval)
  feature_selection.csv     every candidate feature, RFE rank, MI score, kept?
  feature_selection.json    summary: retained list, candidate count
  stage1_summary.json       one-glance summary of the whole stage
  stage1_bundle.pkl         pickled bundle for downstream stages
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..utils.config import RunContext
from ..utils.metriclog import RunLogger
from . import feature_select as fs
from . import loaders, zeroday, zones
from .schemas import DatasetSchema
from .transforms import Preprocessor
from .zeroday import ZeroDaySplits
from .zones import ZonePartition


@dataclass
class Stage1Bundle:
    schema: DatasetSchema
    splits: ZeroDaySplits
    zones: ZonePartition
    preprocessor: Preprocessor
    features: List[str]
    n_candidate_features: int


def run_stage1(ctx: RunContext, logger: RunLogger) -> Stage1Bundle:
    cfg = ctx.cfg
    seed = ctx.seed
    logger.info(f"=== STAGE 1: data pipeline (seed={seed}, "
                f"source={cfg['data'].get('source')}) ===")

    # 1. load raw
    df, schema = loaders.load_raw(cfg["data"], seed)
    logger.info(f"loaded {len(df):,} rows, {df.shape[1]} columns "
                f"(schema={schema.name})")
    logger.event("stage1_load", rows=len(df), cols=int(df.shape[1]),
                 family_counts=loaders.family_counts(df, schema))

    # 2. zero-day protocol splits
    splits = zeroday.make_splits(df, schema, cfg["zeroday"], seed)
    cc_rows = zeroday.class_counts_report(splits, schema)
    logger.table("class_counts", cc_rows,
                 fieldnames=["split", "family", "count", "is_withheld"])
    for name, sdf in splits.as_dict().items():
        logger.info(f"  split {name:16s}: {len(sdf):>8,} rows")
    logger.info(f"  withheld (zero-day) families: {splits.withheld_families}")

    # 3. feature selection on universal_train (seen families only)
    logger.info("running RF+RFE then MI feature selection ...")
    fsr = fs.select_features(splits.universal_train, schema,
                             cfg["feature_selection"], seed)
    logger.table("feature_selection", fsr.detail,
                 fieldnames=["feature", "is_categorical", "rfe_rank",
                             "survived_rfe", "mi_score", "retained", "reason"])
    logger.dump_json("feature_selection.json", {
        "n_candidate_features": fsr.n_candidate,
        "n_retained": len(fsr.retained),
        "retained_features": fsr.retained,
        "mi_scores": {k: round(v, 6) for k, v in
                      sorted(fsr.mi_scores.items(), key=lambda x: -x[1])},
    })
    logger.info(f"  retained {len(fsr.retained)}/{fsr.n_candidate} features: "
                f"{fsr.retained}")

    # 4. fitted preprocessor (fit on universal_train, no zero-day leakage)
    pre = Preprocessor.fit(splits.universal_train, schema, fsr.retained)
    logger.info(f"  preprocessor fit: input dim = {pre.n_features}")

    # 5. zones
    zp = zones.make_zones(splits, schema, cfg["zones"], seed)
    zd_rows = zones.zone_distribution_report(zp, schema)
    logger.table("zone_distribution", zd_rows,
                 fieldnames=["zone", "split", "family", "count"])
    for z in range(zp.n_zones):
        logger.info(f"  zone {z}: train={len(zp.zone_train[z]):>7,}  "
                    f"eval={len(zp.zone_eval[z]):>6,}")

    bundle = Stage1Bundle(
        schema=schema, splits=splits, zones=zp, preprocessor=pre,
        features=fsr.retained, n_candidate_features=fsr.n_candidate,
    )

    # persist
    summary = {
        "schema": schema.name,
        "data_source": cfg["data"].get("source"),
        "total_rows": int(len(df)),
        "withheld_families": splits.withheld_families,
        "split_sizes": {k: int(len(v)) for k, v in splits.as_dict().items()},
        "n_candidate_features": fsr.n_candidate,
        "n_retained_features": len(fsr.retained),
        "retained_features": fsr.retained,
        "n_zones": zp.n_zones,
        "zone_train_sizes": [int(len(x)) for x in zp.zone_train],
        "zone_eval_sizes": [int(len(x)) for x in zp.zone_eval],
    }
    logger.dump_json("stage1_summary.json", summary)
    with open(ctx.path("stage1_bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    logger.info("stage 1 complete; bundle + reports written.")
    return bundle
