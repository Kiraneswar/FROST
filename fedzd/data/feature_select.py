"""Feature selection: Random Forest + RFE, then Mutual Information scoring.

ZAID protocol: "Random Forest Classifier with Recursive Feature Elimination"
followed by Mutual Information, retaining the top 40% of MI scores (=> 32
features on their real ToN_IoT column set).

We reproduce the *method* and log which columns are retained and why. Selection
operates at the level of original feature columns (categoricals are
ordinal-encoded to a single numeric input each) so the retained set is
directly interpretable, exactly as a "feature" in the paper's table.

To land on exactly 32 features for the real data, set
`feature_selection.final_n_features: 32` (an explicit override), or tune
`rfe_n_features` / `mi_top_fraction`; the default reproduces the two-stage
ranking and keeps top-40% by MI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from .schemas import DatasetSchema


@dataclass
class FeatureSelectionResult:
    retained: List[str]
    rfe_ranking: Dict[str, int]          # 1 = kept by RFE, higher = eliminated earlier
    mi_scores: Dict[str, float]
    n_candidate: int
    detail: List[dict] = field(default_factory=list)  # tidy per-feature log


def _encode_for_selection(df: pd.DataFrame, schema: DatasetSchema,
                          feature_cols: List[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c in schema.categorical_columns:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            # fill missing BEFORE astype(str): pandas 3.0 keeps NaN through
            # astype(str), which the encoder would then emit as NaN.
            X[c] = enc.fit_transform(df[[c]].fillna("missing").astype(str)).ravel()
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return X.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)


def candidate_feature_columns(df: pd.DataFrame, schema: DatasetSchema) -> List[str]:
    drop = set(schema.id_columns) | {schema.label_column, schema.family_column}
    return [c for c in df.columns if c not in drop]


def select_features(df: pd.DataFrame, schema: DatasetSchema, cfg_fs: dict,
                    seed: int) -> FeatureSelectionResult:
    """cfg_fs keys:
      rfe_n_features: int | 'half'      -- RFE target count
      mi_top_fraction: float            -- keep top fraction by MI (default 0.40)
      final_n_features: Optional[int]   -- hard override of final count
      max_rows_for_selection: int       -- subsample for tractable RF/RFE/MI
      rf_n_estimators, rf_max_depth
    """
    feature_cols = candidate_feature_columns(df, schema)
    work = df
    cap = int(cfg_fs.get("max_rows_for_selection", 30000))
    if len(df) > cap:
        work = df.sample(n=cap, random_state=seed).reset_index(drop=True)

    X = _encode_for_selection(work, schema, feature_cols)
    y = work[schema.label_column].to_numpy()

    n_candidate = len(feature_cols)
    rfe_n = cfg_fs.get("rfe_n_features", "half")
    if rfe_n == "half":
        rfe_n = max(1, n_candidate // 2)
    rfe_n = min(int(rfe_n), n_candidate)

    rf = RandomForestClassifier(
        n_estimators=int(cfg_fs.get("rf_n_estimators", 200)),
        max_depth=cfg_fs.get("rf_max_depth", None),
        n_jobs=-1, random_state=seed, class_weight="balanced",
    )
    rfe = RFE(estimator=rf, n_features_to_select=rfe_n, step=0.1)
    rfe.fit(X.values, y)
    rfe_ranking = {c: int(r) for c, r in zip(feature_cols, rfe.ranking_)}
    rfe_kept = [c for c in feature_cols if rfe_ranking[c] == 1]

    # MI on the RFE survivors
    mi = mutual_info_classif(X[rfe_kept].values, y, random_state=seed)
    mi_scores = {c: float(s) for c, s in zip(rfe_kept, mi)}
    ranked = sorted(rfe_kept, key=lambda c: mi_scores[c], reverse=True)

    if cfg_fs.get("final_n_features"):
        k = min(int(cfg_fs["final_n_features"]), len(ranked))
    else:
        frac = float(cfg_fs.get("mi_top_fraction", 0.40))
        k = max(1, int(round(frac * n_candidate)))
        k = min(k, len(ranked))
    retained = ranked[:k]

    detail = []
    for c in feature_cols:
        detail.append({
            "feature": c,
            "is_categorical": c in schema.categorical_columns,
            "rfe_rank": rfe_ranking[c],
            "survived_rfe": rfe_ranking[c] == 1,
            "mi_score": round(mi_scores.get(c, float("nan")), 6) if c in mi_scores else "",
            "retained": c in retained,
            "reason": ("retained: survived RFE and in MI top-%d" % k) if c in retained
                      else ("dropped: eliminated by RFE (rank %d)" % rfe_ranking[c]
                            if rfe_ranking[c] != 1 else "dropped: below MI cutoff"),
        })
    detail.sort(key=lambda d: (not d["retained"], d["rfe_rank"],
                               -(d["mi_score"] if isinstance(d["mi_score"], float) else 0)))

    return FeatureSelectionResult(
        retained=retained, rfe_ranking=rfe_ranking, mi_scores=mi_scores,
        n_candidate=n_candidate, detail=detail,
    )
