"""Cross-domain transfer to UNSW-NB15 (ZAID's cross-domain test).

ToN_IoT and UNSW-NB15 have different feature sets, so transfer uses the
documented SHARED_FEATURE_SPACE: the semantically-aligned flow features present
in both (duration, src/dst bytes, src/dst packets). We train the universal
classifier + autoencoder on ToN_IoT *seen* families restricted to those shared
features, then evaluate -- with no target-domain retraining -- on UNSW-NB15
mapped to the same shared features. ZAID's best cross-domain config was
classifier-only (0,1,0); we report classifier-only and fusion.

This conservative 5-feature bridge is intentionally simple and documented; the
team can widen it (more aligned features, or target-domain adapter re-fit) on the
real data. On synthetic surrogates this validates the transfer machinery only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ..data.loaders import load_raw
from ..data.schemas import SHARED_FEATURE_SPACE, DatasetSchema, SCHEMAS
from ..data.transforms import Preprocessor
from ..fl.baselines import _pos_weight, _train_autoencoder
from ..fl.core import binary_metrics, local_train, predict_proba
from ..fl.models import ConvAutoencoder, UniversalCNN
from . import fusion


def _shared_schema() -> DatasetSchema:
    return DatasetSchema(name="shared", id_columns=[], categorical_columns=[],
                         label_column="label", family_column="type",
                         benign_family="normal")


def to_shared_frame(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for canon in SHARED_FEATURE_SPACE:
        col = schema.shared_feature_map[canon]
        out[canon] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    out["label"] = df[schema.label_column].to_numpy().astype(int)
    out["type"] = df[schema.family_column].to_numpy()
    return out


def run_cross_domain(cfg: dict, logger, seed: int) -> Dict:
    logger.info("=== STAGE 5: cross-domain transfer ToN_IoT -> UNSW-NB15 ===")
    ss = _shared_schema()
    withheld = [f.lower() for f in cfg["zeroday"]["withheld_families"]]

    # source: ToN_IoT seen families (exclude withheld), shared features only
    src_df, src_schema = load_raw(cfg["data"], seed)
    src_shared = to_shared_frame(src_df, src_schema)
    src_shared = src_shared[~src_shared["type"].isin(withheld)].reset_index(drop=True)
    pre = Preprocessor.fit(src_shared, ss, SHARED_FEATURE_SPACE)
    Xs, ys = pre.xy(src_shared)

    clf = UniversalCNN(pre.n_features, use_adapters=False)
    local_train(clf, Xs, ys, epochs=int(cfg["zaid"].get("universal_epochs", 5)),
                lr=float(cfg["zaid"].get("universal_lr", 1e-3)),
                batch_size=int(cfg["fl"]["batch_size"]), pos_weight=_pos_weight(ys), seed=seed)
    ae = ConvAutoencoder(pre.n_features)
    _train_autoencoder(ae, Xs[ys == 0], epochs=int(cfg["zaid"].get("ae_epochs", 5)),
                       lr=float(cfg["zaid"].get("ae_lr", 1e-3)),
                       batch_size=int(cfg["fl"]["batch_size"]), seed=seed)
    benign_err = ae.reconstruction_error(torch.as_tensor(Xs[ys == 0][:3000],
                                         dtype=torch.float32)).cpu().numpy()
    norm = fusion.ReconNormalizer.fit(benign_err)

    # target: UNSW-NB15 mapped to the shared features (no retraining)
    tgt_df, tgt_schema = load_raw(cfg["cross_domain"]["unsw_data"], seed)
    tgt_shared = to_shared_frame(tgt_df, tgt_schema)
    Xt, yt = pre.xy(tgt_shared)

    lam_C = predict_proba(clf, Xt)
    lam_A = norm.transform(ae.reconstruction_error(
        torch.as_tensor(Xt, dtype=torch.float32)).cpu().numpy())

    rows = []
    configs = [{"name": "classifier_only", "wA": 0.0, "wC": 1.0, "thr": 0.5},
               {"name": "fusion_A1C1", "wA": 1.0, "wC": 1.0, "thr": 1.0 * 0.5}]
    for c in configs:
        AS = fusion.fuse(lam_A, lam_C, np.zeros_like(lam_C), c["wA"], c["wC"], 0.0)
        pred = fusion.predict(AS, c["thr"]).astype(float)
        m = binary_metrics(yt, pred, thr=0.5)  # pred is already 0/1
        rows.append({"config": c["name"],
                     **{k: round(m[k], 4) for k in ("accuracy", "precision", "recall", "f1")}})
        logger.info(f"  UNSW transfer [{c['name']:16s}] acc={m['accuracy']:.3f} "
                    f"prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f}")

    logger.table("cross_domain_unsw", rows,
                 fieldnames=["config", "accuracy", "precision", "recall", "f1"])
    return {"cross_domain": rows, "n_target": int(len(yt)),
            "target_attack_rate": float(yt.mean())}
