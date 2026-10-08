"""Turn a Stage-1 bundle into model-ready tensors: the central universal-train
splits and per-zone train/eval arrays, all through the fitted preprocessor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class FedData:
    input_dim: int
    Xu_tr: np.ndarray; yu_tr: np.ndarray            # universal train (seen)
    Xu_val: np.ndarray; yu_val: np.ndarray          # universal val (seen)
    Xtest: np.ndarray; ytest: np.ndarray            # in-distribution seen test
    zone_Xtr: List[np.ndarray]; zone_ytr: List[np.ndarray]
    zone_Xev: List[np.ndarray]; zone_yev: List[np.ndarray]
    zone_fam_ev: List[np.ndarray]                   # family labels for zone eval
    n_zones: int

    def pooled_eval(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.concatenate(self.zone_Xev); y = np.concatenate(self.zone_yev)
        f = np.concatenate(self.zone_fam_ev)
        return X, y, f


def build_fed_data(bundle) -> FedData:
    pre = bundle.preprocessor
    sp = bundle.splits
    zp = bundle.zones
    Xu_tr, yu_tr = pre.xy(sp.universal_train)
    Xu_val, yu_val = pre.xy(sp.universal_val)
    Xtest, ytest = pre.xy(sp.test_seen)
    zXtr = [pre.transform(df) for df in zp.zone_train]
    zytr = [pre.labels(df) for df in zp.zone_train]
    zXev = [pre.transform(df) for df in zp.zone_eval]
    zyev = [pre.labels(df) for df in zp.zone_eval]
    zfev = [pre.families(df) for df in zp.zone_eval]
    return FedData(
        input_dim=pre.n_features,
        Xu_tr=Xu_tr, yu_tr=yu_tr, Xu_val=Xu_val, yu_val=yu_val,
        Xtest=Xtest, ytest=ytest,
        zone_Xtr=zXtr, zone_ytr=zytr, zone_Xev=zXev, zone_yev=zyev,
        zone_fam_ev=zfev, n_zones=zp.n_zones,
    )
