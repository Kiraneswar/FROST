"""Stage 5 zero-day evaluation: fuse the autoencoder + classifier signals, sweep
fusion weights, and report per-withheld-family detection on the ToN_IoT zone
evaluation set (with injected MITM/DDoS/DoS).

The ZAID reproduction's head-to-head numbers (target 83.16 / 97.69 / 67.86 /
79.40 acc/prec/rec/F1 on ToN_IoT zero-day) come from running this on the REAL
data; on the synthetic surrogate it only validates the fusion + metric machinery.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..fl.baselines import run_zaid
from ..fl.core import predict_proba
from ..fl.fed_data import FedData
from . import fusion, metrics


def _signals(artifacts, X: np.ndarray) -> Dict[str, np.ndarray]:
    lam_U = predict_proba(artifacts["universal"], X)
    lam_C = predict_proba(artifacts["classifier"], X)
    errs = artifacts["autoencoder"].reconstruction_error(
        torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()
    return {"U": lam_U, "C": lam_C, "A_err": errs}


def default_weight_grid() -> List[dict]:
    """ZAID's reported configs plus a small proposed sweep (weights on A,C,U)."""
    return [
        {"name": "classifier_only", "wA": 0.0, "wC": 1.0, "wU": 0.0, "thr": 0.5},
        {"name": "zaid_best", "wA": 1.0, "wC": 1.5, "wU": 1.5, "thr": 2.5 * 0.5},
        {"name": "equal", "wA": 1.0, "wC": 1.0, "wU": 1.0, "thr": 1.5 * 0.5},
        {"name": "ae_heavy", "wA": 1.5, "wC": 1.0, "wU": 1.0, "thr": 1.75 * 0.5},
        {"name": "proposed_A2C1U1", "wA": 2.0, "wC": 1.0, "wU": 1.0, "thr": 2.0 * 0.5},
        {"name": "proposed_A1_5C1_5U1", "wA": 1.5, "wC": 1.5, "wU": 1.0, "thr": 2.0 * 0.5},
    ]


def run_zeroday_eval(fd: FedData, cfg: dict, logger, seed: int) -> Dict:
    logger.info("=== STAGE 5: zero-day fusion evaluation ===")
    schema_benign = "normal"
    withheld = [f.lower() for f in cfg["zeroday"]["withheld_families"]]

    # train ZAID-style pipeline -> universal / adapter / autoencoder signals
    z = run_zaid(fd, cfg, logger, seed, tag="zaid_eval")
    art = z["artifacts"]

    # calibrate lambda_A on benign in-distribution test errors
    benign_err = art["autoencoder"].reconstruction_error(
        torch.as_tensor(fd.Xtest[fd.ytest == 0], dtype=torch.float32)).cpu().numpy()
    norm = fusion.ReconNormalizer.fit(benign_err)

    Xp, yp, famp = fd.pooled_eval()
    sig = _signals(art, Xp)
    lam_A = norm.transform(sig["A_err"])
    lam_C, lam_U = sig["C"], sig["U"]

    # aggregate-accuracy gap flag
    pos_rate = float(yp.mean())
    logger.info(f"eval set: n={len(yp)}, attack rate={pos_rate:.3f}, "
                f"zero-day fraction={float(np.isin(famp, withheld).mean()):.4f} "
                f"-> aggregate accuracy is misleading; report per-family recall")

    sweep_rows, per_family_best, best = [], None, None
    for w in default_weight_grid():
        AS = fusion.fuse(lam_A, lam_C, lam_U, w["wA"], w["wC"], w["wU"])
        pred = fusion.predict(AS, w["thr"])
        om = metrics.overall_metrics(yp, pred)
        pf = metrics.per_family_report(yp, pred, famp, schema_benign, withheld)
        zd = metrics.zero_day_summary(pf)
        row = {"weights": w["name"], "wA": w["wA"], "wC": w["wC"], "wU": w["wU"],
               "thr": round(w["thr"], 3), **{k: round(v, 4) for k, v in om.items() if k != "n"},
               "zero_day_recall": zd.get("zero_day_weighted_recall", 0.0)}
        sweep_rows.append(row)
        logger.event("fusion_sweep", **row)
        # select best by zero-day recall then F1 (zero-day detection is the goal)
        key = (row["zero_day_recall"], row["f1"])
        if best is None or key > best[0]:
            best = (key, w["name"]); per_family_best = pf
        logger.info(f"  [{w['name']:20s}] acc={om['accuracy']:.3f} f1={om['f1']:.3f} "
                    f"zero-day recall={row['zero_day_recall']:.3f}")

    logger.table("fusion_sweep", sweep_rows,
                 fieldnames=["weights", "wA", "wC", "wU", "thr", "accuracy",
                             "precision", "recall", "f1", "zero_day_recall"])
    logger.table("per_family_best", per_family_best,
                 fieldnames=["family", "is_zero_day", "n", "recall",
                             "precision_vs_benign", "f1_vs_benign"])
    logger.info(f"best fusion config: {best[1]} (by zero-day recall, then F1)")
    return {"sweep": sweep_rows, "per_family_best": per_family_best,
            "best_config": best[1], "attack_rate": pos_rate,
            "artifacts": art}
