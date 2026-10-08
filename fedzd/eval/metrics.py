"""Zero-day evaluation metrics.

The withheld families are a tiny fraction of the evaluation set, so AGGREGATE
accuracy is misleading -- a detector that ignores them entirely still scores
high. We therefore report, in addition to overall precision/recall/F1:
  * per-withheld-family recall (the true zero-day detection rate), and
  * per-family precision/F1 in a one-vs-benign framing,
and flag the accuracy-imbalance gap explicitly.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def overall_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "n": int(len(y_true)),
    }


def per_family_report(y_true: np.ndarray, pred: np.ndarray, families: np.ndarray,
                      benign_name: str, withheld: List[str]) -> List[dict]:
    """One row per attack family. Recall = detection rate for that family.
    Precision/F1 computed on {benign + this family} so FPs are attributable."""
    rows = []
    benign_mask = families == benign_name
    for fam in sorted(set(families.tolist())):
        if fam == benign_name:
            continue
        fam_mask = families == fam
        # recall: fraction of this family flagged as attack
        recall = float(pred[fam_mask].mean()) if fam_mask.any() else 0.0
        # one-vs-benign subset for precision/f1
        sub = fam_mask | benign_mask
        yt = y_true[sub]; pp = pred[sub]
        rows.append({
            "family": fam,
            "is_zero_day": fam in withheld,
            "n": int(fam_mask.sum()),
            "recall": round(recall, 4),
            "precision_vs_benign": round(float(precision_score(yt, pp, zero_division=0)), 4),
            "f1_vs_benign": round(float(f1_score(yt, pp, zero_division=0)), 4),
        })
    return rows


def zero_day_summary(per_family: List[dict]) -> Dict[str, float]:
    zd = [r for r in per_family if r["is_zero_day"]]
    if not zd:
        return {}
    tot = sum(r["n"] for r in zd)
    wrec = sum(r["recall"] * r["n"] for r in zd) / max(1, tot)
    return {"zero_day_families": [r["family"] for r in zd],
            "zero_day_weighted_recall": round(wrec, 4),
            "zero_day_mean_recall": round(float(np.mean([r["recall"] for r in zd])), 4)}
