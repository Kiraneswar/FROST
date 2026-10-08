"""Stage 5 regression checks. Run: python tests/test_stage5.py
Validates the fusion normalizer, weighted fusion + thresholding, and the
per-family / zero-day metric computations on controlled inputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from fedzd.eval import fusion, metrics


def test_recon_normalizer():
    benign = np.random.default_rng(0).normal(1.0, 0.1, size=5000)
    norm = fusion.ReconNormalizer.fit(benign)
    lo = norm.transform(np.array([benign.min()]))
    hi = norm.transform(np.array([benign.max() * 5]))
    assert 0.0 <= lo[0] <= 0.2 and hi[0] == 1.0
    # monotonic
    xs = np.linspace(0.5, 3.0, 20)
    ts = norm.transform(xs)
    assert all(ts[i] <= ts[i + 1] + 1e-9 for i in range(len(ts) - 1))


def test_fuse_and_predict():
    lamA = np.array([0.0, 1.0]); lamC = np.array([0.0, 1.0]); lamU = np.array([0.0, 1.0])
    AS = fusion.fuse(lamA, lamC, lamU, 1.0, 1.5, 1.5)
    assert np.allclose(AS, [0.0, 4.0])
    pred = fusion.predict(AS, 2.5)
    assert pred.tolist() == [0, 1]


def test_per_family_and_zero_day():
    # families: benign 'normal', seen 'xss', zero-day 'ddos'
    families = np.array(["normal"] * 4 + ["xss"] * 2 + ["ddos"] * 2)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # predict: all attacks caught except one ddos; one benign false-positive
    pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    rows = metrics.per_family_report(y_true, pred, families, "normal", ["ddos"])
    by = {r["family"]: r for r in rows}
    assert by["ddos"]["is_zero_day"] is True and by["xss"]["is_zero_day"] is False
    assert abs(by["ddos"]["recall"] - 0.5) < 1e-9   # 1 of 2 ddos caught
    assert abs(by["xss"]["recall"] - 1.0) < 1e-9
    zd = metrics.zero_day_summary(rows)
    assert abs(zd["zero_day_weighted_recall"] - 0.5) < 1e-9


if __name__ == "__main__":
    test_recon_normalizer()
    test_fuse_and_predict()
    test_per_family_and_zero_day()
    print("OK: Stage 5 invariants hold (recon normalizer monotonic+clipped, "
          "weighted fusion+threshold correct, per-family/zero-day metrics correct)")
