"""Stage 4 regression checks. Run: python tests/test_stage4.py
Verifies robust aggregators reject an injected outlier, compression achieves a
ratio > 1 with bounded error, staleness weighting decays, and the Byzantine set
is deterministic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fedzd.fl.aggregate_robust import coord_median, coord_trimmed_mean
from fedzd.fl.attacks import select_byzantine
from fedzd.fl.compression import apply_compression
from fedzd.fl.core import fedavg
from fedzd.fl.staleness import staleness_weight


def test_robust_rejects_outlier():
    honest = [{"w": torch.zeros(50) + 0.1 * i} for i in range(6)]
    poison = [{"w": torch.zeros(50) + 100.0}, {"w": torch.zeros(50) - 100.0}]
    states = honest + poison
    honest_mean = torch.stack([h["w"] for h in honest]).mean(0)
    mean_all = fedavg(states, [1] * len(states))["w"]
    med = coord_median(states)["w"]
    tm = coord_trimmed_mean(states, trim_ratio=0.25)["w"]
    # median & trimmed-mean must stay far closer to the honest mean than plain mean
    d_mean = (mean_all - honest_mean).abs().mean().item()
    d_med = (med - honest_mean).abs().mean().item()
    d_tm = (tm - honest_mean).abs().mean().item()
    assert d_med < d_mean and d_tm < d_mean, (d_mean, d_med, d_tm)
    assert d_med < 1.0 and d_tm < 1.0


def test_compression_ratio_and_error():
    torch.manual_seed(0)
    base = {"w": torch.zeros(1000)}
    update = {"w": torch.randn(1000)}
    recon, ratio = apply_compression(update, base, {"method": "topk", "k_frac": 0.1})
    assert ratio > 1.0, ratio
    recon_q, ratio_q = apply_compression(update, base, {"method": "quantize", "bits": 8})
    assert ratio_q == 4.0
    # 8-bit quant should reconstruct closely
    assert (recon_q["w"] - update["w"]).abs().mean().item() < 0.05


def test_staleness_decay():
    ws = [staleness_weight(s, "exponential", lam=0.3) for s in range(6)]
    assert all(ws[i] > ws[i + 1] for i in range(len(ws) - 1))
    assert staleness_weight(0, "constant") == 1.0 and staleness_weight(5, "constant") == 1.0


def test_byzantine_deterministic():
    a = select_byzantine(4, 0.5, seed=0)
    b = select_byzantine(4, 0.5, seed=0)
    assert a == b and len(a) == 2
    assert select_byzantine(4, 0.0, seed=0) == set()


if __name__ == "__main__":
    test_robust_rejects_outlier()
    test_compression_ratio_and_error()
    test_staleness_decay()
    test_byzantine_deterministic()
    print("OK: Stage 4 invariants hold (robust rules reject outliers, compression "
          "ratio>1 w/ bounded error, staleness decays, Byzantine set deterministic)")
