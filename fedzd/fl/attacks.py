"""Adversarial (Byzantine) client behaviors for robustness experiments.

  label_flip     : the client trains on flipped binary labels (data poisoning) --
                   its honestly-shaped update pulls the model toward misclassifying.
  gaussian       : the client ignores training and transmits the pulled global
                   plus large Gaussian noise (model poisoning / random update).

`select_byzantine` picks a deterministic malicious subset of zones given a
fraction, so a run's compromised set is reproducible from the seed.
"""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import torch


def select_byzantine(n_zones: int, fraction: float, seed: int) -> Set[int]:
    k = int(round(fraction * n_zones))
    if k <= 0:
        return set()
    rng = np.random.default_rng(seed + 4242)
    return set(rng.choice(n_zones, size=min(k, n_zones), replace=False).tolist())


def flip_labels(y: np.ndarray) -> np.ndarray:
    return 1.0 - y


def gaussian_update(pulled_state: Dict[str, torch.Tensor], sigma: float, seed: int
                    ) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in pulled_state.items():
        if v.dtype.is_floating_point:
            noise = torch.randn(v.shape, generator=g) * sigma
            out[k] = (v.float() + noise).to(v.dtype)
        else:
            out[k] = v.clone()
    return out
