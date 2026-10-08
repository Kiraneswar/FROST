"""Deterministic seeding across python / numpy / torch.

Every experiment run is pinned to a single integer seed that is logged to the
run config. Reproducibility requirement (Stage 6): every reported number must
be traceable to a seed recorded on disk.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed all RNGs we use. Torch is seeded only if importable (Stage 1 has no
    torch dependency, later stages do)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Favor reproducibility over throughput; the paper needs determinism.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def new_rng(seed: int) -> np.random.Generator:
    """A local, isolated numpy Generator so different components (zone split,
    channel sim, poisoning) can derive independent streams from one master seed
    without stepping on the global RNG."""
    return np.random.default_rng(seed)
