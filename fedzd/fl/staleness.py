"""Staleness weighting functions for asynchronous aggregation.

In async FL a client's update was computed from a global model that may be
several versions old; its influence should decay with that staleness `s`
(FedAsync, Xie et al. 2019; FedASMU, 2024). We expose the standard families:

  constant     : s(t) = 1                      (no weighting -> naive async)
  exponential  : s(t) = exp(-lambda * t)       (FedAsync exponential)
  polynomial   : s(t) = (1 + t) ** (-a)        (FedAsync polynomial / hinge)

The effective mixing rate at the server is base_alpha * s(staleness).
"""
from __future__ import annotations

import math


def staleness_weight(staleness: int, kind: str = "exponential",
                     lam: float = 0.3, a: float = 0.5) -> float:
    s = max(0, int(staleness))
    if kind == "constant":
        return 1.0
    if kind == "exponential":
        return math.exp(-lam * s)
    if kind == "polynomial":
        return (1.0 + s) ** (-a)
    raise ValueError(f"unknown staleness kind '{kind}'")


def effective_alpha(base_alpha: float, staleness: int, cfg: dict) -> float:
    kind = cfg.get("kind", "exponential")
    lam = float(cfg.get("lam", 0.3))
    a = float(cfg.get("a", 0.5))
    return base_alpha * staleness_weight(staleness, kind, lam, a)
