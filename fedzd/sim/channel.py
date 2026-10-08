"""Gilbert-Elliott two-state Markov channel model.

Models bursty, jamming-like connectivity between a tactical zone and the
aggregation server. Two hidden states:

    GOOD  -- link usable; updates delivered with prob (1 - loss_good), low latency
    BAD   -- jammed/degraded; updates delivered with prob (1 - loss_bad), high latency

State transitions each simulated step:
    P(GOOD -> BAD) = p_gb        P(BAD -> GOOD) = p_bg

The stationary probability of being in BAD is  p_gb / (p_gb + p_bg); the mean
BAD-burst (outage) length is 1 / p_bg steps. Both transition probabilities are
configurable so a scenario can be swept from "mild" to "severe" DDIL.

This is intentionally decoupled from the FL loop: a channel just answers, per
step, "is my update to the server delivered, and with what latency?"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

GOOD, BAD = "good", "bad"


@dataclass(frozen=True)
class GEParams:
    p_gb: float          # P(good -> bad) per step
    p_bg: float          # P(bad -> good) per step
    loss_good: float     # per-step delivery-failure prob while GOOD
    loss_bad: float      # per-step delivery-failure prob while BAD
    latency_good: float  # nominal latency (s) when delivered in GOOD
    latency_bad: float   # nominal latency (s) when delivered in BAD
    latency_jitter: float = 0.15  # multiplicative jitter std

    @property
    def stationary_bad(self) -> float:
        denom = self.p_gb + self.p_bg
        return self.p_gb / denom if denom > 0 else 0.0

    @property
    def mean_bad_burst(self) -> float:
        return (1.0 / self.p_bg) if self.p_bg > 0 else float("inf")


# DDIL severity presets. Transition probs chosen so BAD-burst length and outage
# fraction grow monotonically ideal -> severe. Latencies in seconds.
PRESETS: Dict[str, GEParams] = {
    "ideal":    GEParams(p_gb=0.00, p_bg=1.00, loss_good=0.0, loss_bad=1.0,
                         latency_good=0.20, latency_bad=5.0),
    "mild":     GEParams(p_gb=0.05, p_bg=0.60, loss_good=0.0, loss_bad=0.7,
                         latency_good=0.30, latency_bad=3.0),
    "moderate": GEParams(p_gb=0.15, p_bg=0.40, loss_good=0.02, loss_bad=0.9,
                         latency_good=0.50, latency_bad=6.0),
    "severe":   GEParams(p_gb=0.35, p_bg=0.20, loss_good=0.05, loss_bad=1.0,
                         latency_good=0.80, latency_bad=12.0),
}


def params_from_cfg(cfg_channel: dict) -> GEParams:
    """Build GEParams from a config block: a `preset` name plus optional
    explicit overrides of any field."""
    preset = cfg_channel.get("preset", "moderate")
    if preset not in PRESETS:
        raise ValueError(f"unknown channel preset '{preset}'; have {list(PRESETS)}")
    base = PRESETS[preset]
    overrides = {k: cfg_channel[k] for k in
                 ("p_gb", "p_bg", "loss_good", "loss_bad", "latency_good",
                  "latency_bad", "latency_jitter") if k in cfg_channel}
    return GEParams(**{**base.__dict__, **overrides})


@dataclass
class ChannelStep:
    state: str
    delivered: bool
    latency: float       # seconds; only meaningful when delivered


class GilbertElliottChannel:
    def __init__(self, params: GEParams, seed: int, start_state: str = GOOD):
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.state = start_state

    def step(self) -> ChannelStep:
        # 1) transition
        u = self.rng.random()
        if self.state == GOOD:
            if u < self.p.p_gb:
                self.state = BAD
        else:
            if u < self.p.p_bg:
                self.state = GOOD
        # 2) delivery + latency conditioned on the (new) state
        if self.state == GOOD:
            delivered = self.rng.random() >= self.p.loss_good
            base = self.p.latency_good
        else:
            delivered = self.rng.random() >= self.p.loss_bad
            base = self.p.latency_bad
        latency = float(base * (1.0 + self.rng.normal(0, self.p.latency_jitter)))
        latency = max(latency, 0.01)
        return ChannelStep(state=self.state, delivered=delivered, latency=latency)
