"""Per-node resource state: battery % and CPU load, evolving over simulated time.

Each tactical node carries a battery that drains when it participates in a round
(local training + transmission cost) and drains slowly when idle, with a
stochastic opportunistic recharge (e.g. solar / vehicle power in the field). CPU
load is sampled each round with occasional spikes (competing on-node workloads).

Stage 4's resource-aware client selection queries `available(min_battery,
max_cpu)`; the drain/recharge model is fully configurable so scenarios can range
from well-powered to power-starved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ResourceParams:
    battery_init: float = 100.0
    battery_init_jitter: float = 10.0      # per-node initial spread
    drain_active: float = 4.0              # % per round when participating
    drain_active_jitter: float = 1.0
    drain_idle: float = 0.5                # % per round when not selected
    recharge_prob: float = 0.25            # chance of a recharge event per round
    recharge_amount: float = 8.0           # % gained on a recharge event
    battery_min: float = 0.0
    battery_max: float = 100.0
    cpu_base: float = 0.35                 # baseline CPU load (0..1)
    cpu_jitter: float = 0.10
    cpu_spike_prob: float = 0.10           # chance of a load spike
    cpu_spike_amount: float = 0.5


PRESETS: Dict[str, ResourceParams] = {
    "abundant":  ResourceParams(drain_active=2.0, recharge_prob=0.4, recharge_amount=10.0,
                                cpu_base=0.25, cpu_spike_prob=0.05),
    "nominal":   ResourceParams(),  # defaults
    "constrained": ResourceParams(battery_init=70.0, drain_active=7.0, drain_idle=1.0,
                                  recharge_prob=0.15, recharge_amount=5.0,
                                  cpu_base=0.5, cpu_spike_prob=0.2),
}


def params_from_cfg(cfg_res: dict) -> ResourceParams:
    preset = cfg_res.get("preset", "nominal")
    if preset not in PRESETS:
        raise ValueError(f"unknown resource preset '{preset}'; have {list(PRESETS)}")
    base = PRESETS[preset]
    fields = ResourceParams.__dataclass_fields__.keys()
    overrides = {k: cfg_res[k] for k in fields if k in cfg_res}
    return ResourceParams(**{**base.__dict__, **overrides})


@dataclass
class ResourceState:
    battery: float
    cpu_load: float


class NodeResource:
    def __init__(self, params: ResourceParams, seed: int):
        self.p = params
        self.rng = np.random.default_rng(seed)
        b0 = params.battery_init + self.rng.normal(0, params.battery_init_jitter)
        self.battery = float(np.clip(b0, params.battery_min, params.battery_max))
        self.cpu_load = float(np.clip(params.cpu_base, 0, 1))

    def step(self, active: bool) -> ResourceState:
        # battery drain
        if active:
            drain = self.p.drain_active + abs(self.rng.normal(0, self.p.drain_active_jitter))
        else:
            drain = self.p.drain_idle
        self.battery -= drain
        # opportunistic recharge
        if self.rng.random() < self.p.recharge_prob:
            self.battery += self.p.recharge_amount
        self.battery = float(np.clip(self.battery, self.p.battery_min, self.p.battery_max))
        # cpu load sample (+ occasional spike)
        load = self.p.cpu_base + self.rng.normal(0, self.p.cpu_jitter)
        if self.rng.random() < self.p.cpu_spike_prob:
            load += self.p.cpu_spike_amount
        self.cpu_load = float(np.clip(load, 0.0, 1.0))
        return ResourceState(battery=self.battery, cpu_load=self.cpu_load)

    def available(self, min_battery: float, max_cpu: float) -> bool:
        return self.battery >= min_battery and self.cpu_load <= max_cpu
