"""DDIL environment: per-zone Gilbert-Elliott channel + node resource model,
advanced round by round. This is the single object the FL loop (Stages 3-4)
queries to know, for each zone each round: is the link up, what is the latency,
what is the battery/CPU, and is the node resource-available.

Determinism: every zone's channel and resource draw from independent RNG
streams derived from one master seed, so a whole scenario is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .channel import GEParams, GilbertElliottChannel, params_from_cfg as channel_cfg
from .resources import (NodeResource, ResourceParams,
                        params_from_cfg as resource_cfg)


@dataclass
class ZoneRoundState:
    zone: int
    round: int
    channel_state: str
    link_up: bool          # update delivered to server this round
    latency: float         # seconds (only meaningful if link_up)
    battery: float
    cpu_load: float
    resource_ok: bool      # meets battery/CPU thresholds
    active: bool           # was this zone asked to participate this round


class DDILEnvironment:
    def __init__(self, n_zones: int, cfg_sim: dict, seed: int):
        self.n_zones = n_zones
        self.cfg = cfg_sim
        gep: GEParams = channel_cfg(cfg_sim.get("channel", {}))
        rp: ResourceParams = resource_cfg(cfg_sim.get("resources", {}))
        self.gep, self.rp = gep, rp
        self.min_battery = float(cfg_sim.get("min_battery", 20.0))
        self.max_cpu = float(cfg_sim.get("max_cpu", 0.85))
        # independent streams per zone
        self.channels = [GilbertElliottChannel(gep, seed=seed * 1000 + z + 1)
                         for z in range(n_zones)]
        self.resources = [NodeResource(rp, seed=seed * 2000 + z + 1)
                          for z in range(n_zones)]
        self._round = 0

    def step(self, participating: Optional[Set[int]] = None) -> List[ZoneRoundState]:
        """Advance one round. `participating` = zones selected to train this round
        (affects battery drain); default: all zones active."""
        if participating is None:
            participating = set(range(self.n_zones))
        out = []
        for z in range(self.n_zones):
            active = z in participating
            cs = self.channels[z].step()
            rs = self.resources[z].step(active=active)
            ok = self.resources[z].available(self.min_battery, self.max_cpu)
            out.append(ZoneRoundState(
                zone=z, round=self._round, channel_state=cs.state,
                link_up=cs.delivered, latency=cs.latency,
                battery=rs.battery, cpu_load=rs.cpu_load,
                resource_ok=ok, active=active,
            ))
        self._round += 1
        return out

    def simulate(self, n_rounds: int) -> List[ZoneRoundState]:
        """Free-running trace (all zones active) for characterization/plots."""
        trace: List[ZoneRoundState] = []
        for _ in range(n_rounds):
            trace.extend(self.step())
        return trace


def trace_summary(trace: List[ZoneRoundState], n_zones: int) -> dict:
    """Aggregate stats used to sanity-check a scenario and to report DDIL severity."""
    import numpy as np

    rounds = max(s.round for s in trace) + 1
    per_zone = {}
    for z in range(n_zones):
        zs = [s for s in trace if s.zone == z]
        link = np.array([s.link_up for s in zs], dtype=float)
        bad = np.array([s.channel_state == "bad" for s in zs], dtype=float)
        batt = np.array([s.battery for s in zs], dtype=float)
        # mean outage burst length (consecutive not-delivered runs)
        bursts, cur = [], 0
        for up in link:
            if up < 0.5:
                cur += 1
            elif cur:
                bursts.append(cur); cur = 0
        if cur:
            bursts.append(cur)
        per_zone[z] = {
            "delivery_rate": round(float(link.mean()), 4),
            "bad_state_fraction": round(float(bad.mean()), 4),
            "mean_outage_burst": round(float(np.mean(bursts)) if bursts else 0.0, 3),
            "battery_final": round(float(batt[-1]), 2),
            "battery_min": round(float(batt.min()), 2),
        }
    overall_delivery = float(np.mean([s.link_up for s in trace]))
    return {
        "rounds": rounds,
        "overall_delivery_rate": round(overall_delivery, 4),
        "per_zone": per_zone,
    }
