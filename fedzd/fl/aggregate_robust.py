"""Byzantine-robust aggregation rules, swappable with `fedavg`.

All take a list of state dicts (one per client update in the buffer) and return
one aggregated state dict, filtering outliers coordinate-wise so a bounded
fraction of poisoned updates cannot dominate.

  coord_median       : per-coordinate median across updates (Blanchard/Yin).
  coord_trimmed_mean : per-coordinate mean after trimming `trim_ratio` of the
                       highest and lowest values (Yin et al. 2018).

Robust rules are deliberately *unweighted* over the buffer: weighting by
client-reported sample counts is itself an attack surface. Staleness attenuation
is applied by the caller at the global-mix step, not inside these rules.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import torch


def coord_median(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = copy.deepcopy(states[0])
    for k in out:
        if out[k].dtype.is_floating_point:
            stacked = torch.stack([s[k].float() for s in states], dim=0)
            out[k] = stacked.median(dim=0).values.to(states[0][k].dtype)
        else:
            out[k] = states[0][k].clone()
    return out


def coord_trimmed_mean(states: List[Dict[str, torch.Tensor]], trim_ratio: float = 0.2
                       ) -> Dict[str, torch.Tensor]:
    m = len(states)
    k = int(trim_ratio * m)               # trim k from each end
    out = copy.deepcopy(states[0])
    for key in out:
        if not out[key].dtype.is_floating_point:
            out[key] = states[0][key].clone(); continue
        stacked = torch.stack([s[key].float() for s in states], dim=0)  # [m, ...]
        sorted_, _ = torch.sort(stacked, dim=0)
        lo, hi = k, m - k
        if hi <= lo:                       # too few to trim -> plain mean
            trimmed = stacked
        else:
            trimmed = sorted_[lo:hi]
        out[key] = trimmed.mean(dim=0).to(states[0][key].dtype)
    return out


def robust_aggregate(states: List[Dict[str, torch.Tensor]], rule: str,
                     trim_ratio: float = 0.2, weights=None) -> Dict[str, torch.Tensor]:
    from .core import fedavg
    if rule in ("mean", "fedavg", None):
        w = weights if weights is not None else [1.0] * len(states)
        return fedavg(states, w)
    if rule == "median":
        return coord_median(states)
    if rule == "trimmed_mean":
        return coord_trimmed_mean(states, trim_ratio)
    raise ValueError(f"unknown aggregation rule '{rule}'")


def robust_delta_step(global_state: Dict[str, torch.Tensor],
                      states: List[Dict[str, torch.Tensor]], rule: str,
                      trim_ratio: float, alpha: float) -> Dict[str, torch.Tensor]:
    """Aggregate the buffered updates as *deltas* relative to the current global,
    then apply the robustly-combined delta at rate `alpha`:

        new_global = global + alpha * robust_rule({update_i - global})

    Aggregating deltas (rather than raw weights) centers the robust rule at the
    current model, so coordinate-wise median / trimmed-mean stay coherent under
    non-IID clients while still discarding poisoned outliers.
    """
    float_keys = [k for k in global_state if global_state[k].dtype.is_floating_point]
    deltas = [{k: (s[k].float() - global_state[k].float()) for k in float_keys}
              for s in states]
    agg_delta = robust_aggregate(deltas, rule, trim_ratio)
    out = copy.deepcopy(global_state)
    for k in float_keys:
        out[k] = (global_state[k].float() + alpha * agg_delta[k]).to(global_state[k].dtype)
    return out
