"""Stage 6 experimental rigor: multi-seed orchestration, mean +/- std
aggregation, the incremental ablation table, per-round convergence curves, and
automated sanity / anomaly flagging.

Every configuration is run across >= n_seeds random seeds; we report mean and
standard deviation, never single-run numbers. Raw per-round metrics are retained
so convergence curves (not just endpoints) can be plotted.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..data.pipeline import run_stage1
from ..fl.baselines import run_sync_fedavg
from ..fl.fed_data import build_fed_data
from ..fl.proposed import run_proposed
from ..sim.environment import DDILEnvironment
from ..utils.config import make_run_context


class NullLogger:
    """Silences inner per-run logging; Stage 6 records its own aggregates."""
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def event(self, *a, **k): pass
    def table(self, *a, **k): return None
    def dump_json(self, *a, **k): return None


def _set(cfg, **over):
    c = copy.deepcopy(cfg)
    for dotted, val in over.items():
        ks = dotted.split("."); cur = c
        for k in ks[:-1]:
            cur = cur.setdefault(k, {})
        cur[ks[-1]] = val
    return c


def _env(cfg, seed, preset):
    sim = copy.deepcopy(cfg["simulation"]); sim.setdefault("channel", {})["preset"] = preset
    return DDILEnvironment(int(cfg["zones"]["n_zones"]), sim, seed=seed)


# -- one seed: run every ladder + byzantine config, return final metrics + curves

def run_configs_one_seed(base_cfg: dict, seed: int, ddil: str,
                         bundle=None) -> Dict[str, dict]:
    """Returns {config_name: {'final': metrics_dict, 'curve': [acc per step]}}."""
    log = NullLogger()
    # Stage 1 once per seed (data split depends on seed)
    if bundle is None:
        ctx = make_run_context(_set(base_cfg, seed=seed), runs_root="runs",
                               tag=f"stage6_seed{seed}")
        bundle = run_stage1(ctx, log)
    fd = build_fed_data(bundle)
    out: Dict[str, dict] = {}

    def curve(res, key_step="per_step"):
        rows = res.get(key_step) or res.get("per_round") or []
        ck = "acc_seen"
        return [r.get(ck) for r in rows if ck in r]

    # ---- clean incremental ladder (no attack) ----
    # 1. baseline: synchronous FedAvg under DDIL
    r = run_sync_fedavg(fd, base_cfg, log, seed, env=_env(base_cfg, seed, ddil),
                        tag="sync_ddil")
    out["1_sync_ddil"] = {"final": r["summary"]["final_test_seen"],
                          "zone": r["summary"]["final_zone_eval"],
                          "curve": curve(r, "per_round")}
    # 2. + asynchronous (naive, constant staleness)
    c = _set(base_cfg, **{"proposed.staleness.kind": "constant",
                          "proposed.resource_aware": False,
                          "proposed.aggregation": "mean", "compression.method": "none",
                          "byzantine.attack": "none"})
    r = run_proposed(fd, c, log, seed, _env(c, seed, ddil), tag="async")
    out["2_+async"] = {"final": r["summary"]["final_test_seen"],
                       "zone": r["summary"]["final_zone_eval"], "curve": curve(r)}
    # 3. + staleness weighting
    c = _set(c, **{"proposed.staleness.kind": "exponential"})
    r = run_proposed(fd, c, log, seed, _env(c, seed, ddil), tag="staleness")
    out["3_+staleness"] = {"final": r["summary"]["final_test_seen"],
                           "zone": r["summary"]["final_zone_eval"], "curve": curve(r)}
    # 4. + resource-aware selection
    c = _set(c, **{"proposed.resource_aware": True})
    r = run_proposed(fd, c, log, seed, _env(c, seed, ddil), tag="resource")
    out["4_+resource"] = {"final": r["summary"]["final_test_seen"],
                          "zone": r["summary"]["final_zone_eval"], "curve": curve(r),
                          "participation": r["summary"]["participation_per_zone"]}
    # 5. + bandwidth-aware compression (quantization)
    c = _set(c, **{"compression.method": "quantize", "compression.bits": 8})
    r = run_proposed(fd, c, log, seed, _env(c, seed, ddil), tag="compression")
    out["5_+compression"] = {"final": r["summary"]["final_test_seen"],
                             "zone": r["summary"]["final_zone_eval"], "curve": curve(r),
                             "comp_ratio": r["summary"]["mean_compression_ratio"]}

    # ---- Byzantine-robust step: under attack, mean vs robust rules ----
    byz = base_cfg.get("byzantine", {})
    frac = float(byz.get("eval_fraction", 0.25))
    attack = byz.get("eval_attack", "gaussian")
    for rule in ["mean", "trimmed_mean", "median"]:
        c = _set(base_cfg, **{"proposed.staleness.kind": "exponential",
                              "proposed.resource_aware": False,
                              "proposed.aggregation": rule,
                              "byzantine.fraction": frac, "byzantine.attack": attack,
                              "compression.method": "none"})
        r = run_proposed(fd, c, log, seed, _env(c, seed, ddil), tag=f"byz_{rule}")
        out[f"byz_{attack}_{rule}"] = {"final": r["summary"]["final_test_seen"],
                                       "zone": r["summary"]["final_zone_eval"],
                                       "curve": curve(r)}
    return out


# -- aggregate across seeds ---------------------------------------------------

def aggregate_seeds(per_seed: List[Dict[str, dict]]) -> Dict[str, dict]:
    """{config: {metric: (mean,std), 'curve_mean':[...], 'curve_std':[...]}}"""
    configs = per_seed[0].keys()
    agg: Dict[str, dict] = {}
    for cfg_name in configs:
        metrics = ["accuracy", "precision", "recall", "f1"]
        stat = {}
        for m in metrics:
            vals = [s[cfg_name]["final"][m] for s in per_seed]
            stat[m] = (float(np.mean(vals)), float(np.std(vals)))
        for m in ["accuracy", "f1"]:
            vals = [s[cfg_name]["zone"][m] for s in per_seed]
            stat[f"zone_{m}"] = (float(np.mean(vals)), float(np.std(vals)))
        # convergence curve mean/std (align by min length across seeds)
        curves = [s[cfg_name].get("curve") or [] for s in per_seed]
        L = min((len(c) for c in curves if c), default=0)
        if L:
            arr = np.array([c[:L] for c in curves if len(c) >= L], dtype=float)
            stat["curve_mean"] = arr.mean(0).tolist()
            stat["curve_std"] = arr.std(0).tolist()
        agg[cfg_name] = stat
    return agg


def ablation_rows(agg: Dict[str, dict]) -> List[dict]:
    """Incremental ladder rows with marginal Delta-accuracy vs previous stage."""
    ladder = ["1_sync_ddil", "2_+async", "3_+staleness", "4_+resource", "5_+compression"]
    rows, prev = [], None
    for name in ladder:
        if name not in agg:
            continue
        a_m, a_s = agg[name]["accuracy"]
        f_m, f_s = agg[name]["f1"]
        delta = "" if prev is None else round(a_m - prev, 4)
        rows.append({"stage": name,
                     "acc_mean": round(a_m, 4), "acc_std": round(a_s, 4),
                     "f1_mean": round(f_m, 4), "f1_std": round(f_s, 4),
                     "marginal_delta_acc": delta})
        prev = a_m
    return rows


def byzantine_rows(agg: Dict[str, dict], attack: str) -> List[dict]:
    rows = []
    for rule in ["mean", "trimmed_mean", "median"]:
        key = f"byz_{attack}_{rule}"
        if key not in agg:
            continue
        a_m, a_s = agg[key]["accuracy"]; f_m, f_s = agg[key]["f1"]
        rows.append({"attack": attack, "aggregation": rule,
                     "acc_mean": round(a_m, 4), "acc_std": round(a_s, 4),
                     "f1_mean": round(f_m, 4), "f1_std": round(f_s, 4)})
    return rows


# -- automated sanity / anomaly flagging --------------------------------------

def sanity_flags(agg: Dict[str, dict], per_seed: List[dict], attack: str) -> List[dict]:
    """Flag results that are implausible or internally inconsistent, so no
    anomalous number is silently accepted (Stage 6 requirement)."""
    flags = []

    def add(sev, msg):
        flags.append({"severity": sev, "flag": msg})

    # 1. seeds must actually vary (else seeding is broken)
    zero_var = [c for c in agg if agg[c]["accuracy"][1] < 1e-9]
    if len(zero_var) == len(agg) and len(per_seed) > 1:
        add("high", "All configs have zero variance across seeds -- seeding may "
            "not be perturbing the pipeline.")

    # 2. any metric out of [0,1]
    for c, st in agg.items():
        for m in ("accuracy", "precision", "recall", "f1"):
            mean = st[m][0]
            if not (0.0 <= mean <= 1.0):
                add("high", f"{c}: {m} mean {mean:.3f} outside [0,1].")

    # 3. suspiciously perfect detection
    for c, st in agg.items():
        if st["recall"][0] >= 0.999 and st["accuracy"][0] >= 0.999:
            add("medium", f"{c}: near-perfect accuracy and recall "
                f"({st['accuracy'][0]:.3f}/{st['recall'][0]:.3f}) -- verify no label "
                "leakage before trusting.")

    # 4. Byzantine robustness sanity: robust rule should not be worse than mean
    #    under attack by a large margin (else robust agg mis-configured)
    m_key, t_key = f"byz_{attack}_mean", f"byz_{attack}_trimmed_mean"
    if m_key in agg and t_key in agg:
        if agg[t_key]["accuracy"][0] + 0.05 < agg[m_key]["accuracy"][0]:
            add("medium", f"Robust trimmed-mean underperforms plain mean under "
                f"{attack} attack by >5% -- check buffer size / trim ratio.")

    # 5. ladder monotonicity note (informational, not an error on synthetic)
    ladder = ["1_sync_ddil", "2_+async", "3_+staleness", "4_+resource", "5_+compression"]
    accs = [agg[c]["accuracy"][0] for c in ladder if c in agg]
    if accs and accs != sorted(accs):
        add("info", "Ablation ladder is non-monotonic in accuracy -- some mechanisms "
            "(async, resource-aware selection) trade a little accuracy for resilience "
            "and efficiency; this is expected, not an error.")

    if not flags:
        add("info", "No anomalies detected by automated checks.")
    return flags
