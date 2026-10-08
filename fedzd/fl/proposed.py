"""The proposed FedZD trainer: staleness-weighted asynchronous aggregation with
optional resource-aware client selection, Byzantine-robust buffered aggregation,
and bandwidth-aware update compression. Every mechanism is a config toggle so the
Stage 6 ablation can enable them one at a time.

Loop (FedAsync + buffered robust aggregation, FedBuff-style):
  * advance the DDIL environment one tick;
  * a zone participates iff (resource-aware off, or its battery/CPU are ok) AND
    its link is up this tick;
  * a participating zone computes an update from the global it last pulled
    (Byzantine zones instead poison it: label-flip or Gaussian);
  * the update's delta is compressed, then buffered with its staleness;
  * when the buffer fills, the buffer is combined with the chosen aggregation
    rule (mean / trimmed-mean / median) and mixed into the global at rate
    base_alpha * staleness_weight(mean staleness).
"""
from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import torch

from .aggregate_robust import robust_aggregate, robust_delta_step
from .attacks import flip_labels, gaussian_update, select_byzantine
from .baselines import _eval_all, _new_model, _pos_weight
from .compression import apply_compression
from .core import clone_state, fedavg, load_state, local_train
from .fed_data import FedData
from .staleness import effective_alpha


def run_proposed(fd: FedData, cfg: dict, logger, seed: int, env, tag: str = "fedzd") -> Dict:
    fl = cfg["fl"]; pc = cfg["proposed"]; device = "cpu"
    steps = int(pc.get("async_steps", fl.get("async_steps", 40)))
    base_alpha = float(pc.get("base_alpha", 0.3))
    buffer_size = int(pc.get("buffer_size", 2))
    stale_cfg = pc.get("staleness", {"kind": "exponential", "lam": 0.3})
    resource_aware = bool(pc.get("resource_aware", False))
    rule = pc.get("aggregation", "mean")
    trim_ratio = float(pc.get("trim_ratio", 0.2))
    byz = cfg.get("byzantine", {})
    byz_frac = float(byz.get("fraction", 0.0))
    byz_attack = byz.get("attack", "none")
    byz_sigma = float(byz.get("sigma", 1.0))
    comp = cfg.get("compression", {"method": "none"})

    byz_set = select_byzantine(fd.n_zones, byz_frac, seed) if byz_attack != "none" else set()

    global_model = _new_model(fd.input_dim, fl, seed=seed)
    global_state = clone_state(global_model)
    version = 0
    pulled_version = [0] * fd.n_zones
    zone_state = [copy.deepcopy(global_state) for _ in range(fd.n_zones)]
    participation = [0] * fd.n_zones
    buffer: List = []
    comp_ratios: List[float] = []
    staleness_seen: List[int] = []
    per_step = []
    last_participants = set(range(fd.n_zones))

    for t in range(steps):
        states = env.step(participating=last_participants)
        participants = set()
        for z in range(fd.n_zones):
            resource_ok = states[z].resource_ok or (not resource_aware)
            if resource_ok:
                participants.add(z)          # eligible to train this tick
            if not (resource_ok and states[z].link_up):
                continue
            participation[z] += 1
            staleness = version - pulled_version[z]
            staleness_seen.append(staleness)
            is_byz = z in byz_set
            if is_byz and byz_attack == "gaussian":
                update = gaussian_update(zone_state[z], byz_sigma, seed + t * 100 + z)
            else:
                m = _new_model(fd.input_dim, fl, seed=seed + z)
                load_state(m, zone_state[z])
                y = flip_labels(fd.zone_ytr[z]) if (is_byz and byz_attack == "label_flip") \
                    else fd.zone_ytr[z]
                update, _ = local_train(m, fd.zone_Xtr[z], y,
                                        epochs=int(fl["local_epochs"]), lr=float(fl["lr"]),
                                        batch_size=int(fl["batch_size"]), device=device,
                                        pos_weight=_pos_weight(fd.zone_ytr[z]),
                                        seed=seed + t * 100 + z)
            recon, ratio = apply_compression(update, zone_state[z], comp)
            comp_ratios.append(ratio)
            buffer.append((recon, staleness))
            # zone pulls the current global for its next round
            zone_state[z] = copy.deepcopy(global_state)
            pulled_version[z] = version

            if len(buffer) >= buffer_size:
                mean_stale = float(np.mean([b[1] for b in buffer]))
                a_eff = effective_alpha(base_alpha, int(round(mean_stale)), stale_cfg)
                # aggregate buffered updates as deltas relative to the current
                # global (robust rules stay coherent under non-IID), apply at a_eff
                global_state = robust_delta_step(
                    global_state, [b[0] for b in buffer], rule, trim_ratio, a_eff)
                version += 1
                buffer = []
        last_participants = participants if resource_aware else set(range(fd.n_zones))

        if t % max(1, steps // 15) == 0 or t == steps - 1:
            load_state(global_model, global_state)
            ev = _eval_all(global_model, fd)
            rec = {"baseline": tag, "step": t, "version": version,
                   "acc_seen": round(ev["test_seen"]["accuracy"], 4),
                   "f1_seen": round(ev["test_seen"]["f1"], 4),
                   "acc_zone": round(ev["zone_eval"]["accuracy"], 4)}
            per_step.append(rec); logger.event(f"{tag}_step", **rec)

    load_state(global_model, global_state)
    final = _eval_all(global_model, fd)
    summary = {
        "baseline": tag,
        "final_test_seen": final["test_seen"],
        "final_zone_eval": final["zone_eval"],
        "config": {"staleness": stale_cfg, "resource_aware": resource_aware,
                   "aggregation": rule, "trim_ratio": trim_ratio,
                   "byzantine": {"fraction": byz_frac, "attack": byz_attack},
                   "compression": comp, "buffer_size": buffer_size,
                   "base_alpha": base_alpha},
        "byzantine_zones": sorted(byz_set),
        "participation_per_zone": participation,
        "mean_staleness": round(float(np.mean(staleness_seen)) if staleness_seen else 0, 3),
        "max_staleness": int(np.max(staleness_seen)) if staleness_seen else 0,
        "mean_compression_ratio": round(float(np.mean(comp_ratios)) if comp_ratios else 1.0, 3),
        "updates_applied": version,
    }
    logger.info(f"[{tag}] seen acc={final['test_seen']['accuracy']:.3f} "
                f"f1={final['test_seen']['f1']:.3f} | zone acc={final['zone_eval']['accuracy']:.3f} "
                f"| stale(mean)={summary['mean_staleness']} "
                f"| comp={summary['mean_compression_ratio']}x "
                f"| byz={byz_attack}({byz_frac}) rule={rule}")
    return {"per_step": per_step, "summary": summary,
            "artifacts": {"classifier": global_model}}
