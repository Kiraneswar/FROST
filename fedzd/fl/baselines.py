"""Stage 3 baselines (all run and logged, never skipped):

  1. sync FedAvg, ideal connectivity      -- sanity-check upper bound
  2. sync FedAvg, DDIL connectivity        -- explicit stall/straggler failure mode
                                              (logs wall-clock lost to stragglers)
  3. naive async FL, NO staleness weighting -- shows async alone is insufficient
  4. ZAID core (universal CNN + autoencoder + FedAvg) -- closest published baseline

Each returns a dict with per-round metrics (raw, for convergence curves) and a
final summary. The proposed mechanisms (Stage 4) reuse these same loops.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch

from ..sim.channel import GilbertElliottChannel
from .core import (binary_metrics, clone_state, fedavg, load_state, local_train,
                   predict_proba)
from .fed_data import FedData
from .models import ConvAutoencoder, UniversalCNN


def _pos_weight(y: np.ndarray) -> float:
    pos = max(1, int(y.sum())); neg = max(1, int(len(y) - y.sum()))
    return neg / pos


def _new_model(input_dim: int, cfg: dict, use_adapters=False, seed=0) -> UniversalCNN:
    torch.manual_seed(seed)
    return UniversalCNN(input_dim, dropout=float(cfg.get("dropout", 0.2)),
                        use_adapters=use_adapters)


def _eval_all(model, fd: FedData, thr=0.5) -> Dict[str, Dict]:
    m_seen = binary_metrics(fd.ytest, predict_proba(model, fd.Xtest), thr)
    Xp, yp, _ = fd.pooled_eval()
    m_zone = binary_metrics(yp, predict_proba(model, Xp), thr)
    return {"test_seen": m_seen, "zone_eval": m_zone}


# --- straggler timing for synchronous rounds under DDIL -----------------------

def _deliver_time(ch: GilbertElliottChannel, compute_s: float, max_attempts: int,
                  timeout_s: float):
    """Time (s) until this zone's update reaches the server, retrying through
    outages. Returns (time, attempts, delivered)."""
    t = compute_s
    for a in range(1, max_attempts + 1):
        step = ch.step()
        if step.delivered:
            return t + step.latency, a, True
        t += min(step.latency, timeout_s)   # failed attempt still burns time
    return t, max_attempts, False


# --- 1 & 2: synchronous FedAvg -----------------------------------------------

def run_sync_fedavg(fd: FedData, cfg: dict, logger, seed: int,
                    env=None, tag: str = "sync") -> Dict:
    fl = cfg["fl"]; rounds = int(fl["rounds"]); device = "cpu"
    compute_s = float(fl.get("local_compute_s", 1.0))
    max_attempts = int(fl.get("max_delivery_attempts", 50))
    timeout_s = float(fl.get("delivery_timeout_s", 5.0))
    ideal_latency = float(fl.get("ideal_latency_s", 0.2))

    global_model = _new_model(fd.input_dim, fl, seed=seed)
    global_state = clone_state(global_model)
    per_round = []
    cum_wall = 0.0; cum_ideal = 0.0

    for r in range(rounds):
        states, weights = [], []
        round_wall = compute_s + ideal_latency  # baseline (fastest possible)
        zone_times = []
        for z in range(fd.n_zones):
            m = _new_model(fd.input_dim, fl, seed=seed + z)
            load_state(m, global_state)
            st, n = local_train(m, fd.zone_Xtr[z], fd.zone_ytr[z],
                                epochs=int(fl["local_epochs"]), lr=float(fl["lr"]),
                                batch_size=int(fl["batch_size"]), device=device,
                                pos_weight=_pos_weight(fd.zone_ytr[z]),
                                seed=seed + r * 100 + z)
            states.append(st); weights.append(n)
            # timing: sync must wait for the slowest zone
            if env is not None:
                t, attempts, delivered = _deliver_time(
                    env.channels[z], compute_s, max_attempts, timeout_s)
                zone_times.append(t)
        if env is not None and zone_times:
            round_wall = max(zone_times)
        cum_wall += round_wall
        cum_ideal += compute_s + ideal_latency

        global_state = fedavg(states, weights)
        load_state(global_model, global_state)
        ev = _eval_all(global_model, fd)
        straggler_loss = round_wall - (compute_s + ideal_latency)
        rec = {"baseline": tag, "round": r,
               "acc_seen": round(ev["test_seen"]["accuracy"], 4),
               "f1_seen": round(ev["test_seen"]["f1"], 4),
               "acc_zone": round(ev["zone_eval"]["accuracy"], 4),
               "f1_zone": round(ev["zone_eval"]["f1"], 4),
               "round_wallclock_s": round(round_wall, 3),
               "straggler_loss_s": round(max(0.0, straggler_loss), 3),
               "cum_wallclock_s": round(cum_wall, 3)}
        per_round.append(rec)
        logger.event(f"{tag}_round", **rec)

    final = _eval_all(global_model, fd)
    summary = {"baseline": tag, "rounds": rounds,
               "final_test_seen": final["test_seen"],
               "final_zone_eval": final["zone_eval"],
               "total_wallclock_s": round(cum_wall, 2),
               "ideal_wallclock_s": round(cum_ideal, 2),
               "total_straggler_loss_s": round(cum_wall - cum_ideal, 2)}
    logger.info(f"[{tag}] final seen acc={final['test_seen']['accuracy']:.3f} "
                f"f1={final['test_seen']['f1']:.3f} | zone acc={final['zone_eval']['accuracy']:.3f} "
                f"| wall={cum_wall:.1f}s (straggler loss {cum_wall-cum_ideal:.1f}s)")
    return {"per_round": per_round, "summary": summary}


# --- 3: naive asynchronous FL (no staleness weighting) -----------------------

def run_naive_async(fd: FedData, cfg: dict, logger, seed: int, env,
                    tag: str = "naive_async") -> Dict:
    fl = cfg["fl"]; steps = int(fl.get("async_steps", fl["rounds"] * fd.n_zones))
    alpha = float(fl.get("async_mix", 0.3)); device = "cpu"

    global_model = _new_model(fd.input_dim, fl, seed=seed)
    global_state = clone_state(global_model)
    global_version = 0
    pulled_version = [0] * fd.n_zones
    zone_state = [copy.deepcopy(global_state) for _ in range(fd.n_zones)]
    per_step = []; staleness_seen = []

    for t in range(steps):
        states = env.step()  # advance channels/resources one tick
        for z in range(fd.n_zones):
            if not states[z].link_up:
                continue
            # zone trains from the global it last pulled
            m = _new_model(fd.input_dim, fl, seed=seed + z)
            load_state(m, zone_state[z])
            st, n = local_train(m, fd.zone_Xtr[z], fd.zone_ytr[z],
                                epochs=int(fl["local_epochs"]), lr=float(fl["lr"]),
                                batch_size=int(fl["batch_size"]), device=device,
                                pos_weight=_pos_weight(fd.zone_ytr[z]),
                                seed=seed + t * 100 + z)
            staleness = global_version - pulled_version[z]
            staleness_seen.append(staleness)
            # NAIVE: equal-weight mix regardless of staleness
            global_state = fedavg([global_state, st], [1.0 - alpha, alpha])
            global_version += 1
            # zone pulls fresh global
            zone_state[z] = copy.deepcopy(global_state)
            pulled_version[z] = global_version
        if t % max(1, steps // 20) == 0 or t == steps - 1:
            load_state(global_model, global_state)
            ev = _eval_all(global_model, fd)
            rec = {"baseline": tag, "step": t, "global_version": global_version,
                   "acc_seen": round(ev["test_seen"]["accuracy"], 4),
                   "f1_seen": round(ev["test_seen"]["f1"], 4),
                   "acc_zone": round(ev["zone_eval"]["accuracy"], 4)}
            per_step.append(rec); logger.event(f"{tag}_step", **rec)

    load_state(global_model, global_state)
    final = _eval_all(global_model, fd)
    summary = {"baseline": tag, "steps": steps, "updates_applied": global_version,
               "final_test_seen": final["test_seen"],
               "final_zone_eval": final["zone_eval"],
               "mean_staleness": round(float(np.mean(staleness_seen)) if staleness_seen else 0, 3),
               "max_staleness": int(np.max(staleness_seen)) if staleness_seen else 0}
    logger.info(f"[{tag}] final seen acc={final['test_seen']['accuracy']:.3f} "
                f"f1={final['test_seen']['f1']:.3f} | mean staleness={summary['mean_staleness']}")
    return {"per_step": per_step, "summary": summary}


# --- 4: ZAID core (universal CNN + adapters via FedAvg + autoencoder) ---------

def run_zaid(fd: FedData, cfg: dict, logger, seed: int, tag: str = "zaid") -> Dict:
    fl = cfg["fl"]; zc = cfg.get("zaid", {}); device = "cpu"

    # (a) train universal backbone centrally on seen universal_train
    uni = _new_model(fd.input_dim, fl, use_adapters=True, seed=seed)
    local_train(uni, fd.Xu_tr, fd.yu_tr,
                epochs=int(zc.get("universal_epochs", 5)), lr=float(zc.get("universal_lr", 1e-3)),
                batch_size=int(fl["batch_size"]), device=device,
                pos_weight=_pos_weight(fd.yu_tr), seed=seed)
    uni_state = clone_state(uni)
    val = binary_metrics(fd.yu_val, predict_proba(uni, fd.Xu_val))
    logger.info(f"[{tag}] universal backbone val acc={val['accuracy']:.3f} f1={val['f1']:.3f}")

    # (b) freeze backbone; personalize per-zone adapters+head via FedAvg rounds
    global_model = _new_model(fd.input_dim, fl, use_adapters=True, seed=seed)
    load_state(global_model, uni_state)
    global_model.freeze_backbone()
    global_state = clone_state(global_model)
    per_round = []
    for r in range(int(zc.get("adapter_rounds", fl["rounds"]))):
        states, weights = [], []
        for z in range(fd.n_zones):
            m = _new_model(fd.input_dim, fl, use_adapters=True, seed=seed + z)
            load_state(m, global_state); m.freeze_backbone()
            st, n = local_train(m, fd.zone_Xtr[z], fd.zone_ytr[z],
                                epochs=int(zc.get("adapter_local_epochs", 2)),
                                lr=float(zc.get("adapter_lr", 5e-4)),
                                batch_size=int(fl["batch_size"]), device=device,
                                pos_weight=_pos_weight(fd.zone_ytr[z]),
                                only_params=m.adapter_and_head_parameters(),
                                seed=seed + r * 100 + z)
            states.append(st); weights.append(n)
        global_state = fedavg(states, weights)
        load_state(global_model, global_state)
        ev = _eval_all(global_model, fd)
        rec = {"baseline": tag, "round": r,
               "acc_seen": round(ev["test_seen"]["accuracy"], 4),
               "f1_seen": round(ev["test_seen"]["f1"], 4),
               "acc_zone": round(ev["zone_eval"]["accuracy"], 4)}
        per_round.append(rec); logger.event(f"{tag}_round", **rec)

    # (c) autoencoder on benign universal_train (unsupervised anomaly signal)
    ae = ConvAutoencoder(fd.input_dim, dropout=float(fl.get("dropout", 0.2)))
    _train_autoencoder(ae, fd.Xu_tr[fd.yu_tr == 0],
                       epochs=int(zc.get("ae_epochs", 5)), lr=float(zc.get("ae_lr", 1e-3)),
                       batch_size=int(fl["batch_size"]), seed=seed)
    rec_err_benign = ae.reconstruction_error(torch.as_tensor(
        fd.Xtest[fd.ytest == 0][:2000], dtype=torch.float32)).mean().item()
    logger.info(f"[{tag}] autoencoder trained; mean benign recon err={rec_err_benign:.4f}")

    final = _eval_all(global_model, fd)
    summary = {"baseline": tag, "universal_val": val,
               "final_test_seen": final["test_seen"],
               "final_zone_eval": final["zone_eval"],
               "benign_recon_err": round(rec_err_benign, 6)}
    # stash trained artifacts for Stage 5 fusion eval:
    #   universal  -> lambda_U signal (pre-adapter backbone classifier)
    #   classifier -> lambda_C signal (zone-personalized adapter model)
    #   autoencoder-> lambda_A signal (reconstruction error)
    return {"per_round": per_round, "summary": summary,
            "artifacts": {"universal": uni, "classifier": global_model,
                          "autoencoder": ae}}


def _train_autoencoder(ae, X_benign, *, epochs, lr, batch_size, seed=0):
    import torch.nn as nn
    g = torch.Generator().manual_seed(seed)
    Xt = torch.as_tensor(X_benign, dtype=torch.float32)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ae.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            r = ae(Xt[idx])
            loss = loss_fn(r, Xt[idx])
            loss.backward(); opt.step()
    return ae
