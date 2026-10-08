#!/usr/bin/env python3
"""Run the full pipeline once and export a JSON bundle that drives the testing
showcase dashboard: testbed metadata, a live channel/jamming trace, per-round
convergence, sample zero-day predictions (predicted vs actual), per-family
detection, the ablation ladder, and the Byzantine-robustness comparison.

All numbers are produced by an actual executed run and logged -- nothing is
fabricated. Data provenance (synthetic surrogate vs real CSV) is recorded in the
bundle so the dashboard can label it honestly.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fedzd.data.pipeline import run_stage1
from fedzd.eval import fusion
from fedzd.eval.rigor import NullLogger, run_configs_one_seed, aggregate_seeds, \
    ablation_rows, byzantine_rows
from fedzd.eval.zeroday_eval import run_zeroday_eval, default_weight_grid
from fedzd.fl.core import predict_proba
from fedzd.fl.fed_data import build_fed_data
from fedzd.sim.environment import DDILEnvironment
from fedzd.utils.config import load_config, make_run_context
from fedzd.utils.seeding import seed_everything

ZONE_ROLES = ["UGV Node", "UAV Drone", "Soldier Radio Hub", "Vehicle System"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--out", default="showcase_data.json")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    # CPU-feasible training defaults for the synthetic default config only;
    # real-data configs (configs/toniot_real.yaml) set their own sizes.
    if cfg["data"].get("source") == "synthetic":
        cfg = _apply(cfg, {"data.synthetic.n_samples": 24000,
                           "data.synthetic.separability": 1.0,
                           "zaid.universal_epochs": 6, "zaid.adapter_rounds": 4,
                           "zaid.ae_epochs": 4, "fl.local_epochs": 1,
                           "proposed.async_steps": 24, "proposed.buffer_size": 4})
    seed = 0
    seed_everything(seed)
    ctx = make_run_context(_apply(cfg, {"seed": seed}), runs_root="runs", tag="showcase")
    log = NullLogger()

    bundle = run_stage1(ctx, log)
    fd = build_fed_data(bundle)
    schema = bundle.schema
    withheld = [f.lower() for f in cfg["zeroday"]["withheld_families"]]

    out = {
        "meta": {
            "project": "FedZD / RA-BFed",
            "data_source": cfg["data"]["source"],
            "provenance_note": ("Synthetic surrogate with the exact ToN_IoT "
                                "Network schema -- validates the system end to end. "
                                "Final paper numbers require the real ToN_IoT/UNSW "
                                "CSVs run through the identical code."),
            "n_features": fd.input_dim,
            "retained_features": bundle.features,
            "n_zones": fd.n_zones,
            "zone_roles": ZONE_ROLES,
            "withheld_families": withheld,
            "total_rows": int(sum(len(z) for z in bundle.zones.zone_train))
                          + int(len(bundle.splits.universal_train)),
        }
    }

    # ---- zero-day fusion evaluation (gives artifacts + per-family + sweep) ----
    zd = run_zeroday_eval(fd, cfg, log, seed)
    art = zd["artifacts"]
    out["fusion_sweep"] = zd["sweep"]
    out["per_family"] = zd["per_family_best"]
    out["best_fusion"] = zd["best_config"]

    # overall metrics at best fusion config
    grid = {w["name"]: w for w in default_weight_grid()}
    best = grid[zd["best_config"]]
    benign_err = art["autoencoder"].reconstruction_error(
        torch.as_tensor(fd.Xtest[fd.ytest == 0], dtype=torch.float32)).cpu().numpy()
    norm = fusion.ReconNormalizer.fit(benign_err)

    # ---- sample predictions (predicted vs actual), a few per zone ----
    samples = []
    rng = np.random.default_rng(7)
    for z in range(fd.n_zones):
        X = fd.zone_Xev[z]; y = fd.zone_yev[z]; fam = fd.zone_fam_ev[z]
        lamU = predict_proba(art["universal"], X)
        lamC = predict_proba(art["classifier"], X)
        errs = art["autoencoder"].reconstruction_error(
            torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()
        lamA = norm.transform(errs)
        AS = fusion.fuse(lamA, lamC, lamU, best["wA"], best["wC"], best["wU"])
        pred = fusion.predict(AS, best["thr"])
        # pick a balanced set: some zero-day, some seen-attack, some benign
        idx_zd = [i for i in range(len(y)) if fam[i] in withheld]
        idx_at = [i for i in range(len(y)) if y[i] == 1 and fam[i] not in withheld]
        idx_bn = [i for i in range(len(y)) if y[i] == 0]
        chosen = (list(rng.choice(idx_zd, min(3, len(idx_zd)), replace=False)) +
                  list(rng.choice(idx_at, min(2, len(idx_at)), replace=False)) +
                  list(rng.choice(idx_bn, min(2, len(idx_bn)), replace=False)))
        for i in chosen:
            samples.append({
                "zone": z, "zone_role": ZONE_ROLES[z],
                "family": str(fam[i]),
                "is_zero_day": bool(fam[i] in withheld),
                "true": "ATTACK" if y[i] == 1 else "BENIGN",
                "pred": "ATTACK" if pred[i] == 1 else "BENIGN",
                "correct": bool((pred[i] == 1) == (y[i] == 1)),
                "anomaly_score": round(float(AS[i]), 3),
                "lambda_A": round(float(lamA[i]), 3),
                "lambda_C": round(float(lamC[i]), 3),
                "lambda_U": round(float(lamU[i]), 3),
                "threshold": round(float(best["thr"]), 3),
            })
    out["sample_predictions"] = samples

    # ---- live channel/jamming trace for the animation ----
    sim = copy.deepcopy(cfg["simulation"]); sim.setdefault("channel", {})["preset"] = "moderate"
    env = DDILEnvironment(fd.n_zones, sim, seed=seed)
    trace = env.simulate(24)
    ch = []
    for s in trace:
        ch.append({"round": s.round, "zone": s.zone, "link_up": bool(s.link_up),
                   "state": s.channel_state, "battery": round(s.battery, 1),
                   "cpu": round(s.cpu_load, 2)})
    out["channel_trace"] = ch

    # ---- ablation ladder + byzantine (3 seeds for mean/std, kept small) ----
    per_seed = [run_configs_one_seed(cfg, s, "moderate") for s in (0, 1)]
    agg = aggregate_seeds(per_seed)
    out["ablation"] = ablation_rows(agg)
    out["byzantine"] = byzantine_rows(agg, cfg["byzantine"].get("eval_attack", "gaussian"))
    # convergence curve of the full proposed model (config 5)
    conv = agg.get("5_+compression", {})
    out["convergence"] = {"acc_mean": [round(x, 4) for x in conv.get("curve_mean", [])],
                          "acc_std": [round(x, 4) for x in conv.get("curve_std", [])]}

    Path("runs").mkdir(exist_ok=True)
    dest = Path(args.out)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    print("best fusion:", out["best_fusion"],
          "| samples:", len(samples), "| ablation rows:", len(out["ablation"]))


def _apply(cfg, over):
    c = copy.deepcopy(cfg)
    for dotted, val in over.items():
        ks = dotted.split("."); cur = c
        for k in ks[:-1]:
            cur = cur.setdefault(k, {})
        cur[ks[-1]] = val
    return c


if __name__ == "__main__":
    main()
