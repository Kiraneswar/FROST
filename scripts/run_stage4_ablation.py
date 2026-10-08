#!/usr/bin/env python3
"""Stage 4: incremental ablation of the proposed FedZD mechanisms + a Byzantine
defense comparison.

Ablation ladder (clean, no attack): naive async -> +staleness -> +resource-aware
-> +compression, each adding one mechanism, so the marginal contribution is
visible. Byzantine block: under a fixed attack, compare mean vs trimmed-mean vs
median aggregation to show robustness recovery.

Usage:
  python scripts/run_stage4_ablation.py --config configs/base.yaml configs/stage3_smoke.yaml \
      --set seed=0 --tag stage4
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fedzd.data.pipeline import run_stage1
from fedzd.fl.fed_data import build_fed_data
from fedzd.fl.proposed import run_proposed
from fedzd.sim.environment import DDILEnvironment
from fedzd.utils.config import load_config, make_run_context
from fedzd.utils.metriclog import RunLogger
from fedzd.utils.seeding import seed_everything


def variant(cfg, **over):
    """Deep-copy cfg and apply nested overrides like proposed.staleness.kind=..."""
    c = copy.deepcopy(cfg)
    for dotted, val in over.items():
        keys = dotted.split("."); cur = c
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = val
    return c


def run_one(fd, cfg, logger, seed, ddil_preset, tag):
    sim = copy.deepcopy(cfg["simulation"]); sim.setdefault("channel", {})["preset"] = ddil_preset
    env = DDILEnvironment(fd.n_zones, sim, seed=seed)
    return run_proposed(fd, cfg, logger, seed, env=env, tag=tag)["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--tag", default="stage4")
    ap.add_argument("--ddil", default="moderate")
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    ctx = make_run_context(cfg, runs_root=args.runs_root, tag=args.tag)
    seed_everything(ctx.seed)
    logger = RunLogger(ctx.run_dir)
    logger.info(f"=== STAGE 4: proposed-framework ablation (seed={ctx.seed}, "
                f"DDIL={args.ddil}) ===")

    bundle = run_stage1(ctx, logger)
    fd = build_fed_data(bundle)
    seed = ctx.seed

    # --- clean ablation ladder (no attack) ---
    ladder = []
    steps = [
        ("naive_async", dict(**{"proposed.staleness.kind": "constant",
                                "proposed.resource_aware": False,
                                "proposed.aggregation": "mean",
                                "compression.method": "none"})),
        ("+staleness", {"proposed.staleness.kind": "exponential",
                        "proposed.resource_aware": False,
                        "proposed.aggregation": "mean", "compression.method": "none"}),
        ("+resource_aware", {"proposed.staleness.kind": "exponential",
                             "proposed.resource_aware": True,
                             "proposed.aggregation": "mean", "compression.method": "none"}),
        ("+compression", {"proposed.staleness.kind": "exponential",
                          "proposed.resource_aware": True,
                          "proposed.aggregation": "mean",
                          "compression.method": "quantize", "compression.bits": 8}),
    ]
    for name, over in steps:
        c = variant(cfg, **over)
        s = run_one(fd, c, logger, seed, args.ddil, tag=name)
        ladder.append({"variant": name,
                       "acc_seen": round(s["final_test_seen"]["accuracy"], 4),
                       "f1_seen": round(s["final_test_seen"]["f1"], 4),
                       "acc_zone": round(s["final_zone_eval"]["accuracy"], 4),
                       "mean_staleness": s["mean_staleness"],
                       "comp_ratio": s["mean_compression_ratio"],
                       "participation": str(s["participation_per_zone"])})
    logger.table("ablation_ladder", ladder,
                 fieldnames=["variant", "acc_seen", "f1_seen", "acc_zone",
                             "mean_staleness", "comp_ratio", "participation"])

    # --- Byzantine defense comparison (fixed attack, swap aggregation rule) ---
    byz_rows = []
    for attack in ["label_flip", "gaussian"]:
        for rule in ["mean", "trimmed_mean", "median"]:
            c = variant(cfg,
                        **{"proposed.staleness.kind": "exponential",
                           "proposed.resource_aware": False,
                           "proposed.aggregation": rule,
                           "byzantine.fraction": float(cfg.get("byzantine", {}).get("eval_fraction", 0.25)),
                           "byzantine.attack": attack,
                           "compression.method": "none"})
            s = run_one(fd, c, logger, seed, args.ddil, tag=f"byz_{attack}_{rule}")
            byz_rows.append({"attack": attack, "aggregation": rule,
                             "acc_seen": round(s["final_test_seen"]["accuracy"], 4),
                             "f1_seen": round(s["final_test_seen"]["f1"], 4),
                             "acc_zone": round(s["final_zone_eval"]["accuracy"], 4),
                             "byz_zones": str(s["byzantine_zones"])})
    # clean reference (no attack) for the same aggregation set
    for rule in ["mean", "trimmed_mean", "median"]:
        c = variant(cfg, **{"proposed.staleness.kind": "exponential",
                            "proposed.aggregation": rule,
                            "byzantine.attack": "none", "compression.method": "none"})
        s = run_one(fd, c, logger, seed, args.ddil, tag=f"clean_{rule}")
        byz_rows.append({"attack": "none", "aggregation": rule,
                         "acc_seen": round(s["final_test_seen"]["accuracy"], 4),
                         "f1_seen": round(s["final_test_seen"]["f1"], 4),
                         "acc_zone": round(s["final_zone_eval"]["accuracy"], 4),
                         "byz_zones": "[]"})
    logger.table("byzantine_defense", byz_rows,
                 fieldnames=["attack", "aggregation", "acc_seen", "f1_seen",
                             "acc_zone", "byz_zones"])
    # --- compression ratio vs accuracy sweep (Stage 4d) ---
    comp_rows = []
    sweeps = [("none", {}), ("topk", {"k_frac": 0.3}), ("topk", {"k_frac": 0.1}),
              ("quantize", {"bits": 8}), ("quantize", {"bits": 4})]
    for method, params in sweeps:
        over = {"proposed.staleness.kind": "exponential", "proposed.aggregation": "mean",
                "byzantine.attack": "none", "compression.method": method}
        for pk, pv in params.items():
            over[f"compression.{pk}"] = pv
        c = variant(cfg, **over)
        s = run_one(fd, c, logger, seed, args.ddil, tag=f"comp_{method}_{params}")
        comp_rows.append({"method": method, "params": str(params),
                          "compression_ratio": s["mean_compression_ratio"],
                          "acc_seen": round(s["final_test_seen"]["accuracy"], 4),
                          "f1_seen": round(s["final_test_seen"]["f1"], 4)})
    logger.table("compression_sweep", comp_rows,
                 fieldnames=["method", "params", "compression_ratio", "acc_seen", "f1_seen"])

    logger.dump_json("stage4_ablation.json",
                     {"ladder": ladder, "byzantine": byz_rows, "compression": comp_rows})
    logger.info("stage 4 complete.")
    print(f"\nStage 4 outputs in: {ctx.run_dir}")


if __name__ == "__main__":
    main()
