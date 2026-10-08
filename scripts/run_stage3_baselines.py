#!/usr/bin/env python3
"""Stage 3: run all four baselines on one data split and log a comparison.

  1. sync FedAvg (ideal)      2. sync FedAvg (DDIL, straggler loss logged)
  3. naive async (no staleness)   4. ZAID core (universal + adapters + AE)

Usage:
  python scripts/run_stage3_baselines.py --config configs/base.yaml configs/stage3_smoke.yaml \
      --set seed=0 --tag stage3
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fedzd.data.pipeline import run_stage1
from fedzd.fl.baselines import (run_naive_async, run_sync_fedavg, run_zaid)
from fedzd.fl.fed_data import build_fed_data
from fedzd.sim.environment import DDILEnvironment
from fedzd.utils.config import load_config, make_run_context
from fedzd.utils.metriclog import RunLogger
from fedzd.utils.seeding import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--tag", default="stage3")
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    ctx = make_run_context(cfg, runs_root=args.runs_root, tag=args.tag)
    seed_everything(ctx.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    logger = RunLogger(ctx.run_dir)
    logger.info(f"=== STAGE 3: baselines (seed={ctx.seed}, "
                f"source={cfg['data'].get('source')}) ===")

    bundle = run_stage1(ctx, logger)
    fd = build_fed_data(bundle)
    logger.info(f"FedData: input_dim={fd.input_dim}, zones={fd.n_zones}")

    ddil = cfg["baselines"]["ddil_preset"]
    sim_ddil = copy.deepcopy(cfg["simulation"]); sim_ddil.setdefault("channel", {})["preset"] = ddil

    results = {}
    # 1. sync ideal
    results["sync_ideal"] = run_sync_fedavg(fd, cfg, logger, ctx.seed, env=None, tag="sync_ideal")
    # 2. sync DDIL
    env2 = DDILEnvironment(fd.n_zones, sim_ddil, seed=ctx.seed)
    results["sync_ddil"] = run_sync_fedavg(fd, cfg, logger, ctx.seed, env=env2, tag=f"sync_ddil_{ddil}")
    # 3. naive async
    env3 = DDILEnvironment(fd.n_zones, sim_ddil, seed=ctx.seed)
    results["naive_async"] = run_naive_async(fd, cfg, logger, ctx.seed, env=env3, tag="naive_async")
    # 4. ZAID
    results["zaid"] = run_zaid(fd, cfg, logger, ctx.seed, tag="zaid")

    # comparison table
    rows = []
    for key, res in results.items():
        s = res["summary"]
        rows.append({
            "baseline": s["baseline"],
            "acc_seen": round(s["final_test_seen"]["accuracy"], 4),
            "f1_seen": round(s["final_test_seen"]["f1"], 4),
            "acc_zone": round(s["final_zone_eval"]["accuracy"], 4),
            "f1_zone": round(s["final_zone_eval"]["f1"], 4),
            "wallclock_s": s.get("total_wallclock_s", ""),
            "straggler_loss_s": s.get("total_straggler_loss_s", ""),
            "mean_staleness": s.get("mean_staleness", ""),
        })
    logger.table("baseline_comparison", rows,
                 fieldnames=["baseline", "acc_seen", "f1_seen", "acc_zone",
                             "f1_zone", "wallclock_s", "straggler_loss_s",
                             "mean_staleness"])
    logger.dump_json("stage3_summaries.json", {k: v["summary"] for k, v in results.items()})
    logger.info("stage 3 complete.")
    print(f"\nStage 3 outputs in: {ctx.run_dir}")


if __name__ == "__main__":
    main()
