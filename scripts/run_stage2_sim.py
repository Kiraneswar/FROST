#!/usr/bin/env python3
"""Characterize the DDIL environment (Stage 2): run a free-running channel +
resource trace and log per-round, per-zone state plus summary statistics.

By default it sweeps all four channel presets so you can see delivery-rate and
outage-burst growth from 'ideal' to 'severe' in one run.

Usage:
  python scripts/run_stage2_sim.py --config configs/base.yaml [--set ...] \
      [--sweep ideal mild moderate severe] [--tag stage2_sim]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedzd.sim.environment import DDILEnvironment, trace_summary
from fedzd.utils.config import load_config, make_run_context
from fedzd.utils.metriclog import RunLogger
from fedzd.utils.seeding import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--sweep", nargs="*", default=["ideal", "mild", "moderate", "severe"])
    ap.add_argument("--tag", default="stage2_sim")
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    ctx = make_run_context(cfg, runs_root=args.runs_root, tag=args.tag)
    seed_everything(ctx.seed)
    logger = RunLogger(ctx.run_dir)
    logger.info(f"=== STAGE 2: DDIL environment characterization (seed={ctx.seed}) ===")

    n_zones = int(cfg["zones"]["n_zones"])
    n_rounds = int(cfg["simulation"].get("characterize_rounds", 200))

    summary_rows = []
    for preset in args.sweep:
        sim_cfg = copy.deepcopy(cfg["simulation"])
        sim_cfg.setdefault("channel", {})["preset"] = preset
        env = DDILEnvironment(n_zones, sim_cfg, seed=ctx.seed)
        trace = env.simulate(n_rounds)
        summ = trace_summary(trace, n_zones)

        # log full per-round trace for this preset
        rows = [{"preset": preset, "zone": s.zone, "round": s.round,
                 "channel_state": s.channel_state, "link_up": int(s.link_up),
                 "latency": round(s.latency, 3), "battery": round(s.battery, 2),
                 "cpu_load": round(s.cpu_load, 3), "resource_ok": int(s.resource_ok)}
                for s in trace]
        logger.table(f"trace_{preset}", rows,
                     fieldnames=["preset", "zone", "round", "channel_state",
                                 "link_up", "latency", "battery", "cpu_load",
                                 "resource_ok"])
        # theory vs empirical check
        theo_bad = env.gep.stationary_bad
        theo_burst = env.gep.mean_bad_burst
        logger.info(f"[{preset:8s}] delivery={summ['overall_delivery_rate']:.3f} "
                    f"| theory P(bad)={theo_bad:.3f}, mean burst={theo_burst:.2f}")
        logger.event("stage2_preset", preset=preset, theory_bad=theo_bad,
                     theory_mean_burst=theo_burst, **summ)
        for z, zs in summ["per_zone"].items():
            summary_rows.append({"preset": preset, "zone": z,
                                 "theory_P_bad": round(theo_bad, 4),
                                 "theory_mean_burst": round(theo_burst, 3), **zs})

    logger.table("ddil_summary", summary_rows,
                 fieldnames=["preset", "zone", "theory_P_bad", "theory_mean_burst",
                             "delivery_rate", "bad_state_fraction",
                             "mean_outage_burst", "battery_final", "battery_min"])
    logger.info("stage 2 complete; traces + summary written.")
    print(f"\nStage 2 outputs in: {ctx.run_dir}")


if __name__ == "__main__":
    main()
