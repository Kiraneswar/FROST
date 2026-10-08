#!/usr/bin/env python3
"""Stage 6: multi-seed experimental rigor.

Runs the incremental ablation ladder and the Byzantine-robustness block across
>= 5 seeds, reports mean +/- std, saves per-round convergence curves, builds the
marginal-contribution ablation table, and runs automated sanity/anomaly checks.

Usage:
  python scripts/run_stage6_rigor.py --config configs/base.yaml configs/stage3_smoke.yaml \
      --set rigor.seeds=[0,1,2,3,4] --tag stage6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedzd.eval import rigor
from fedzd.utils.config import load_config, make_run_context
from fedzd.utils.metriclog import RunLogger
from fedzd.utils.seeding import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--tag", default="stage6")
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    seeds = args.seeds or cfg.get("rigor", {}).get("seeds", [0, 1, 2, 3, 4])
    ddil = cfg.get("rigor", {}).get("ddil_preset", "moderate")
    attack = cfg.get("byzantine", {}).get("eval_attack", "gaussian")

    ctx = make_run_context(cfg, runs_root=args.runs_root, tag=args.tag)
    seed_everything(0)
    logger = RunLogger(ctx.run_dir)
    logger.info(f"=== STAGE 6: rigor over seeds {seeds} (DDIL={ddil}, "
                f"byz attack={attack}) ===")

    per_seed = []
    for s in seeds:
        logger.info(f"--- seed {s} ---")
        res = rigor.run_configs_one_seed(cfg, s, ddil)
        per_seed.append(res)

    agg = rigor.aggregate_seeds(per_seed)

    # ablation table (marginal contributions)
    abl = rigor.ablation_rows(agg)
    logger.table("ablation_table", abl,
                 fieldnames=["stage", "acc_mean", "acc_std", "f1_mean", "f1_std",
                             "marginal_delta_acc"])
    logger.info("Ablation ladder (mean +/- std over %d seeds):" % len(seeds))
    for r in abl:
        logger.info(f"  {r['stage']:16s} acc={r['acc_mean']:.3f}±{r['acc_std']:.3f} "
                    f"f1={r['f1_mean']:.3f}±{r['f1_std']:.3f} "
                    f"Δacc={r['marginal_delta_acc']}")

    # byzantine robustness table
    byz = rigor.byzantine_rows(agg, attack)
    logger.table("byzantine_table", byz,
                 fieldnames=["attack", "aggregation", "acc_mean", "acc_std",
                             "f1_mean", "f1_std"])
    logger.info(f"Byzantine robustness under {attack} (mean±std):")
    for r in byz:
        logger.info(f"  {r['aggregation']:14s} acc={r['acc_mean']:.3f}±{r['acc_std']:.3f}")

    # convergence curves (long format for plotting)
    conv_rows = []
    for cname, st in agg.items():
        cm = st.get("curve_mean"); cs = st.get("curve_std")
        if not cm:
            continue
        for i, (m, sd) in enumerate(zip(cm, cs)):
            conv_rows.append({"config": cname, "step_index": i,
                              "acc_mean": round(m, 4), "acc_std": round(sd, 4)})
    logger.table("convergence_curves", conv_rows,
                 fieldnames=["config", "step_index", "acc_mean", "acc_std"])

    # sanity / anomaly flags
    flags = rigor.sanity_flags(agg, per_seed, attack)
    logger.dump_json("sanity_flags.json", flags)
    logger.info("Automated sanity/anomaly checks:")
    for f in flags:
        logger.info(f"  [{f['severity'].upper()}] {f['flag']}")

    logger.dump_json("stage6_summary.json", {
        "seeds": seeds, "ddil_preset": ddil, "byz_attack": attack,
        "ablation_table": abl, "byzantine_table": byz, "flags": flags,
    })
    logger.info("stage 6 complete.")
    print(f"\nStage 6 outputs in: {ctx.run_dir}")


if __name__ == "__main__":
    main()
