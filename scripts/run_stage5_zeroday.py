#!/usr/bin/env python3
"""Stage 5: zero-day fusion evaluation on ToN_IoT + cross-domain transfer to
UNSW-NB15.

Usage:
  python scripts/run_stage5_zeroday.py --config configs/base.yaml configs/stage3_smoke.yaml \
      --set seed=0 --tag stage5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedzd.data.pipeline import run_stage1
from fedzd.eval.cross_domain import run_cross_domain
from fedzd.eval.zeroday_eval import run_zeroday_eval
from fedzd.fl.fed_data import build_fed_data
from fedzd.utils.config import load_config, make_run_context
from fedzd.utils.metriclog import RunLogger
from fedzd.utils.seeding import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--tag", default="stage5")
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    ctx = make_run_context(cfg, runs_root=args.runs_root, tag=args.tag)
    seed_everything(ctx.seed)
    logger = RunLogger(ctx.run_dir)
    logger.info(f"=== STAGE 5 (seed={ctx.seed}, source={cfg['data'].get('source')}) ===")

    bundle = run_stage1(ctx, logger)
    fd = build_fed_data(bundle)

    zd = run_zeroday_eval(fd, cfg, logger, ctx.seed)
    logger.dump_json("stage5_zeroday.json",
                     {"sweep": zd["sweep"], "per_family_best": zd["per_family_best"],
                      "best_config": zd["best_config"], "attack_rate": zd["attack_rate"]})

    if cfg.get("cross_domain", {}).get("enabled", False):
        cd = run_cross_domain(cfg, logger, ctx.seed)
        logger.dump_json("stage5_cross_domain.json", cd)

    logger.info("stage 5 complete.")
    print(f"\nStage 5 outputs in: {ctx.run_dir}")


if __name__ == "__main__":
    main()
