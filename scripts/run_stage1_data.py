#!/usr/bin/env python3
"""Run Stage 1 (data pipeline) as a standalone, config-driven step.

Usage:
  python scripts/run_stage1_data.py --config configs/base.yaml [more.yaml ...] \
      [--set data.source=synthetic seed=0 ...] [--tag stage1_synth]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedzd.data.pipeline import run_stage1
from fedzd.utils.config import load_config, make_run_context
from fedzd.utils.metriclog import RunLogger
from fedzd.utils.seeding import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    ctx = make_run_context(cfg, runs_root=args.runs_root, tag=args.tag)
    seed_everything(ctx.seed)
    logger = RunLogger(ctx.run_dir)
    logger.info(f"run dir: {ctx.run_dir}  (config hash {ctx.cfg_hash})")
    run_stage1(ctx, logger)
    print(f"\nStage 1 outputs in: {ctx.run_dir}")


if __name__ == "__main__":
    main()
