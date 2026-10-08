# FedZD / RA-BFed — Resilient Asynchronous Federated Learning for Zero-Day Intrusion Detection in Tactical (DDIL) Edge Networks

Research prototype extending **ZAID** (Pasdar et al., 2026, *Collaborative
Zone-Adaptive Zero-Day Intrusion Detection for IoBT*, arXiv:2602.16098) with
(1) asynchronous staleness-weighted aggregation, (2) resource-aware client
selection, (3) Byzantine-robust aggregation, and (4) bandwidth-aware update
compression, evaluated under a withheld-attack-family zero-day protocol.

All seven build stages are implemented, unit-tested, and reproducible from a
clean checkout.

## Reproducibility model (read this first)

- **Every hyperparameter lives in a YAML config** (`configs/`); no magic numbers
  in scripts. Each run snapshots its fully-resolved config + a content hash.
- **Every run is pinned to a logged `seed`**; multi-seed runs report
  **mean ± std**, never single-run numbers.
- **Raw per-round metrics** are saved (`events.jsonl`, per-run CSVs) so
  convergence curves — not just endpoints — can be plotted.
- `data.source` (`synthetic` | `csv`) is recorded in every run.
  **Synthetic-data runs validate the code only and are never reported in the
  paper.** Real numbers come from running the identical code with
  `data.source: csv` on the real datasets.

## Install

```bash
pip install -r requirements.txt
# torch: this repo was validated on CPU; use the CUDA wheel on your GPU box:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Datasets (recommended versions)

- **ToN_IoT** — *Processed Network dataset* (Zeek/Bro flow features). Families:
  `normal, backdoor, ddos, dos, injection, mitm, password, ransomware,
  scanning, xss`. ZAID's ~5.3M records come from the full `Network_dataset_*.csv`
  (~22M rows), not the balanced `Train_Test_Network.csv` (~461K).
- **UNSW-NB15** — canonical `UNSW_NB15_training-set.csv` (175,341) +
  `UNSW_NB15_testing-set.csv` (82,332), 45 columns.

Point the pipeline at real data by editing the `data` block (or via `--set`):

```yaml
data:
  schema: toniot_network
  source: csv
  csv_globs: ["/data/ToN_IoT/Network_dataset_*.csv"]
feature_selection:
  final_n_features: 32     # match ZAID's retained-feature count on real data
```

## The 7 stages and how to run each

Stages 3–6 need PyTorch. The `*_smoke.yaml` overlays make each stage run in
seconds on CPU/synthetic to validate the machinery; drop them (and set
`data.source: csv`) for real runs.

```bash
# Stage 1 — data pipeline (zero-day protocol, zones, RF+RFE->MI feature selection)
python scripts/run_stage1_data.py     --config configs/base.yaml --set seed=0 --tag s1

# Stage 2 — Gilbert-Elliott channel + battery/CPU resource simulation
python scripts/run_stage2_sim.py      --config configs/base.yaml --set seed=0 --tag s2

# Stage 3 — baselines (sync ideal, sync DDIL w/ straggler loss, naive async, ZAID)
python scripts/run_stage3_baselines.py --config configs/base.yaml configs/stage3_smoke.yaml --set seed=0 --tag s3

# Stage 4 — proposed framework ablation + Byzantine defense + compression sweep
python scripts/run_stage4_ablation.py --config configs/base.yaml configs/stage4_smoke.yaml --set seed=0 --ddil moderate --tag s4

# Stage 5 — zero-day fusion eval (weight sweep, per-family recall) + UNSW transfer
python scripts/run_stage5_zeroday.py  --config configs/base.yaml configs/stage3_smoke.yaml --set seed=0 --tag s5

# Stage 6 — multi-seed rigor: ablation table (mean±std) + Byzantine table + curves + flags
python scripts/run_stage6_rigor.py    --config configs/base.yaml configs/stage6_smoke.yaml --seeds 0 1 2 3 4 --tag s6

# Stage 7 — figures + markdown tables from a Stage 6 run
python scripts/make_figures.py --run runs/<stage6_run_dir>
```

## Reproducing the paper numbers (real data, GPU box)

```bash
# 1) Stage-6 rigor on real ToN_IoT, 5 seeds, full model sizes:
python scripts/run_stage6_rigor.py \
    --config configs/base.yaml \
    --set data.source=csv \
          data.csv_globs='["/data/ToN_IoT/Network_dataset_*.csv"]' \
          feature_selection.final_n_features=32 \
    --seeds 0 1 2 3 4 --tag paper_toniot

# 2) Zero-day fusion + UNSW cross-domain, real data:
python scripts/run_stage5_zeroday.py \
    --config configs/base.yaml \
    --set data.source=csv data.csv_globs='["/data/ToN_IoT/Network_dataset_*.csv"]' \
          cross_domain.unsw_data.source=csv \
          cross_domain.unsw_data.csv_globs='["/data/UNSW/UNSW_NB15_training-set.csv","/data/UNSW/UNSW_NB15_testing-set.csv"]' \
          feature_selection.final_n_features=32 --tag paper_zeroday

# 3) Figures + tables for the paper:
python scripts/make_figures.py --run runs/paper_toniot_*
```

The ZAID reproduction (baseline 4 in Stage 3 + fusion in Stage 5) is what to
check against ZAID's reported **ToN_IoT zero-day 83.16 / 97.69 / 67.86 / 79.40**
and **UNSW transfer 71.64 / 75.99 / 62.75 / 67.94** (acc/prec/rec/F1). The
Stage 6 runner's automated `sanity_flags.json` flags any result that is
out-of-range, suspiciously perfect, or inconsistent (e.g. a robust rule
underperforming plain mean under attack), so no anomalous number is silently
accepted.

## Tests

```bash
for t in tests/test_stage*.py; do python "$t"; done
```

Each stage has a regression test locking its key invariants (zero-day
non-leakage, Gilbert-Elliott matches theory, DDIL straggler loss, robust rules
reject outliers, fusion/metrics correctness, seed-aggregation + anomaly flags).

## Layout

```
fedzd/
  data/   schemas, loaders, synthetic surrogate, zero-day protocol, zones,
          RF+RFE->MI feature selection, preprocessing, Stage-1 orchestration
  sim/    Gilbert-Elliott channel, battery/CPU resources, DDIL environment
  fl/     models (CNN/AE/adapters), FL core, baselines, robust aggregation,
          staleness weighting, attacks, compression, proposed FedZD trainer
  eval/   fusion, per-family metrics, zero-day eval, cross-domain, rigor
  utils/  config, seeding, metric logging
configs/  base.yaml + per-stage smoke overlays
scripts/  one runner per stage + make_figures.py
tests/    per-stage regression tests
runs/     per-run outputs (configs, logs, metrics, figures)
```

## Honest notes on synthetic validation

The synthetic surrogate exists so the whole pipeline runs and is testable
without the multi-GB real datasets. On it, some mechanisms (staleness weighting,
resource-aware selection) do not show monotone accuracy gains and cross-domain
transfer is near-chance — both **expected**, because the surrogate is easy,
short-horizon, and has no shared ToN↔UNSW structure. Their benefits are meant to
appear under real high-staleness, severe-DDIL, adversarial, bandwidth-limited
conditions on the real data. Byzantine robustness (trimmed-mean recovering a
Gaussian-poisoning attack) and quantization compression (4–8× nearly free) *do*
show clearly even on synthetic.
