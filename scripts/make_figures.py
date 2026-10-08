#!/usr/bin/env python3
"""Stage 7: generate all paper figures and markdown tables from a Stage 6 run.

Reads the CSVs a Stage 6 run wrote (ablation_table, byzantine_table,
convergence_curves) and produces publication-ready figures plus a tables.md the
paper can lift directly.

Usage:
  python scripts/make_figures.py --run runs/<stage6_run_dir> [--outdir figures]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})
# colorblind-friendly palette
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def read_csv(p: Path):
    with open(p) as f:
        return list(csv.DictReader(f))


def fig_ablation(run: Path, outdir: Path):
    rows = read_csv(run / "ablation_table.csv")
    labels = [r["stage"].replace("_", " ").replace("1 ", "") for r in rows]
    means = [float(r["acc_mean"]) for r in rows]
    stds = [float(r["acc_std"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(range(len(means)), means, yerr=stds, capsize=4,
           color=PAL[0], alpha=0.9, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (mean $\\pm$ std)")
    ax.set_title("Incremental ablation: marginal contribution of each mechanism")
    ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(outdir / "fig_ablation.png"); plt.close(fig)


def fig_byzantine(run: Path, outdir: Path):
    rows = read_csv(run / "byzantine_table.csv")
    rules = [r["aggregation"].replace("_", "\n") for r in rows]
    means = [float(r["acc_mean"]) for r in rows]
    stds = [float(r["acc_std"]) for r in rows]
    attack = rows[0]["attack"] if rows else "attack"
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    colors = [PAL[3] if "mean" == r["aggregation"] else PAL[2] for r in rows]
    ax.bar(range(len(means)), means, yerr=stds, capsize=4, color=colors,
           alpha=0.9, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(rules))); ax.set_xticklabels(rules)
    ax.set_ylabel("Accuracy (mean $\\pm$ std)")
    ax.set_title(f"Byzantine robustness under {attack} poisoning")
    ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(outdir / "fig_byzantine.png"); plt.close(fig)


def fig_convergence(run: Path, outdir: Path):
    rows = read_csv(run / "convergence_curves.csv")
    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append((int(r["step_index"]), float(r["acc_mean"]),
                                float(r["acc_std"])))
    ladder = ["1_sync_ddil", "2_+async", "3_+staleness", "4_+resource", "5_+compression"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ci = 0
    for cfg in ladder:
        if cfg not in by:
            continue
        pts = sorted(by[cfg]); xs = [p[0] for p in pts]
        ms = [p[1] for p in pts]; ss = [p[2] for p in pts]
        color = PAL[ci % len(PAL)]; ci += 1
        ax.plot(xs, ms, marker="o", ms=3, color=color,
                label=cfg.replace("_", " "))
        ax.fill_between(xs, [m - s for m, s in zip(ms, ss)],
                        [m + s for m, s in zip(ms, ss)], color=color, alpha=0.15)
    ax.set_xlabel("Aggregation step / round"); ax.set_ylabel("Accuracy")
    ax.set_title("Convergence curves (mean $\\pm$ std across seeds)")
    ax.legend(fontsize=8, loc="lower right"); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(outdir / "fig_convergence.png"); plt.close(fig)


def write_tables_md(run: Path, outdir: Path):
    def md_table(rows, cols, headers):
        out = ["| " + " | ".join(headers) + " |",
               "|" + "|".join(["---"] * len(headers)) + "|"]
        for r in rows:
            out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return "\n".join(out)

    abl = read_csv(run / "ablation_table.csv")
    byz = read_csv(run / "byzantine_table.csv")
    md = ["# Generated result tables\n",
          f"_Source run: `{run.name}`_\n",
          "## Ablation ladder (accuracy, mean ± std)\n",
          md_table(abl, ["stage", "acc_mean", "acc_std", "f1_mean", "f1_std",
                         "marginal_delta_acc"],
                   ["Stage", "Acc mean", "Acc std", "F1 mean", "F1 std", "Δacc"]),
          "\n\n## Byzantine robustness (accuracy, mean ± std)\n",
          md_table(byz, ["attack", "aggregation", "acc_mean", "acc_std",
                         "f1_mean", "f1_std"],
                   ["Attack", "Aggregation", "Acc mean", "Acc std", "F1 mean", "F1 std"]),
          ""]
    (outdir / "tables.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    outdir = Path(args.outdir) if args.outdir else run / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    fig_ablation(run, outdir)
    fig_byzantine(run, outdir)
    fig_convergence(run, outdir)
    write_tables_md(run, outdir)
    print(f"Figures + tables written to: {outdir}")
    for p in sorted(outdir.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
