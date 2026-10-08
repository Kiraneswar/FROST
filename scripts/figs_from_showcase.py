#!/usr/bin/env python3
"""Generate paper figures + tables.md directly from a showcase_data JSON bundle
(real or synthetic). Usage: python scripts/figs_from_showcase.py showcase_data_real.json [outdir]"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"savefig.dpi":300,"font.size":10,"axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":True,"grid.alpha":.3})
PAL=["#0072B2","#E69F00","#009E73","#D55E00","#CC79A7"]

d=json.load(open(sys.argv[1]))
out=Path(sys.argv[2] if len(sys.argv)>2 else "figures_real"); out.mkdir(exist_ok=True)
src=d["meta"]["data_source"]; tag=f"real ToN_IoT" if src=="csv" else "synthetic"

# 1. ablation
a=d["ablation"]; lab=[r["stage"].replace("1_","").replace("_"," ") for r in a]
m=[r["acc_mean"] for r in a]; s=[r["acc_std"] for r in a]
fig,ax=plt.subplots(figsize=(6.4,3.4))
ax.bar(range(len(m)),m,yerr=s,capsize=4,color=PAL[0],edgecolor="black",lw=.6)
ax.set_xticks(range(len(lab)));ax.set_xticklabels(lab,rotation=20,ha="right")
ax.set_ylabel("Accuracy (mean ± std)");ax.set_ylim(0,1)
ax.set_title(f"Incremental ablation ({tag})");fig.tight_layout();fig.savefig(out/"fig_ablation.png");plt.close()

# 2. byzantine
b=d["byzantine"];order={"mean":0,"trimmed_mean":1,"median":2}
b=sorted(b,key=lambda r:order[r["aggregation"]])
fig,ax=plt.subplots(figsize=(5,3.4))
cols=[PAL[3] if r["aggregation"]=="mean" else PAL[2] for r in b]
ax.bar(range(len(b)),[r["acc_mean"] for r in b],yerr=[r["acc_std"] for r in b],
    capsize=4,color=cols,edgecolor="black",lw=.6)
ax.set_xticks(range(len(b)));ax.set_xticklabels([r["aggregation"].replace("_","\n") for r in b])
ax.set_ylabel("Accuracy (mean ± std)");ax.set_ylim(0,1)
ax.set_title(f"Byzantine robustness ({tag})");fig.tight_layout();fig.savefig(out/"fig_byzantine.png");plt.close()

# 3. per-family recall
pf=sorted(d["per_family"],key=lambda r:-r["recall"])
fig,ax=plt.subplots(figsize=(6.4,3.6))
cols=[PAL[3] if r["is_zero_day"] else PAL[0] for r in pf]
ax.barh(range(len(pf)),[r["recall"] for r in pf],color=cols,edgecolor="black",lw=.5)
ax.set_yticks(range(len(pf)));ax.set_yticklabels([r["family"]+(" ●" if r["is_zero_day"] else "") for r in pf])
ax.invert_yaxis();ax.set_xlabel("Recall");ax.set_xlim(0,1)
ax.set_title(f"Per-family detection recall ({tag}) — ● = zero-day")
fig.tight_layout();fig.savefig(out/"fig_perfamily.png");plt.close()

# 4. convergence
cm=d["convergence"]["acc_mean"];cs=d["convergence"]["acc_std"]
if cm:
    fig,ax=plt.subplots(figsize=(6.4,3.4));x=list(range(len(cm)))
    ax.plot(x,cm,marker="o",ms=3,color=PAL[2])
    ax.fill_between(x,[a-b for a,b in zip(cm,cs)],[a+b for a,b in zip(cm,cs)],color=PAL[2],alpha=.15)
    ax.set_xlabel("Aggregation step");ax.set_ylabel("Accuracy");ax.set_ylim(0,1)
    ax.set_title(f"Convergence, full model ({tag})");fig.tight_layout();fig.savefig(out/"fig_convergence.png");plt.close()

# tables.md
def tbl(rows,cols,hd):
    o=["| "+" | ".join(hd)+" |","|"+"|".join(["---"]*len(hd))+"|"]
    for r in rows:o.append("| "+" | ".join(str(r.get(c,"")) for c in cols)+" |")
    return "\n".join(o)
md=[f"# Results tables — {tag}\n",
 "## Ablation ladder (accuracy, mean ± std)\n",
 tbl(a,["stage","acc_mean","acc_std","f1_mean","marginal_delta_acc"],["Stage","Acc","±","F1","Δacc"]),
 "\n## Byzantine robustness (accuracy, mean ± std)\n",
 tbl(b,["attack","aggregation","acc_mean","acc_std"],["Attack","Rule","Acc","±"]),
 "\n## Per-family recall\n",
 tbl(pf,["family","is_zero_day","recall","n"],["Family","Zero-day","Recall","n"]),""]
(out/"tables.md").write_text("\n".join(md))
print("wrote figures + tables to",out)
for p in sorted(out.iterdir()):print("  ",p.name)
