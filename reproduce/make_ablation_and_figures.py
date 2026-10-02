#!/usr/bin/env python3
"""FAN: hyper-parameter ablation tables (Blur 0.9 where the feature path is active + corrected Noise 0.5)
from the pinned-environment re-runs, and the seed-42 strategy bar / heatmap from the single-execution values of
make_tables.py (FAN = multi-seed seed-42 run). Styling copied from scripts/legacy/generate_final_figures.py.
Writes into outputs/paper/{tables,figures}."""
import json, os
from pathlib import Path
from statistics import mean
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from repro_paths import ART, RUNS, TABLES, FIGURES
OUT = str(RUNS)
def ap(exp, strat): return json.load(open(f"{OUT}/{exp}/{strat}/results.json"))["aggregate_metrics"]["mean_ap50"] * 100
V = [("Default (layers $\\{2,4,6\\}$, $s{=}0.4$, $\\alpha{=}0.05$)", "default"), ("Layers $\\{2,4\\}$", "layers24"), ("Layers $\\{4,6\\}$", "layers46"),
     ("Layers $\\{2,4,6,8\\}$", "layers2468"), ("strength $s = 0.2$", "strength02"), ("strength $s = 0.6$", "strength06"), ("EMA $\\alpha = 0.2$", "ema02")]
nb, db = ap("repro_ablblur_no_adapt", "no_adapt"), ap("repro_ablblur_default", "fan")
nn, dn = ap("repro_no_adapt", "no_adapt"), ap("repro_abl_default", "fan")
L = ["\\begin{table}[t]", "\\centering", "\\caption{\\textbf{FAN hyper-parameter ablation} (seed 42, version-pinned re-run). On Blur~0.9 the "
     "feature-correction path is active; on Noise~0.5 FAN takes the input-smoothing path, so the feature-path settings are not exercised "
     "and the residual spread is run-to-run variation. Layer settings choose the hooked layers; the corrected subset for blur is "
     "$\\{4,6\\}$ (Appendix~\\ref{app:implementation}).}", "\\label{tab:ablation}", "\\small",
     "\\begin{tabular}{lcccc}", "\\toprule", " & \\multicolumn{2}{c}{Blur 0.9} & \\multicolumn{2}{c}{Noise 0.5} \\\\",
     "Variant & AP50 (\\%) & $\\Delta$ & AP50 (\\%) & $\\Delta$ \\\\", "\\midrule",
     f"No Adapt & {nb:.2f} & {nb-db:+.2f} & {nn:.2f} & {nn-dn:+.2f} \\\\"]
for lab, k in V:
    b = db if k == "default" else ap(f"repro_ablblur_{k}", "fan"); n = dn if k == "default" else ap(f"repro_abl_{k}", "fan")
    L.append(f"{lab} & {b:.2f} & {'--' if k=='default' else f'{b-db:+.2f}'} & {n:.2f} & {'--' if k=='default' else f'{n-dn:+.2f}'} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(TABLES / "table3_ablation.tex").write_text("\n".join(L) + "\n")
spread_b = [ap(f"repro_ablblur_{k}", "fan") for _, k in V[1:]] + [db]
print(f"ablation: blur default {db:.2f} no-adapt {nb:.2f} spread {max(spread_b)-min(spread_b):.2f}; noise default {dn:.3f} no-adapt {nn:.3f}")

# ---- seed-42 strategy bar + heatmap (single execution per strategy) ----
MS = json.load(open(ART / "multiseed_v13_5strat.json")); CP = json.load(open(ART / "comprehensive_v6v9v10_results.json"))
C = list(MS); S = ["fan", "tent", "eata", "cotta", "sar", "rotta", "bn_adapt", "periodic_ft"]
D = {"fan": "FAN (ours)", "tent": "TENT", "eata": "EATA", "cotta": "CoTTA", "sar": "SAR", "rotta": "RoTTA", "bn_adapt": "BN Adapt", "periodic_ft": "Periodic FT"}
E = ["Noise 0.3", "Noise 0.5", "Noise 0.7", "Noise 0.9", "Noise 1.0", "Blur 0.5", "Blur 0.7", "Blur 0.9", "Bright 0.7", "Bright 0.9", "Contrast 0.7", "Fog 0.7", "Gradual", "Recurring", "Mixed"]
def val(c, s): return MS[c][s]["42"] if s in ("fan", "tent", "eata", "bn_adapt", "no_adapt") else CP[c][s]["ap50"] * 100
plt.rcParams.update({"font.size": 10, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "savefig.dpi": 300, "savefig.bbox": "tight"})
avg = {s: mean(val(c, s) - val(c, "no_adapt") for c in C) for s in S}
order = sorted(S, key=lambda s: avg[s]); d = [avg[s] for s in order]
col = ["#2ca02c" if x > 0.1 else "#d62728" if x < -0.1 else "#7f7f7f" for x in d]
fig, ax = plt.subplots(figsize=(7, 3.5)); y = np.arange(len(order))
bars = ax.barh(y, d, color=col, edgecolor="black", linewidth=0.5, height=0.6)
bars[order.index("fan")].set_edgecolor("#1a5276"); bars[order.index("fan")].set_linewidth(2)
ax.set_yticks(y); ax.set_yticklabels([D[s] for s in order]); ax.set_xlabel("Average ΔAP50 vs No Adapt (%)")
ax.axvline(0, color="black", linewidth=0.8); ax.grid(axis="x", alpha=0.3)
for b_, v in zip(bars, d):
    ax.text(b_.get_width() + (0.15 if v >= 0 else -0.15), b_.get_y() + b_.get_height() / 2, f"{v:+.3f}", ha="left" if v >= 0 else "right", va="center", fontsize=8)
ax.legend(handles=[mpatches.Patch(color="#2ca02c", label="Helps"), mpatches.Patch(color="#7f7f7f", label="Neutral"),
                   mpatches.Patch(color="#d62728", label="Hurts")], loc="lower left", fontsize=8, framealpha=0.9)
plt.tight_layout(); plt.savefig(FIGURES / "strategy_bar_horizontal.pdf", bbox_inches="tight"); plt.close()
M = np.array([[val(c, s) - val(c, "no_adapt") for s in S] for c in C])
fig, ax = plt.subplots(figsize=(10, 7)); norm = TwoSlopeNorm(vmin=np.nanmin(M), vcenter=0, vmax=max(np.nanmax(M), 0.1))
im = ax.imshow(M, cmap="RdYlGn", norm=norm, aspect="auto")
ax.set_xticks(range(len(S))); ax.set_xticklabels([D[s] for s in S], rotation=45, ha="right"); ax.set_yticks(range(len(C))); ax.set_yticklabels(E)
for i in range(len(C)):
    for j in range(len(S)): ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7, color="white" if abs(M[i, j]) > 5 else "black")
plt.colorbar(im, ax=ax, label="ΔAP50 vs No Adapt (%)", shrink=0.8); ax.set_title("Strategy Impact on AP50 Across Distribution Shifts")
plt.tight_layout(); plt.savefig(FIGURES / "heatmap_all_results.pdf", bbox_inches="tight"); plt.close()
print("bar values:", {D[s]: round(avg[s], 3) for s in S})
