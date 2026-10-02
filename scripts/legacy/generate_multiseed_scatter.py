#!/usr/bin/env python3
"""Generate per-seed scatter plot for FAN vs No Adapt across 15 synthetic shifts.

Reads experiment_results/artifacts/multiseed_full_v12.json and writes
outputs/paper/figures/multiseed_scatter.pdf showing one point per (experiment, seed) pair
with FAN AP50 on y-axis and No Adapt AP50 on x-axis. Points above y=x mean FAN wins.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
JSON_PATH = os.path.join(ROOT, "experiment_results", "artifacts", "multiseed_full_v12.json")
OUT_PDF = os.path.join(ROOT, "outputs", "paper", "figures", "multiseed_scatter.pdf")

EXP_LABELS = {
    "noise_0_3": "Noise 0.3", "noise_0_5": "Noise 0.5", "noise_0_7": "Noise 0.7",
    "noise_0_9": "Noise 0.9", "noise_1_0": "Noise 1.0",
    "blur_0_5": "Blur 0.5", "blur_0_7": "Blur 0.7", "blur_0_9": "Blur 0.9",
    "brightness_0_7": "Bright 0.7", "brightness_0_9": "Bright 0.9",
    "contrast_0_7": "Contrast 0.7", "fog_0_7": "Fog 0.7",
    "gradual_brightness": "Gradual bright.", "recurring_blur": "Recurring blur",
    "mixed_shifts": "Mixed",
}
SEEDS = [42, 123, 456]

def main():
    data = json.load(open(JSON_PATH))
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    cmap = plt.get_cmap("tab20", len(EXP_LABELS))
    for i, (k, lbl) in enumerate(EXP_LABELS.items()):
        if k not in data: continue
        cell = data[k]
        xs, ys = [], []
        for s in SEEDS:
            ss = str(s)
            if ss in cell.get("no_adapt", {}) and ss in cell.get("fan", {}):
                xs.append(cell["no_adapt"][ss])
                ys.append(cell["fan"][ss])
        if xs:
            ax.scatter(xs, ys, color=cmap(i), s=22, label=lbl, edgecolor="k", linewidth=0.3)
    lo, hi = 60, 75
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.8, label="$y=x$ (parity)")
    ax.set_xlabel("No Adapt AP50 (\\%)")
    ax.set_ylabel("FAN AP50 (\\%)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, frameon=False, ncol=1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_PDF}")

if __name__ == "__main__":
    main()
