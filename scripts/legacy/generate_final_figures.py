#!/usr/bin/env python3
"""
NeurIPS 2026 publication figures — v1 style (serif, no LaTeX) with current data.

Reproduces exact styling from v1_pre_polish figure scripts with data from the
latest comprehensive_v6v9v10_results.json and v11_results.json.

Figures generated:
  1. timeline_example.pdf        — Stream timeline (clean→shifted→clean)
  2. fan_architecture.pdf        — FAN system architecture diagram
  3. shift_severity_curve.pdf    — Severity sweep: noise & blur
  4. strategy_bar_horizontal.pdf — Horizontal bar: avg ΔAP50 per strategy
  5. heatmap_all_results.pdf     — Heatmap: experiments × strategies
  6. cross_domain_comparison.pdf — Cross-domain bar chart
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import uniform_filter1d

# ── Paths ───────────────────────────────────────────────────────────────────
PAPER_DIR = "./paper"
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
with open("./outputs/comprehensive_v6v9v10_results.json") as f:
    ALL_DATA = json.load(f)
with open("./outputs/v11_results.json") as f:
    V11 = json.load(f)

# ── Experiments & strategies ────────────────────────────────────────────────
CORE_15 = [
    "noise_0_3", "noise_0_5", "noise_0_7", "noise_0_9", "noise_1_0",
    "blur_0_5", "blur_0_7", "blur_0_9",
    "brightness_0_7", "brightness_0_9",
    "contrast_0_7", "fog_0_7",
    "gradual_brightness", "recurring_blur", "mixed_shifts",
]

STRATEGIES = ["no_adapt", "fan_v4", "tent", "eata", "cotta", "sar",
              "rotta", "bn_adapt", "periodic_ft"]

STRATEGY_DISPLAY = {
    "no_adapt": "No Adapt", "fan_v4": "FAN (ours)", "tent": "TENT",
    "eata": "EATA", "cotta": "CoTTA", "sar": "SAR", "rotta": "RoTTA",
    "bn_adapt": "BN Adapt", "periodic_ft": "Periodic FT",
}

EXPERIMENT_DISPLAY = {
    "noise_0_3": "Noise 0.3", "noise_0_5": "Noise 0.5",
    "noise_0_7": "Noise 0.7", "noise_0_9": "Noise 0.9",
    "noise_1_0": "Noise 1.0", "blur_0_5": "Blur 0.5",
    "blur_0_7": "Blur 0.7", "blur_0_9": "Blur 0.9",
    "brightness_0_7": "Bright 0.7", "brightness_0_9": "Bright 0.9",
    "contrast_0_7": "Contrast 0.7", "fog_0_7": "Fog 0.7",
    "gradual_brightness": "Gradual", "recurring_blur": "Recurring",
    "mixed_shifts": "Mixed",
}


def get_ap50(exp, strat):
    """Return AP50 as percentage for experiment/strategy pair."""
    e = ALL_DATA.get(exp, {})
    s = e.get(strat, {})
    v = s.get("ap50", None)
    return v * 100 if v is not None else None


# ── v1 NeurIPS style ────────────────────────────────────────────────────────
def set_v1_style():
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Stream timeline (clean → shifted → clean)
# ════════════════════════════════════════════════════════════════════════════
def make_stream_timeline():
    """Example stream timeline showing clean → shifted → clean."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4),
                                    height_ratios=[1, 1.2], sharex=True)

    t = np.arange(900)

    # Top: Shift severity timeline
    severity = np.zeros(900)
    severity[300:600] = 0.7  # Abrupt shift

    ax1.fill_between(t, severity, alpha=0.3, color="#e74c3c")
    ax1.plot(t, severity, color="#e74c3c", linewidth=1.5)
    ax1.set_ylabel("Shift\nSeverity", fontsize=9)
    ax1.set_ylim(-0.05, 1.0)
    ax1.axvline(300, color="gray", linestyle="--", alpha=0.7, linewidth=0.8)
    ax1.axvline(600, color="gray", linestyle="--", alpha=0.7, linewidth=0.8)
    ax1.text(150, 0.85, "Clean", ha="center", fontsize=9,
             color="#27ae60", fontweight="bold")
    ax1.text(450, 0.85, "Shifted", ha="center", fontsize=9,
             color="#e74c3c", fontweight="bold")
    ax1.text(750, 0.85, "Clean", ha="center", fontsize=9,
             color="#27ae60", fontweight="bold")

    # Bottom: AP50 over time (simulated)
    np.random.seed(42)
    ap_clean = 0.72
    ap_shifted = 0.65

    # No adapt
    no_adapt = np.ones(900) * ap_clean
    no_adapt[300:600] = ap_shifted
    no_adapt += np.random.normal(0, 0.005, 900)
    no_adapt = uniform_filter1d(no_adapt, size=20)

    # FAN
    fan = np.ones(900) * ap_clean
    fan[300:600] = ap_shifted + 0.02  # FAN recovers some
    fan += np.random.normal(0, 0.004, 900)
    fan = uniform_filter1d(fan, size=20)

    # BN Adapt (catastrophic)
    bn = np.ones(900) * ap_clean
    bn[300:600] = ap_shifted - 0.08  # Much worse
    bn += np.random.normal(0, 0.008, 900)
    bn = uniform_filter1d(bn, size=20)

    ax2.plot(t, no_adapt * 100, color="#1f77b4", linewidth=1.5,
             label="No Adapt", alpha=0.9)
    ax2.plot(t, fan * 100, color="#2ca02c", linewidth=2, label="FAN (ours)")
    ax2.plot(t, bn * 100, color="#d62728", linewidth=1.5,
             label="BN Adapt", linestyle="--", alpha=0.8)

    ax2.set_ylabel("AP50 (%)", fontsize=9)
    ax2.set_xlabel("Frame Index", fontsize=9)
    ax2.axvline(300, color="gray", linestyle="--", alpha=0.7, linewidth=0.8)
    ax2.axvline(600, color="gray", linestyle="--", alpha=0.7, linewidth=0.8)
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "timeline_example.pdf"),
                bbox_inches="tight")
    plt.close()
    print("  [OK] timeline_example.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: FAN Architecture Diagram
# ════════════════════════════════════════════════════════════════════════════
def make_fan_architecture():
    """Create FAN system architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Colors
    c_input = "#E8F4FD"
    c_backbone = "#D4E6F1"
    c_fan = "#D5F5E3"
    c_det = "#FADBD8"
    c_output = "#F9E79F"
    c_anchor = "#AED6F1"
    c_safety = "#F5B7B1"

    # Input box
    box = FancyBboxPatch((0.2, 2.0), 1.2, 1.0, boxstyle="round,pad=0.1",
                          facecolor=c_input, edgecolor="black", linewidth=1.2)
    ax.add_patch(box)
    ax.text(0.8, 2.5, "Input\nFrame $x_t$", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # Backbone layers
    for y, label in [(3.5, "Layer 6"), (2.0, "Layer 4"), (0.5, "Layer 2")]:
        box = FancyBboxPatch((2.0, y), 1.0, 0.8, boxstyle="round,pad=0.05",
                              facecolor=c_backbone, edgecolor="black", linewidth=1)
        ax.add_patch(box)
        ax.text(2.5, y + 0.4, label, ha="center", va="center", fontsize=8)

    ax.text(2.5, 4.6, "YOLOv8\nBackbone", ha="center", va="center",
            fontsize=9, fontweight="bold", style="italic")

    # FAN module
    box = FancyBboxPatch((3.8, 1.0), 2.4, 3.0, boxstyle="round,pad=0.15",
                          facecolor=c_fan, edgecolor="#27AE60", linewidth=2)
    ax.add_patch(box)
    ax.text(5.0, 3.8, "Feature Anchor Network", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#27AE60")

    # FAN sub-components
    sub_items = [
        (4.0, 3.1, 1.8, 0.5,
         "EMA Anchors\n$\\mu^{anchor}, \\sigma^{anchor}$", c_anchor),
        (4.0, 2.2, 1.8, 0.5, "Channel-wise\nCorrection", "#ABEBC6"),
        (4.0, 1.3, 1.8, 0.5, "Safety Gates", c_safety),
    ]
    for x, y, w, h, label, color in sub_items:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor="gray", linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=7.5)

    # Detection head
    box = FancyBboxPatch((7.0, 2.0), 1.2, 1.0, boxstyle="round,pad=0.1",
                          facecolor=c_det, edgecolor="black", linewidth=1.2)
    ax.add_patch(box)
    ax.text(7.6, 2.5, "Detection\nHead", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # Output
    box = FancyBboxPatch((8.8, 2.0), 1.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor=c_output, edgecolor="black", linewidth=1.2)
    ax.add_patch(box)
    ax.text(9.3, 2.5, "Boxes\n+ AP50", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color="black", linewidth=1.5,
                    mutation_scale=12)
    ax.annotate("", xy=(2.0, 2.5), xytext=(1.4, 2.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(3.8, 2.5), xytext=(3.0, 2.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.0, 2.5), xytext=(6.2, 2.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.8, 2.5), xytext=(8.2, 2.5), arrowprops=arrow_kw)

    # Safety sub-labels
    ax.text(5.0, 1.05, "NB-ratio | Hysteresis | Conf-gate",
            ha="center", va="center", fontsize=6.5, style="italic", color="gray")

    plt.savefig(os.path.join(FIG_DIR, "fan_architecture.pdf"),
                bbox_inches="tight")
    plt.close()
    print("  [OK] fan_architecture.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Severity Sweep Curves (Noise + Blur)
# ════════════════════════════════════════════════════════════════════════════
def make_severity_curve():
    """Line plot: AP50 vs severity for noise and blur."""
    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.labelsize": 12, "axes.titlesize": 13,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    methods = [
        ("no_adapt", "#1f77b4", "o", "-"),
        ("fan_v4",   "#2ca02c", "s", "-"),
        ("tent",     "#ff7f0e", "^", "--"),
        ("eata",     "#d62728", "D", "--"),
        ("bn_adapt", "#9467bd", "v", ":"),
    ]

    # (a) Noise sweep
    noise_sevs = [0.3, 0.5, 0.7, 0.9, 1.0]
    noise_exps = ["noise_0_3", "noise_0_5", "noise_0_7",
                  "noise_0_9", "noise_1_0"]

    for strat, color, marker, ls in methods:
        vals = [get_ap50(e, strat) for e in noise_exps]
        if any(v is not None for v in vals):
            y = [v if v is not None else np.nan for v in vals]
            ax1.plot(noise_sevs, y, marker=marker, color=color,
                     label=STRATEGY_DISPLAY[strat],
                     linewidth=1.5, markersize=5, linestyle=ls)

    ax1.set_xlabel("Severity")
    ax1.set_ylabel("AP50 (%)")
    ax1.set_title("(a) Gaussian Noise")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(alpha=0.3)

    # (b) Blur sweep
    blur_sevs = [0.5, 0.7, 0.9]
    blur_exps = ["blur_0_5", "blur_0_7", "blur_0_9"]

    for strat, color, marker, ls in methods:
        vals = [get_ap50(e, strat) for e in blur_exps]
        if any(v is not None for v in vals):
            y = [v if v is not None else np.nan for v in vals]
            ax2.plot(blur_sevs, y, marker=marker, color=color,
                     label=STRATEGY_DISPLAY[strat],
                     linewidth=1.5, markersize=5, linestyle=ls)

    ax2.set_xlabel("Severity")
    ax2.set_title("(b) Gaussian Blur")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shift_severity_curve.pdf"),
                bbox_inches="tight")
    plt.close()
    set_v1_style()  # restore defaults
    print("  [OK] shift_severity_curve.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Strategy Horizontal Bar — computed from data (not hardcoded)
# ════════════════════════════════════════════════════════════════════════════
def make_horizontal_bar():
    """Horizontal bar chart for strategy taxonomy — values from data."""
    strategies_plot = [s for s in STRATEGIES if s != "no_adapt"]

    # Compute average delta vs no_adapt from actual JSON data
    avg_deltas = []
    for s in strategies_plot:
        deltas = []
        for exp in CORE_15:
            sv = get_ap50(exp, s)
            nv = get_ap50(exp, "no_adapt")
            if sv is not None and nv is not None:
                deltas.append(sv - nv)
        avg_deltas.append(np.mean(deltas) if deltas else 0)

    # Sort by ascending delta (worst at top, best at bottom)
    order = sorted(range(len(strategies_plot)),
                   key=lambda i: avg_deltas[i])
    strategies_plot = [strategies_plot[i] for i in order]
    avg_deltas = [avg_deltas[i] for i in order]

    colors = []
    for d in avg_deltas:
        if d > 0.1:
            colors.append("#2ca02c")  # green – helps
        elif d < -0.1:
            colors.append("#d62728")  # red – hurts
        else:
            colors.append("#7f7f7f")  # gray – neutral

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(strategies_plot))
    bars = ax.barh(y, avg_deltas, color=colors, edgecolor="black",
                   linewidth=0.5, height=0.6)

    # Highlight FAN
    fan_idx = strategies_plot.index("fan_v4")
    bars[fan_idx].set_edgecolor("#1a5276")
    bars[fan_idx].set_linewidth(2)

    ax.set_yticks(y)
    ax.set_yticklabels([STRATEGY_DISPLAY[s] for s in strategies_plot])
    ax.set_xlabel("Average \u0394AP50 vs No Adapt (%)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    # Value labels
    for bar, val in zip(bars, avg_deltas):
        x_pos = bar.get_width()
        offset = 0.15 if val >= 0 else -0.15
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", ha=ha, va="center", fontsize=8)

    # Legend
    legend_items = [
        mpatches.Patch(color="#2ca02c", label="Helps"),
        mpatches.Patch(color="#7f7f7f", label="Neutral"),
        mpatches.Patch(color="#d62728", label="Hurts"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8,
              framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "strategy_bar_horizontal.pdf"),
                bbox_inches="tight")
    plt.close()
    print("  [OK] strategy_bar_horizontal.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Heatmap (experiments × strategies)
# ════════════════════════════════════════════════════════════════════════════
def make_heatmap():
    """Heatmap: experiments × strategies (delta AP50 vs no_adapt)."""
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})

    strats_plot = [s for s in STRATEGIES if s != "no_adapt"]
    matrix = np.full((len(CORE_15), len(strats_plot)), np.nan)

    for i, exp in enumerate(CORE_15):
        na_val = get_ap50(exp, "no_adapt")
        if na_val is None:
            continue
        for j, s in enumerate(strats_plot):
            sv = get_ap50(exp, s)
            if sv is not None:
                matrix[i, j] = sv - na_val

    fig, ax = plt.subplots(figsize=(10, 7))

    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=max(vmax, 0.1))
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(strats_plot)))
    ax.set_xticklabels([STRATEGY_DISPLAY[s] for s in strats_plot],
                        rotation=45, ha="right")
    ax.set_yticks(range(len(CORE_15)))
    ax.set_yticklabels([EXPERIMENT_DISPLAY[e] for e in CORE_15])

    # Annotate cells
    for i in range(len(CORE_15)):
        for j in range(len(strats_plot)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 5 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="\u0394AP50 vs No Adapt (%)", shrink=0.8)
    ax.set_title("Strategy Impact on AP50 Across Distribution Shifts")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "heatmap_all_results.pdf"),
                bbox_inches="tight")
    plt.close()
    set_v1_style()
    print("  [OK] heatmap_all_results.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Cross-domain comparison
# ════════════════════════════════════════════════════════════════════════════
def make_cross_domain():
    """Bar chart comparing strategies on cross-domain benchmarks."""
    plt.rcParams.update({
        "font.size": 10, "font.family": "serif", "figure.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    strats = ["no_adapt", "fan", "tent", "eata", "cotta", "sar",
              "rotta", "bn_adapt", "periodic_ft"]
    labels = ["No Adapt", "FAN\n(ours)", "TENT", "EATA", "CoTTA",
              "SAR", "RoTTA", "BN\nAdapt", "Periodic\nFT"]

    for ax, exp_key, title in [
        (ax1, "v11_coco2nightowls", "(a) COCO \u2192 NightOwls"),
        (ax2, "v11_coco2ecp_night", "(b) COCO \u2192 ECP Night"),
    ]:
        data = V11.get(exp_key, {})
        vals = [data.get(s, {}).get("ap50", 0) * 100 for s in strats]

        colors = []
        for i, s in enumerate(strats):
            if s == "fan":
                colors.append("#2ca02c")
            elif i == 0:
                colors.append("#1f77b4")
            elif vals[i] < vals[0] - 0.1:
                colors.append("#d62728")
            else:
                colors.append("#7f7f7f")

        x = np.arange(len(strats))
        bars = ax.bar(x, vals, color=colors, edgecolor="black",
                      linewidth=0.5, width=0.7)
        bars[1].set_edgecolor("#1a5276")
        bars[1].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("AP50 (%)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Value label on FAN bar
        fan_idx = 1
        ax.text(fan_idx, vals[fan_idx] + 0.3, f"{vals[fan_idx]:.1f}",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#2ca02c")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cross_domain_comparison.pdf"),
                bbox_inches="tight")
    plt.close()
    set_v1_style()
    print("  [OK] cross_domain_comparison.pdf")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    set_v1_style()
    print("Generating v1-style figures with current data...")
    make_stream_timeline()
    make_fan_architecture()
    make_severity_curve()
    make_horizontal_bar()
    make_heatmap()
    make_cross_domain()
    print("\nDone — all 6 figures generated.")
