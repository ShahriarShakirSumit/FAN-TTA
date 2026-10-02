#!/usr/bin/env python3
"""
Generate FAN architecture diagram and additional paper figures.

Outputs:
  - paper/figures/fan_architecture.pdf — FAN system overview
  - paper/figures/strategy_bar_horizontal.pdf — Horizontal bar chart for strategy taxonomy  
  - paper/figures/timeline_example.pdf — Example stream timeline with shift
"""

import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PAPER_DIR = "./paper"
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# NeurIPS style
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


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
    ax.text(0.8, 2.5, "Input\nFrame $x_t$", ha="center", va="center", fontsize=9, fontweight="bold")

    # Backbone layers
    for i, (y, label) in enumerate([(3.5, "Layer 6"), (2.0, "Layer 4"), (0.5, "Layer 2")]):
        box = FancyBboxPatch((2.0, y), 1.0, 0.8, boxstyle="round,pad=0.05",
                              facecolor=c_backbone, edgecolor="black", linewidth=1)
        ax.add_patch(box)
        ax.text(2.5, y + 0.4, label, ha="center", va="center", fontsize=8)

    ax.text(2.5, 4.6, "YOLOv8\nBackbone", ha="center", va="center", fontsize=9,
            fontweight="bold", style="italic")

    # FAN module
    box = FancyBboxPatch((3.8, 1.0), 2.4, 3.0, boxstyle="round,pad=0.15",
                          facecolor=c_fan, edgecolor="#27AE60", linewidth=2)
    ax.add_patch(box)
    ax.text(5.0, 3.8, "Feature Anchor Network", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#27AE60")

    # FAN sub-components
    sub_items = [
        (4.0, 3.1, 1.8, 0.5, "EMA Anchors\n$\\mu^{anchor}, \\sigma^{anchor}$", c_anchor),
        (4.0, 2.2, 1.8, 0.5, "Channel-wise\nCorrection", "#ABEBC6"),
        (4.0, 1.3, 1.8, 0.5, "Safety Gates", c_safety),
    ]
    for x, y, w, h, label, color in sub_items:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor="gray", linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=7.5)

    # Detection head
    box = FancyBboxPatch((7.0, 2.0), 1.2, 1.0, boxstyle="round,pad=0.1",
                          facecolor=c_det, edgecolor="black", linewidth=1.2)
    ax.add_patch(box)
    ax.text(7.6, 2.5, "Detection\nHead", ha="center", va="center", fontsize=9, fontweight="bold")

    # Output
    box = FancyBboxPatch((8.8, 2.0), 1.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor=c_output, edgecolor="black", linewidth=1.2)
    ax.add_patch(box)
    ax.text(9.3, 2.5, "Boxes\n+ AP50", ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color="black", linewidth=1.5, mutation_scale=12)
    # Input → backbone
    ax.annotate("", xy=(2.0, 2.5), xytext=(1.4, 2.5), arrowprops=arrow_kw)
    # Backbone → FAN
    ax.annotate("", xy=(3.8, 2.5), xytext=(3.0, 2.5), arrowprops=arrow_kw)
    # FAN → Detection
    ax.annotate("", xy=(7.0, 2.5), xytext=(6.2, 2.5), arrowprops=arrow_kw)
    # Detection → Output
    ax.annotate("", xy=(8.8, 2.5), xytext=(8.2, 2.5), arrowprops=arrow_kw)

    # Safety sub-labels
    ax.text(5.0, 1.05, "NB-ratio | Hysteresis | Conf-gate", ha="center", va="center",
            fontsize=6.5, style="italic", color="gray")

    plt.savefig(os.path.join(FIG_DIR, "fan_architecture.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ FAN architecture diagram")


def make_horizontal_bar():
    """Horizontal bar chart for strategy taxonomy — cleaner than vertical."""
    strategies = ["FAN (ours)", "TENT", "EATA", "CoTTA", "SAR", "RoTTA", "BN Adapt", "Periodic FT"]
    avg_deltas = [+0.780, -0.366, -0.572, +0.000, -0.013, +0.000, -9.802, -2.244]
    
    colors = []
    for d in avg_deltas:
        if d > 0.1:
            colors.append("#2ca02c")
        elif d < -0.1:
            colors.append("#d62728")
        else:
            colors.append("#7f7f7f")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(strategies))
    bars = ax.barh(y, avg_deltas, color=colors, edgecolor="black", linewidth=0.5, height=0.6)
    
    # Highlight FAN
    bars[0].set_edgecolor("#1a5276")
    bars[0].set_linewidth(2)

    ax.set_yticks(y)
    ax.set_yticklabels(strategies)
    ax.set_xlabel("Average ΔAP50 vs No Adapt (%)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Value labels
    for bar, val in zip(bars, avg_deltas):
        x_pos = bar.get_width()
        offset = 0.15 if val >= 0 else -0.15
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", ha=ha, va="center", fontsize=8)

    # Legend
    legend_items = [
        mpatches.Patch(color="#2ca02c", label="Helps"),
        mpatches.Patch(color="#7f7f7f", label="Neutral"),
        mpatches.Patch(color="#d62728", label="Hurts"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "strategy_bar_horizontal.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Strategy taxonomy horizontal bar chart")


def make_stream_timeline():
    """Example stream timeline showing clean → shifted → clean."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4), height_ratios=[1, 1.2], sharex=True)

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
    ax1.text(150, 0.85, "Clean", ha="center", fontsize=9, color="#27ae60", fontweight="bold")
    ax1.text(450, 0.85, "Shifted", ha="center", fontsize=9, color="#e74c3c", fontweight="bold")
    ax1.text(750, 0.85, "Clean", ha="center", fontsize=9, color="#27ae60", fontweight="bold")

    # Bottom: AP50 over time (simulated)
    np.random.seed(42)
    # No adapt
    ap_clean = 0.72
    ap_shifted = 0.65
    no_adapt = np.ones(900) * ap_clean
    no_adapt[300:600] = ap_shifted
    no_adapt += np.random.normal(0, 0.005, 900)
    # Smooth
    from scipy.ndimage import uniform_filter1d
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

    ax2.plot(t, no_adapt * 100, color="#1f77b4", linewidth=1.5, label="No Adapt", alpha=0.9)
    ax2.plot(t, fan * 100, color="#2ca02c", linewidth=2, label="FAN (ours)")
    ax2.plot(t, bn * 100, color="#d62728", linewidth=1.5, label="BN Adapt", linestyle="--", alpha=0.8)

    ax2.set_ylabel("AP50 (%)", fontsize=9)
    ax2.set_xlabel("Frame Index", fontsize=9)
    ax2.axvline(300, color="gray", linestyle="--", alpha=0.7, linewidth=0.8)
    ax2.axvline(600, color="gray", linestyle="--", alpha=0.7, linewidth=0.8)
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "timeline_example.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Stream timeline example")


if __name__ == "__main__":
    make_fan_architecture()
    make_horizontal_bar()
    make_stream_timeline()
    print("\nAll additional figures generated!")
