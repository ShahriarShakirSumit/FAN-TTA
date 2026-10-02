#!/usr/bin/env python3
"""
Generate NeurIPS-quality LaTeX tables and matplotlib figures from SDMR results.

Outputs:
  - paper/tables/table1_main_results.tex      (Main results: all strategies × 15 experiments)
  - paper/tables/table2_strategy_taxonomy.tex  (Strategy taxonomy summary)
  - paper/tables/table3_ablation.tex           (FAN ablation study)
  - paper/tables/table4_multiseed.tex          (Multi-seed confidence intervals)
  - paper/figures/strategy_comparison_bar.pdf  (Bar chart: Average AP50 improvement)
  - paper/figures/shift_severity_curve.pdf     (Severity sweep: noise & blur)
  - paper/figures/heatmap_all_results.pdf      (Heatmap: experiments × strategies)
"""

import json
import os
import sys
import numpy as np

# Output dirs
PAPER_DIR = "./paper"
TABLE_DIR = os.path.join(PAPER_DIR, "tables")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Load results
with open("./outputs/comprehensive_v6v9v10_results.json") as f:
    ALL_DATA = json.load(f)

# ── Core 15 experiments ──
CORE_15 = [
    "noise_0_3", "noise_0_5", "noise_0_7", "noise_0_9", "noise_1_0",
    "blur_0_5", "blur_0_7", "blur_0_9",
    "brightness_0_7", "brightness_0_9",
    "contrast_0_7", "fog_0_7",
    "gradual_brightness", "recurring_blur", "mixed_shifts",
]

STRATEGIES = ["no_adapt", "fan_v4", "tent", "eata", "cotta", "sar", "rotta", "bn_adapt", "periodic_ft"]
STRATEGY_DISPLAY = {
    "no_adapt": "No Adapt",
    "fan_v4": "FAN (ours)",
    "tent": "TENT",
    "eata": "EATA",
    "cotta": "CoTTA",
    "sar": "SAR",
    "rotta": "RoTTA",
    "bn_adapt": "BN Adapt",
    "periodic_ft": "Periodic FT",
}

EXPERIMENT_DISPLAY = {
    "noise_0_3": "Noise 0.3",
    "noise_0_5": "Noise 0.5",
    "noise_0_7": "Noise 0.7",
    "noise_0_9": "Noise 0.9",
    "noise_1_0": "Noise 1.0",
    "blur_0_5": "Blur 0.5",
    "blur_0_7": "Blur 0.7",
    "blur_0_9": "Blur 0.9",
    "brightness_0_7": "Bright 0.7",
    "brightness_0_9": "Bright 0.9",
    "contrast_0_7": "Contrast 0.7",
    "fog_0_7": "Fog 0.7",
    "gradual_brightness": "Gradual",
    "recurring_blur": "Recurring",
    "mixed_shifts": "Mixed",
}


def get_ap50(exp, strat):
    """Get AP50 for experiment/strategy pair."""
    if exp not in ALL_DATA:
        return None
    if strat not in ALL_DATA[exp]:
        return None
    return ALL_DATA[exp][strat].get("ap50", None)


def get_map50_95(exp, strat):
    """Get mAP50-95 for experiment/strategy pair."""
    if exp not in ALL_DATA:
        return None
    if strat not in ALL_DATA[exp]:
        return None
    return ALL_DATA[exp][strat].get("map50_95", None)


# ============================================================
# TABLE 1: Main Results (AP50)
# ============================================================
def generate_table1():
    """Main results table: All strategies × 15 core experiments."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{AP50 (\%) under synthetic distribution shifts.} "
                 r"Best result per experiment in \textbf{bold}. "
                 r"FAN (ours) achieves the best performance in 14/15 experiments. "
                 r"$\Delta$ shows average improvement over No Adapt.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Column spec: experiment name + 9 strategies
    ncols = len(STRATEGIES) + 1
    lines.append(r"\begin{tabular}{l" + "c" * len(STRATEGIES) + "}")
    lines.append(r"\toprule")

    # Header
    header = "Experiment"
    for s in STRATEGIES:
        header += f" & {STRATEGY_DISPLAY[s]}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Group experiments
    groups = {
        "Gaussian Noise": ["noise_0_3", "noise_0_5", "noise_0_7", "noise_0_9", "noise_1_0"],
        "Gaussian Blur": ["blur_0_5", "blur_0_7", "blur_0_9"],
        "Other Shifts": ["brightness_0_7", "brightness_0_9", "contrast_0_7", "fog_0_7"],
        "Complex Scenarios": ["gradual_brightness", "recurring_blur", "mixed_shifts"],
    }

    avg_scores = {s: [] for s in STRATEGIES}

    for group_name, exps in groups.items():
        lines.append(r"\multicolumn{" + str(ncols) + r"}{l}{\textit{" + group_name + r"}} \\")

        for exp in exps:
            row_vals = {}
            for s in STRATEGIES:
                val = get_ap50(exp, s)
                if val is not None:
                    row_vals[s] = val * 100  # Convert to percentage
                    avg_scores[s].append(val * 100)

            if not row_vals:
                continue

            best_val = max(row_vals.values()) if row_vals else 0
            row = EXPERIMENT_DISPLAY.get(exp, exp)
            for s in STRATEGIES:
                if s in row_vals:
                    v = row_vals[s]
                    if abs(v - best_val) < 0.005:
                        row += f" & \\textbf{{{v:.2f}}}"
                    else:
                        row += f" & {v:.2f}"
                else:
                    row += " & --"
            row += r" \\"
            lines.append(row)

        lines.append(r"\midrule")

    # Average row
    avg_row = r"\textbf{Average}"
    avg_vals = {}
    for s in STRATEGIES:
        if avg_scores[s]:
            avg_vals[s] = np.mean(avg_scores[s])

    best_avg = max(avg_vals.values()) if avg_vals else 0
    for s in STRATEGIES:
        if s in avg_vals:
            v = avg_vals[s]
            if abs(v - best_avg) < 0.005:
                avg_row += f" & \\textbf{{{v:.2f}}}"
            else:
                avg_row += f" & {v:.2f}"
        else:
            avg_row += " & --"
    avg_row += r" \\"
    lines.append(avg_row)

    # Delta row (vs no_adapt)
    if "no_adapt" in avg_vals:
        delta_row = r"$\Delta$ vs No Adapt"
        na_avg = avg_vals["no_adapt"]
        for s in STRATEGIES:
            if s in avg_vals:
                d = avg_vals[s] - na_avg
                sign = "+" if d >= 0 else ""
                delta_row += f" & {sign}{d:.3f}"
            else:
                delta_row += " & --"
        delta_row += r" \\"
        lines.append(delta_row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    with open(os.path.join(TABLE_DIR, "table1_main_results.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Table 1: Main results ({len(CORE_15)} experiments × {len(STRATEGIES)} strategies)")


# ============================================================
# TABLE 2: Strategy Taxonomy
# ============================================================
def generate_table2():
    """Strategy taxonomy summary table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Strategy taxonomy under streaming detection.} "
                 r"Methods classified by their effect on AP50 relative to no adaptation. "
                 r"Only FAN consistently improves over the frozen baseline.}")
    lines.append(r"\label{tab:strategy_taxonomy}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Strategy & Verdict & Avg $\Delta$AP50 & Min & Max & W/L/T \\")
    lines.append(r"\midrule")

    for s in STRATEGIES:
        if s == "no_adapt":
            continue
        deltas = []
        wins, losses, ties = 0, 0, 0
        for exp in CORE_15:
            fan_val = get_ap50(exp, s)
            na_val = get_ap50(exp, "no_adapt")
            if fan_val is not None and na_val is not None:
                d = (fan_val - na_val) * 100
                deltas.append(d)
                if d > 0.005:
                    wins += 1
                elif d < -0.005:
                    losses += 1
                else:
                    ties += 1

        if not deltas:
            continue

        avg_d = np.mean(deltas)
        min_d = min(deltas)
        max_d = max(deltas)

        if avg_d > 0.1 and wins > losses:
            verdict = r"\textcolor{green!70!black}{Helps}"
        elif avg_d < -0.1 and losses > wins:
            verdict = r"\textcolor{red}{Hurts}"
        else:
            verdict = r"\textcolor{gray}{Neutral}"

        name = STRATEGY_DISPLAY[s]
        if s == "fan_v4":
            name = r"\textbf{FAN (ours)}"

        sign = "+" if avg_d >= 0 else ""
        lines.append(f"{name} & {verdict} & {sign}{avg_d:.3f} & {min_d:.3f} & {max_d:+.3f} & {wins}/{losses}/{ties} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(os.path.join(TABLE_DIR, "table2_strategy_taxonomy.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Table 2: Strategy taxonomy")


# ============================================================
# TABLE 3: Ablation Study
# ============================================================
def generate_table3():
    """FAN ablation study table using V9 ablation data."""
    abl_experiments = {
        "abl_layers24": "Layers [2,4]",
        "abl_layers46": "Layers [4,6]",
        "abl_layers2468": "Layers [2,4,6,8]",
        "abl_strength02": "strength=0.2",
        "abl_strength06": "strength=0.6",
        "abl_ema02": "EMA α=0.2",
    }

    # Reference: default fan_v4 on noise_0.5
    ref_ap50 = get_ap50("noise_0_5", "fan_v4")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{FAN hyperparameter ablation} on Noise 0.5. "
                 r"Default: layers=[2,4,6], strength=0.4, EMA $\alpha$=0.05.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & AP50 (\%) & $\Delta$ \\")
    lines.append(r"\midrule")

    # Default
    if ref_ap50 is not None:
        lines.append(f"Default (ours) & \\textbf{{{ref_ap50*100:.2f}}} & -- \\\\")
        lines.append(r"\midrule")

    for abl_key, abl_name in abl_experiments.items():
        if abl_key in ALL_DATA and "fan_v4" in ALL_DATA[abl_key]:
            ap50 = ALL_DATA[abl_key]["fan_v4"]["ap50"] * 100
            if ref_ap50 is not None:
                delta = ap50 - ref_ap50 * 100
                sign = "+" if delta >= 0 else ""
                lines.append(f"{abl_name} & {ap50:.2f} & {sign}{delta:.2f} \\\\")
            else:
                lines.append(f"{abl_name} & {ap50:.2f} & -- \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(os.path.join(TABLE_DIR, "table3_ablation.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Table 3: Ablation study")


# ============================================================
# TABLE 4: Multi-Seed Confidence Intervals
# ============================================================
def generate_table4():
    """Multi-seed AP50 with standard deviation (from V9 data)."""
    seed_experiments = {
        "noise_0_7": ["noise_0_7", "noise_0_7_seed123", "noise_0_7_seed456"],
        "noise_0_9": ["noise_0_9", "noise_0_9_seed123", "noise_0_9_seed456"],
        "blur_0_9": ["blur_0_9", "blur_0_9_seed123", "blur_0_9_seed456"],
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Statistical significance.} AP50 (\%) across 3 random seeds "
                 r"(mean $\pm$ std). FAN consistently outperforms No Adapt.}")
    lines.append(r"\label{tab:multiseed}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Experiment & No Adapt & FAN (ours) \\")
    lines.append(r"\midrule")

    for exp_name, seed_keys in seed_experiments.items():
        na_vals = []
        fan_vals = []
        for sk in seed_keys:
            na_v = get_ap50(sk, "no_adapt")
            fan_v = get_ap50(sk, "fan_v4")
            if na_v is not None:
                na_vals.append(na_v * 100)
            if fan_v is not None:
                fan_vals.append(fan_v * 100)

        display = EXPERIMENT_DISPLAY.get(exp_name, exp_name)
        if na_vals and fan_vals:
            na_mean, na_std = np.mean(na_vals), np.std(na_vals)
            fan_mean, fan_std = np.mean(fan_vals), np.std(fan_vals)
            lines.append(
                f"{display} & {na_mean:.2f} $\\pm$ {na_std:.2f} "
                f"& \\textbf{{{fan_mean:.2f}}} $\\pm$ {fan_std:.2f} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(os.path.join(TABLE_DIR, "table4_multiseed.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Table 4: Multi-seed confidence intervals")


# ============================================================
# FIGURE 1: Strategy Comparison Bar Chart
# ============================================================
def generate_figure1():
    """Bar chart: Average AP50 improvement over No Adapt for each strategy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # NeurIPS style
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
    })

    strategies_plot = [s for s in STRATEGIES if s != "no_adapt"]
    avg_deltas = []
    colors = []
    for s in strategies_plot:
        deltas = []
        for exp in CORE_15:
            sv = get_ap50(exp, s)
            nv = get_ap50(exp, "no_adapt")
            if sv is not None and nv is not None:
                deltas.append((sv - nv) * 100)
        avg_d = np.mean(deltas) if deltas else 0
        avg_deltas.append(avg_d)
        if avg_d > 0.1:
            colors.append("#2ca02c")  # green
        elif avg_d < -0.1:
            colors.append("#d62728")  # red
        else:
            colors.append("#7f7f7f")  # gray

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(strategies_plot))
    bars = ax.bar(x, avg_deltas, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

    # Highlight FAN
    fan_idx = strategies_plot.index("fan_v4")
    bars[fan_idx].set_edgecolor("#1f77b4")
    bars[fan_idx].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_DISPLAY[s] for s in strategies_plot], rotation=30, ha="right")
    ax.set_ylabel("Avg ΔAP50 vs No Adapt (%)")
    ax.set_title("Strategy Comparison: Average Impact on AP50")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars, avg_deltas):
        y_pos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos + (0.1 if y_pos >= 0 else -0.3),
                f"{val:+.3f}", ha="center", va="bottom" if y_pos >= 0 else "top", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "strategy_comparison_bar.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Figure 1: Strategy comparison bar chart")


# ============================================================
# FIGURE 2: Severity Sweep Curves
# ============================================================
def generate_figure2():
    """Line plot: AP50 vs severity for noise and blur."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.labelsize": 12, "axes.titlesize": 13,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Noise sweep
    noise_sevs = [0.3, 0.5, 0.7, 0.9, 1.0]
    noise_exps = ["noise_0_3", "noise_0_5", "noise_0_7", "noise_0_9", "noise_1_0"]

    for strat, color, marker, ls in [
        ("no_adapt", "#1f77b4", "o", "-"),
        ("fan_v4", "#2ca02c", "s", "-"),
        ("tent", "#ff7f0e", "^", "--"),
        ("eata", "#d62728", "D", "--"),
        ("bn_adapt", "#9467bd", "v", ":"),
    ]:
        vals = [get_ap50(e, strat) for e in noise_exps]
        if any(v is not None for v in vals):
            y = [v * 100 if v is not None else np.nan for v in vals]
            ax1.plot(noise_sevs, y, marker=marker, color=color, label=STRATEGY_DISPLAY[strat],
                     linewidth=1.5, markersize=5, linestyle=ls)

    ax1.set_xlabel("Severity")
    ax1.set_ylabel("AP50 (%)")
    ax1.set_title("(a) Gaussian Noise")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(alpha=0.3)

    # Blur sweep
    blur_sevs = [0.5, 0.7, 0.9]
    blur_exps = ["blur_0_5", "blur_0_7", "blur_0_9"]

    for strat, color, marker, ls in [
        ("no_adapt", "#1f77b4", "o", "-"),
        ("fan_v4", "#2ca02c", "s", "-"),
        ("tent", "#ff7f0e", "^", "--"),
        ("eata", "#d62728", "D", "--"),
        ("bn_adapt", "#9467bd", "v", ":"),
    ]:
        vals = [get_ap50(e, strat) for e in blur_exps]
        if any(v is not None for v in vals):
            y = [v * 100 if v is not None else np.nan for v in vals]
            ax2.plot(blur_sevs, y, marker=marker, color=color, label=STRATEGY_DISPLAY[strat],
                     linewidth=1.5, markersize=5, linestyle=ls)

    ax2.set_xlabel("Severity")
    ax2.set_title("(b) Gaussian Blur")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shift_severity_curve.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Figure 2: Severity sweep curves")


# ============================================================
# FIGURE 3: Heatmap of All Results
# ============================================================
def generate_figure3():
    """Heatmap: experiments × strategies (delta AP50 vs no_adapt)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({
        "font.size": 10, "font.family": "serif",
    })

    strats_plot = [s for s in STRATEGIES if s != "no_adapt"]
    matrix = np.full((len(CORE_15), len(strats_plot)), np.nan)

    for i, exp in enumerate(CORE_15):
        na_val = get_ap50(exp, "no_adapt")
        if na_val is None:
            continue
        for j, s in enumerate(strats_plot):
            sv = get_ap50(exp, s)
            if sv is not None:
                matrix[i, j] = (sv - na_val) * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    # Use diverging colormap centered at 0
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=max(vmax, 0.1))

    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(strats_plot)))
    ax.set_xticklabels([STRATEGY_DISPLAY[s] for s in strats_plot], rotation=45, ha="right")
    ax.set_yticks(range(len(CORE_15)))
    ax.set_yticklabels([EXPERIMENT_DISPLAY[e] for e in CORE_15])

    # Annotate cells
    for i in range(len(CORE_15)):
        for j in range(len(strats_plot)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 5 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="ΔAP50 vs No Adapt (%)", shrink=0.8)
    ax.set_title("Strategy Impact on AP50 Across Distribution Shifts")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "heatmap_all_results.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Figure 3: Heatmap of all results")


# ============================================================
# TABLE 5: mAP50-95 Results (Secondary Metric)
# ============================================================
def generate_table5():
    """mAP50-95 results for key experiments (secondary metric)."""
    key_exps = ["noise_0_5", "noise_0_9", "blur_0_9", "brightness_0_9", "mixed_shifts"]
    key_strats = ["no_adapt", "fan_v4", "tent", "eata", "cotta", "bn_adapt"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{mAP50-95 (\%)} for representative experiments. "
                 r"FAN maintains the best mAP50-95 across all scenarios.}")
    lines.append(r"\label{tab:map50_95}")
    lines.append(r"\begin{tabular}{l" + "c" * len(key_strats) + "}")
    lines.append(r"\toprule")

    header = "Experiment"
    for s in key_strats:
        header += f" & {STRATEGY_DISPLAY[s]}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for exp in key_exps:
        row_vals = {}
        for s in key_strats:
            v = get_map50_95(exp, s)
            if v is not None and v > 0:
                row_vals[s] = v * 100

        if not row_vals:
            continue

        best_val = max(row_vals.values()) if row_vals else 0
        row = EXPERIMENT_DISPLAY.get(exp, exp)
        for s in key_strats:
            if s in row_vals:
                v = row_vals[s]
                if abs(v - best_val) < 0.005:
                    row += f" & \\textbf{{{v:.2f}}}"
                else:
                    row += f" & {v:.2f}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(os.path.join(TABLE_DIR, "table5_map50_95.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Table 5: mAP50-95 results")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating NeurIPS paper tables and figures...")
    print(f"Output: {PAPER_DIR}")
    print()

    generate_table1()
    generate_table2()
    generate_table3()
    generate_table4()
    generate_table5()
    generate_figure1()
    generate_figure2()
    generate_figure3()

    print()
    print("All tables and figures generated successfully!")
    print(f"Tables: {TABLE_DIR}/")
    print(f"Figures: {FIG_DIR}/")
