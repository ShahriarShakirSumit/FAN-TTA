#!/usr/bin/env python3
"""
V7 Results Aggregator — Collects all V7 experiment results into
summary tables for the NeurIPS paper.

Usage:
  python scripts/aggregate_v7_results.py [--output-dir /path/to/outputs]

Produces:
  - Table 1: Main comparison (all strategies, noise 0.5)
  - Table 2: Noise severity sweep
  - Table 3: Cross-corruption robustness
  - Table 4: Temporal shift patterns
  - Table 5: FAN ablation study
  - Table 6: Multi-seed statistical significance
  - LaTeX-formatted tables for paper
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_results(base_dir: str, experiment_name: str, strategy: str) -> dict:
    """Load results.json for a given experiment/strategy."""
    path = Path(base_dir) / experiment_name / strategy / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_frames(base_dir: str, experiment_name: str, strategy: str) -> list:
    """Load frames.jsonl for per-frame analysis."""
    path = Path(base_dir) / experiment_name / strategy / "frames.jsonl"
    if not path.exists():
        return []
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def compute_shifted_metrics(frames: list) -> dict:
    """Compute metrics only for shifted frames."""
    shifted = [f for f in frames if f.get("is_shifted", False)]
    if not shifted:
        return {"ap50_shifted": None, "f1_shifted": None, "n_shifted": 0}
    ap50s = [f["ap50"] for f in shifted]
    f1s = [f.get("f1", 0) for f in shifted]
    return {
        "ap50_shifted": np.mean(ap50s),
        "f1_shifted": np.mean(f1s),
        "n_shifted": len(shifted),
    }


def compute_clean_metrics(frames: list) -> dict:
    """Compute metrics only for clean (non-shifted) frames."""
    clean = [f for f in frames if not f.get("is_shifted", False)]
    if not clean:
        return {"ap50_clean": None, "f1_clean": None, "n_clean": 0}
    ap50s = [f["ap50"] for f in clean]
    f1s = [f.get("f1", 0) for f in clean]
    return {
        "ap50_clean": np.mean(ap50s),
        "f1_clean": np.mean(f1s),
        "n_clean": len(clean),
    }


def fmt(val, precision=4):
    """Format a numeric value or return '-' if None."""
    if val is None:
        return "-"
    return f"{val:.{precision}f}"


def bold_best(values: list, keys: list, higher_better=True):
    """Return the key with the best value."""
    valid = [(v, k) for v, k in zip(values, keys) if v is not None]
    if not valid:
        return None
    if higher_better:
        return max(valid, key=lambda x: x[0])[1]
    return min(valid, key=lambda x: x[0])[1]


def print_table(title: str, headers: list, rows: list, best_cols=None):
    """Pretty-print a table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Calculate column widths
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4))
              for i, h in enumerate(headers)]

    # Header
    header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    # Rows
    for row in rows:
        line = "  ".join(str(r).ljust(w) for r, w in zip(row, widths))
        print(line)

    print()


def generate_latex_table(title: str, headers: list, rows: list,
                         caption: str, label: str, best_col_idx=None):
    """Generate LaTeX table code."""
    ncols = len(headers)
    col_spec = "l" + "c" * (ncols - 1)

    lines = [
        f"% {title}",
        f"\\begin{{table}}[t]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]

    for row in rows:
        cells = []
        for i, val in enumerate(row):
            s = str(val)
            # Bold the best value in specified columns
            if best_col_idx and i in best_col_idx:
                try:
                    float(s)
                    cells.append(s)
                except ValueError:
                    cells.append(s)
            else:
                cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# ============================================================
# TABLE GENERATORS
# ============================================================

def table1_main_comparison(base_dir: str):
    """Table 1: All strategies at noise severity 0.5."""
    strategies = [
        "no_adapt", "fan", "sdmr", "tent", "eata",
        "cotta", "sar", "rotta", "bn_adapt", "periodic_ft"
    ]
    exp_name = "v7_noise_0_5"

    headers = ["Strategy", "AP50↑", "mAP↑", "F1↑", "AP50_shift↑",
               "AP50_clean", "Δ_shift", "Adapt_ms"]
    rows = []

    baseline_ap50_shift = None

    for strategy in strategies:
        res = load_results(base_dir, exp_name, strategy)
        frames = load_frames(base_dir, exp_name, strategy)

        if res is None:
            rows.append([strategy] + ["-"] * 7)
            continue

        sm = compute_shifted_metrics(frames)
        cm = compute_clean_metrics(frames)

        ap50 = res.get("mean_ap50")
        mAP = res.get("mean_map")
        f1 = res.get("mean_f1")
        ap50_shift = sm["ap50_shifted"]
        ap50_clean = cm["ap50_clean"]
        adapt_ms = res.get("mean_adapt_ms", res.get("adapt_ms_mean"))

        if strategy == "no_adapt" and ap50_shift is not None:
            baseline_ap50_shift = ap50_shift

        delta = None
        if baseline_ap50_shift is not None and ap50_shift is not None:
            delta = ap50_shift - baseline_ap50_shift

        rows.append([
            strategy,
            fmt(ap50), fmt(mAP), fmt(f1),
            fmt(ap50_shift), fmt(ap50_clean),
            fmt(delta, 4) if delta is not None else "-",
            fmt(adapt_ms, 1) if adapt_ms else "-",
        ])

    print_table("TABLE 1: Main Comparison — All Strategies (Noise σ=0.5)", headers, rows)
    return headers, rows


def table2_noise_sweep(base_dir: str):
    """Table 2: Noise severity sweep."""
    strategies = ["no_adapt", "fan", "sdmr", "tent", "eata"]
    severities = ["0_3", "0_5", "0_7", "0_9", "1_0"]

    headers = ["Severity"] + strategies
    rows = []

    for sev in severities:
        exp_name = f"v7_noise_{sev}"
        row = [sev.replace("_", ".")]
        for strategy in strategies:
            res = load_results(base_dir, exp_name, strategy)
            frames = load_frames(base_dir, exp_name, strategy)
            sm = compute_shifted_metrics(frames)
            row.append(fmt(sm["ap50_shifted"]))
        rows.append(row)

    print_table("TABLE 2: Noise Severity Sweep — AP50 on Shifted Frames", headers, rows)
    return headers, rows


def table3_cross_corruption(base_dir: str):
    """Table 3: Cross-corruption robustness."""
    strategies = ["no_adapt", "fan", "sdmr", "tent", "eata"]
    corruptions = [
        ("blur_0_5", "Blur 0.5"),
        ("blur_0_7", "Blur 0.7"),
        ("blur_0_9", "Blur 0.9"),
        ("brightness_0_7", "Bright 0.7"),
        ("brightness_0_9", "Bright 0.9"),
        ("fog_0_7", "Fog 0.7"),
        ("contrast_0_7", "Contrast 0.7"),
    ]

    headers = ["Corruption"] + strategies
    rows = []

    for corr_key, corr_label in corruptions:
        exp_name = f"v7_{corr_key}"
        row = [corr_label]
        for strategy in strategies:
            res = load_results(base_dir, exp_name, strategy)
            frames = load_frames(base_dir, exp_name, strategy)
            sm = compute_shifted_metrics(frames)
            row.append(fmt(sm["ap50_shifted"]))
        rows.append(row)

    print_table("TABLE 3: Cross-Corruption Robustness — AP50 on Shifted Frames", headers, rows)
    return headers, rows


def table4_temporal(base_dir: str):
    """Table 4: Temporal shift patterns."""
    strategies = ["no_adapt", "fan", "sdmr", "tent", "eata"]
    patterns = [
        ("v7_gradual_brightness", "Gradual Brightness"),
        ("v7_recurring_blur", "Recurring Blur"),
        ("v7_mixed_shifts", "Mixed Shifts"),
    ]

    headers = ["Pattern"] + strategies
    rows = []

    for exp_name, label in patterns:
        row = [label]
        for strategy in strategies:
            res = load_results(base_dir, exp_name, strategy)
            if res:
                row.append(fmt(res.get("mean_ap50")))
            else:
                row.append("-")
        rows.append(row)

    print_table("TABLE 4: Temporal Shift Patterns — Mean AP50", headers, rows)
    return headers, rows


def table5_ablation(base_dir: str):
    """Table 5: FAN ablation study."""
    configs = [
        ("v7_abl_baseline", "no_adapt", "No Adapt (baseline)"),
        ("v7_noise_0_7", "fan", "FAN default [2,4,6] α=0.8"),
        ("v7_abl_layers24", "fan", "FAN [2,4]"),
        ("v7_abl_layers46", "fan", "FAN [4,6]"),
        ("v7_abl_layers2468", "fan", "FAN [2,4,6,8]"),
        ("v7_abl_strength04", "fan", "FAN α=0.4"),
        ("v7_abl_strength12", "fan", "FAN α=1.2"),
        ("v7_abl_ema02", "fan", "FAN ema=0.2"),
    ]

    headers = ["Config", "AP50↑", "AP50_shift↑", "Δ_vs_baseline"]
    rows = []
    baseline_shift = None

    for exp_name, strategy, label in configs:
        res = load_results(base_dir, exp_name, strategy)
        frames = load_frames(base_dir, exp_name, strategy)
        sm = compute_shifted_metrics(frames)

        ap50 = res.get("mean_ap50") if res else None
        ap50_shift = sm["ap50_shifted"]

        if label.startswith("No Adapt"):
            baseline_shift = ap50_shift

        delta = None
        if baseline_shift is not None and ap50_shift is not None:
            delta = ap50_shift - baseline_shift

        rows.append([label, fmt(ap50), fmt(ap50_shift),
                     fmt(delta, 4) if delta is not None else "-"])

    print_table("TABLE 5: FAN Ablation Study (Noise 0.7)", headers, rows)
    return headers, rows


def table6_multi_seed(base_dir: str):
    """Table 6: Multi-seed statistical significance."""
    strategies = ["no_adapt", "fan", "sdmr", "tent", "eata"]
    seeds = [42, 123, 456]
    severities = ["0_5", "0_7"]

    headers = ["Severity", "Strategy", "Seed 42", "Seed 123", "Seed 456",
               "Mean±Std"]
    rows = []

    for sev in severities:
        for strategy in strategies:
            vals = []
            seed_vals = []
            for seed in seeds:
                if seed == 42:
                    exp_name = f"v7_noise_{sev}"
                else:
                    exp_name = f"v7_noise_{sev}_seed{seed}"
                frames = load_frames(base_dir, exp_name, strategy)
                sm = compute_shifted_metrics(frames)
                v = sm["ap50_shifted"]
                seed_vals.append(fmt(v))
                if v is not None:
                    vals.append(v)

            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                mean_std = f"{mean:.4f}±{std:.4f}"
            else:
                mean_std = "-"

            rows.append([
                sev.replace("_", ".") if strategy == strategies[0] else "",
                strategy,
                *seed_vals,
                mean_std,
            ])
        rows.append(["", "", "", "", "", ""])  # separator

    print_table("TABLE 6: Multi-Seed Statistical Significance", headers, rows)
    return headers, rows


def per_frame_analysis(base_dir: str):
    """Detailed per-frame analysis for FAN vs no_adapt."""
    exp_name = "v7_noise_0_5"

    fan_frames = load_frames(base_dir, exp_name, "fan")
    noadapt_frames = load_frames(base_dir, exp_name, "no_adapt")

    if not fan_frames or not noadapt_frames:
        print("\n  Per-frame analysis: waiting for results...")
        return

    print(f"\n{'='*80}")
    print("  PER-FRAME ANALYSIS: FAN vs No-Adapt (Noise 0.5)")
    print(f"{'='*80}")

    # Compare shifted frames
    fan_shifted = {f["frame_idx"]: f for f in fan_frames if f.get("is_shifted")}
    na_shifted = {f["frame_idx"]: f for f in noadapt_frames if f.get("is_shifted")}

    common = set(fan_shifted.keys()) & set(na_shifted.keys())
    helped = 0
    hurt = 0
    same = 0
    total_delta = 0

    for idx in sorted(common):
        delta = fan_shifted[idx]["ap50"] - na_shifted[idx]["ap50"]
        total_delta += delta
        if delta > 0.001:
            helped += 1
        elif delta < -0.001:
            hurt += 1
        else:
            same += 1

    print(f"  Shifted frames analyzed: {len(common)}")
    print(f"  FAN helped:  {helped} frames ({100*helped/max(len(common),1):.1f}%)")
    print(f"  FAN hurt:    {hurt} frames ({100*hurt/max(len(common),1):.1f}%)")
    print(f"  Same:        {same} frames ({100*same/max(len(common),1):.1f}%)")
    print(f"  Total delta: {total_delta:.4f}")
    print(f"  Mean delta:  {total_delta/max(len(common),1):.4f}")

    # FAN mode distribution
    fan_modes = defaultdict(int)
    for f in fan_frames:
        fan_modes[f.get("fan_mode", "?")] += 1
    print(f"\n  FAN mode distribution:")
    for mode, count in sorted(fan_modes.items()):
        print(f"    {mode}: {count} frames")


def main():
    parser = argparse.ArgumentParser(description="V7 Results Aggregator")
    parser.add_argument("--output-dir", default="./outputs",
                        help="Base output directory")
    parser.add_argument("--latex", action="store_true",
                        help="Also generate LaTeX tables")
    parser.add_argument("--save", type=str, default=None,
                        help="Save aggregated results to JSON file")
    args = parser.parse_args()

    base = args.output_dir

    print(f"\nV7 Results Aggregation — {base}")
    print(f"{'='*80}")

    # Check how many results exist
    result_count = 0
    for root, dirs, files in os.walk(base):
        if "results.json" in files and "v7_" in root:
            result_count += 1
    print(f"Found {result_count} V7 result files")

    # Generate all tables
    t1 = table1_main_comparison(base)
    t2 = table2_noise_sweep(base)
    t3 = table3_cross_corruption(base)
    t4 = table4_temporal(base)
    t5 = table5_ablation(base)
    t6 = table6_multi_seed(base)

    # Per-frame analysis
    per_frame_analysis(base)

    # Save aggregated results
    if args.save:
        all_results = {}
        for root, dirs, files in os.walk(base):
            if "results.json" in files and "v7_" in root:
                rel = os.path.relpath(root, base)
                with open(os.path.join(root, "results.json")) as f:
                    all_results[rel] = json.load(f)
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved aggregated results to {args.save}")

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
