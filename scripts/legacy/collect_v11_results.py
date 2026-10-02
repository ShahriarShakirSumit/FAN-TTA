#!/usr/bin/env python3
"""
Collect and analyze V11 experiment results.

Checks for completed results and generates updated tables/figures.
Run periodically to check progress.
"""

import json
import os
import glob
from collections import defaultdict

OUTPUT_DIR = "./outputs"
PAPER_DIR = "./paper"

def find_v11_results():
    """Find all V11 results."""
    results = {}
    
    # Search for v11_* experiment directories
    for results_file in glob.glob(os.path.join(OUTPUT_DIR, "v11_*", "*", "results.json")):
        parts = results_file.split(os.sep)
        exp_name = parts[-3]  # e.g., v11_coco2nightowls
        strategy = parts[-2]  # e.g., fan, no_adapt
        
        with open(results_file) as f:
            data = json.load(f)
        
        if exp_name not in results:
            results[exp_name] = {}
        
        # Extract key metrics
        metrics = data.get("metrics", data)
        ap50 = metrics.get("mean_ap50", metrics.get("ap50", None))
        map50_95 = metrics.get("mean_map50_95", metrics.get("map50_95", None))
        
        results[exp_name][strategy] = {
            "ap50": ap50,
            "map50_95": map50_95,
            "source": "v11",
        }
    
    return results


def print_results_table(results):
    """Print results in a readable table."""
    if not results:
        print("No V11 results found yet.")
        return
    
    print(f"\n{'='*80}")
    print(f"V11 RESULTS — {len(results)} experiments found")
    print(f"{'='*80}")
    
    # Group by experiment type
    groups = {
        "Cross-Domain": {},
        "YOLOv8n": {},
        "Multi-Seed": {},
    }
    
    for exp, strats in sorted(results.items()):
        if "nightowls" in exp or "ecp" in exp:
            groups["Cross-Domain"][exp] = strats
        elif "yolov8n" in exp:
            groups["YOLOv8n"][exp] = strats
        elif "seed" in exp:
            groups["Multi-Seed"][exp] = strats
        else:
            groups.setdefault("Other", {})[exp] = strats
    
    for group_name, group_exps in groups.items():
        if not group_exps:
            continue
        print(f"\n--- {group_name} ---")
        print(f"{'Experiment':<35} {'Strategy':<15} {'AP50':>8} {'mAP50-95':>10}")
        print("-" * 70)
        
        for exp, strats in sorted(group_exps.items()):
            for strat, metrics in sorted(strats.items()):
                ap50 = metrics.get("ap50")
                map50_95 = metrics.get("map50_95")
                ap50_str = f"{ap50*100:.2f}" if ap50 else "N/A"
                map_str = f"{map50_95*100:.2f}" if map50_95 else "N/A"
                print(f"{exp:<35} {strat:<15} {ap50_str:>8} {map_str:>10}")
    
    # FAN vs No-Adapt comparison
    print(f"\n{'='*80}")
    print("FAN vs No-Adapt Comparison")
    print(f"{'='*80}")
    
    for exp, strats in sorted(results.items()):
        if "fan" in strats and "no_adapt" in strats:
            fan_ap50 = strats["fan"].get("ap50", 0) or 0
            na_ap50 = strats["no_adapt"].get("ap50", 0) or 0
            delta = (fan_ap50 - na_ap50) * 100
            winner = "FAN" if delta > 0.005 else ("TIE" if abs(delta) < 0.005 else "No-Adapt")
            print(f"{exp:<35} Δ={delta:+.3f}% → {winner}")


def save_results_json(results):
    """Save results to JSON."""
    output_path = os.path.join(OUTPUT_DIR, "v11_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    print("Collecting V11 results...")
    results = find_v11_results()
    print_results_table(results)
    
    if results:
        save_results_json(results)
    
    # Summary
    total_expected = 46
    total_found = sum(len(s) for s in results.values())
    print(f"\nProgress: {total_found}/{total_expected} results collected")
    
    if total_found == total_expected:
        print("ALL V11 RESULTS COMPLETE!")
