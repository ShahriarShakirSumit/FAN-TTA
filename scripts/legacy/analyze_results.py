#!/usr/bin/env python3
"""
Analyze SDMR experiment results and generate paper-ready outputs.

Usage:
    python scripts/analyze_results.py --results-dir ./outputs
    python scripts/analyze_results.py --results-dir outputs/ --figures-dir figures/
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdmr.experiments.results_analyzer import ResultsAnalyzer
from sdmr.experiments.plot_utils import StreamPlotter, create_comparison_figure


def main():
    parser = argparse.ArgumentParser(description="Analyze SDMR experiment results")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing experiment outputs")
    parser.add_argument("--experiment", type=str, default="coco_person_abrupt_noise",
                        help="Experiment name subdirectory (e.g., coco_person_abrupt_noise)")
    parser.add_argument("--figures-dir", type=str, default="figures",
                        help="Directory to save figures")
    parser.add_argument("--shift-onset", type=int, default=300,
                        help="Frame index of shift onset")
    parser.add_argument("--shift-end", type=int, default=600,
                        help="Frame index of shift end")
    args = parser.parse_args()

    # The results dir contains experiment subdirs, each with strategy subdirs
    experiment_dir = Path(args.results_dir) / args.experiment
    if not experiment_dir.exists():
        # Maybe the user pointed directly to the experiment dir
        experiment_dir = Path(args.results_dir)

    print("=" * 60)
    print("SDMR Results Analysis")
    print("=" * 60)

    # Load results
    analyzer = ResultsAnalyzer(str(experiment_dir))
    n = analyzer.load_all()
    print(f"\nLoaded {n} strategy results from {experiment_dir}")

    if n == 0:
        print("No results found. Ensure experiments have completed.")
        return

    # Print comparison table
    table = analyzer.comparison_table()
    print("\n" + "=" * 60)
    print("COMPARISON TABLE (Markdown)")
    print("=" * 60)
    print(table.to_markdown())

    # Rank by AP@50
    print("\n" + "=" * 60)
    print("STRATEGY RANKING by AP@50")
    print("=" * 60)
    for i, r in enumerate(analyzer.rank_by("mean_ap50"), 1):
        print(f"  {i}. {r.strategy:15s}  AP50={r.mean_ap50:.4f}  "
              f"mAP={r.mean_map50_95:.4f}  F1={r.mean_f1:.4f}  "
              f"FPS={r.fps:.1f}  Adapt%={r.adaptation_ratio*100:.1f}%")

    # Compute improvements
    if "no_adapt" in analyzer.results and "sdmr" in analyzer.results:
        impr = analyzer.compute_relative_improvement("no_adapt", "sdmr")
        print("\n" + "=" * 60)
        print("SDMR vs No Adapt (relative improvement)")
        print("=" * 60)
        for metric, v in impr.items():
            print(f"  {metric}: {v:+.2f}%")

    # Generate figures
    shift_schedule = [(args.shift_onset, args.shift_end)]
    figures = create_comparison_figure(
        str(experiment_dir), args.figures_dir, shift_schedule
    )
    print(f"\nGenerated {len(figures)} figures in {args.figures_dir}/")
    for f in figures:
        print(f"  - {f}")

    # Generate LaTeX table
    latex = table.to_latex(
        caption="Comparison of adaptation strategies on COCO-Person "
                "streaming detection benchmark under abrupt distribution shift.",
        label="tab:main_results",
    )
    latex_path = Path(args.figures_dir) / "main_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {latex_path}")

    # Summary JSON
    summary = {
        "n_strategies": n,
        "best_strategy": analyzer.rank_by("mean_ap50")[0].strategy,
        "best_ap50": analyzer.rank_by("mean_ap50")[0].mean_ap50,
        "strategies": {s: r.to_dict() for s, r in analyzer.results.items()},
    }
    summary_path = Path(experiment_dir) / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
