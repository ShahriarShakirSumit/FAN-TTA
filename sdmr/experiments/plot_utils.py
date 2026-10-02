"""
Plotting utilities for SDMR experiments.

Generates publication-quality figures for NeurIPS:
- Stream timeline with shift regions highlighted
- AP over time for multiple strategies
- Signal evolution plots
- Adaptation action distribution
- Scatter: harm vs actual AP drop
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Optional
import pandas as pd


# NeurIPS-compatible style
NEURIPS_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "figure.figsize": (6, 3.5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

STRATEGY_COLORS = {
    "sdmr": "#2196F3",          # Blue (ours)
    "no_adapt": "#9E9E9E",      # Gray
    "bn_adapt": "#4CAF50",      # Green
    "periodic_ft": "#FF9800",   # Orange
    "tent": "#F44336",          # Red
    "eata": "#9C27B0",          # Purple
    "cotta": "#00BCD4",         # Cyan
    "sar": "#795548",           # Brown
    "rotta": "#E91E63",         # Pink
    "oracle": "#FFD700",        # Gold
}

STRATEGY_LABELS = {
    "sdmr": "SDMR (Ours)",
    "no_adapt": "No Adapt",
    "bn_adapt": "BN Adapt",
    "periodic_ft": "Periodic FT",
    "tent": "TENT",
    "eata": "EATA",
    "cotta": "CoTTA",
    "sar": "SAR",
    "rotta": "RoTTA",
    "oracle": "Oracle",
}


class StreamPlotter:
    """
    Generate streaming experiment timeline plots.

    Main figure: AP@50 over time for multiple strategies,
    with shift regions shaded.
    """

    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_ap_timeline(
        self,
        frame_data: dict[str, pd.DataFrame],
        shift_schedule: list[tuple[int, int]],
        metric: str = "ap50",
        smoothing_window: int = 10,
        save_name: str = "ap_timeline.pdf",
        title: str = "",
    ) -> str:
        """
        Plot AP@50 over time for multiple strategies.

        Args:
            frame_data: {strategy_name: DataFrame with 'frame_idx' and metric columns}
            shift_schedule: List of (onset, end) shift periods.
            metric: Column name to plot (e.g., "ap50", "map50_95", "f1").
            smoothing_window: Rolling average window.
            save_name: Output filename.
            title: Plot title.

        Returns:
            Path to saved figure.
        """
        with plt.style.context(NEURIPS_STYLE):
            fig, ax = plt.subplots(figsize=(7, 3.5))

            # Shade shift regions
            for onset, end in shift_schedule:
                ax.axvspan(onset, end, alpha=0.15, color="#FF6B6B",
                          label="Shift" if onset == shift_schedule[0][0] else "")

            # Plot each strategy
            for strategy, df in frame_data.items():
                if metric not in df.columns:
                    continue
                color = STRATEGY_COLORS.get(strategy, "#555555")
                label = STRATEGY_LABELS.get(strategy, strategy)

                y = df[metric].rolling(window=smoothing_window, min_periods=1).mean()
                ax.plot(df["frame_idx"], y, color=color, label=label,
                       alpha=0.9, linewidth=1.5 if strategy == "sdmr" else 1.0)

            ax.set_xlabel("Frame Index")
            ax.set_ylabel(metric.upper().replace("_", " "))
            ax.set_ylim(0, 1)
            if title:
                ax.set_title(title)

            # Legend outside
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                     frameon=True, framealpha=0.9, edgecolor="gray")

            fig.tight_layout()
            path = self.output_dir / save_name
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return str(path)

    def plot_signal_evolution(
        self,
        df: pd.DataFrame,
        signals: list[str],
        shift_schedule: list[tuple[int, int]],
        save_name: str = "signals.pdf",
    ) -> str:
        """Plot diagnosis signal evolution over time."""
        n_signals = len(signals)
        with plt.style.context(NEURIPS_STYLE):
            fig, axes = plt.subplots(n_signals, 1, figsize=(7, 2 * n_signals),
                                     sharex=True)
            if n_signals == 1:
                axes = [axes]

            for i, sig in enumerate(signals):
                if sig not in df.columns:
                    continue
                ax = axes[i]
                for onset, end in shift_schedule:
                    ax.axvspan(onset, end, alpha=0.1, color="#FF6B6B")
                ax.plot(df["frame_idx"], df[sig], linewidth=0.8, color="#2196F3")
                ax.set_ylabel(sig.replace("_", "\n"), fontsize=8)
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Frame Index")
            fig.suptitle("Diagnosis Signal Evolution", fontsize=11)
            fig.tight_layout()
            path = self.output_dir / save_name
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return str(path)

    def plot_action_distribution(
        self,
        frame_data: dict[str, pd.DataFrame],
        save_name: str = "action_dist.pdf",
    ) -> str:
        """Bar chart: adaptation action distribution per strategy."""
        action_order = ["none", "bn_refresh", "head_only", "bounded_full", "reset_and_ft"]
        action_colors = {
            "none": "#BDBDBD",
            "bn_refresh": "#81C784",
            "head_only": "#64B5F6",
            "bounded_full": "#FFB74D",
            "reset_and_ft": "#E57373",
        }

        with plt.style.context(NEURIPS_STYLE):
            strategies = []
            data = {}
            for strat, df in frame_data.items():
                if "action" not in df.columns:
                    continue
                strategies.append(strat)
                counts = df["action"].value_counts()
                total = len(df)
                for a in action_order:
                    data.setdefault(a, []).append(
                        counts.get(a, 0) / total * 100
                    )

            if not strategies:
                return ""

            fig, ax = plt.subplots(figsize=(7, 3))
            x = np.arange(len(strategies))
            width = 0.15

            for i, action in enumerate(action_order):
                if action in data:
                    ax.bar(x + i * width, data[action], width,
                          label=action.replace("_", " ").title(),
                          color=action_colors.get(action, "#999"))

            ax.set_xticks(x + width * 2)
            ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies],
                              rotation=30, ha="right")
            ax.set_ylabel("% of frames")
            ax.set_title("Adaptation Action Distribution")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = self.output_dir / save_name
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return str(path)

    def plot_drop_recovery(
        self,
        frame_data: dict[str, pd.DataFrame],
        shift_onset: int,
        shift_end: int,
        metric: str = "ap50",
        window: int = 20,
        save_name: str = "drop_recovery.pdf",
    ) -> str:
        """
        Detailed view of AP drop at shift onset and recovery afterwards.
        Shows a zoomed-in view of [onset-50, end+100].
        """
        with plt.style.context(NEURIPS_STYLE):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            view_start = max(0, shift_onset - 50)
            view_end = shift_end + 100

            ax.axvspan(shift_onset, shift_end, alpha=0.15, color="#FF6B6B",
                      label="Shift")
            ax.axvline(shift_onset, color="#FF6B6B", linestyle="--", alpha=0.5)
            ax.axvline(shift_end, color="#FF6B6B", linestyle="--", alpha=0.5)

            for strat, df in frame_data.items():
                if metric not in df.columns:
                    continue
                mask = (df["frame_idx"] >= view_start) & (df["frame_idx"] <= view_end)
                sub = df[mask]
                y = sub[metric].rolling(window=window, min_periods=1).mean()
                color = STRATEGY_COLORS.get(strat, "#555")
                label = STRATEGY_LABELS.get(strat, strat)
                ax.plot(sub["frame_idx"], y, color=color, label=label,
                       linewidth=1.5 if strat == "sdmr" else 1.0)

            ax.set_xlabel("Frame Index")
            ax.set_ylabel(metric.upper())
            ax.set_title("AP Drop & Recovery at Shift Onset")
            ax.legend(loc="lower right", fontsize=8)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            path = self.output_dir / save_name
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return str(path)

    def plot_harm_vs_actual(
        self,
        predicted_harm: list[float],
        actual_drop: list[float],
        save_name: str = "harm_scatter.pdf",
    ) -> str:
        """Scatter: predicted harm vs actual AP drop (calibration quality)."""
        with plt.style.context(NEURIPS_STYLE):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(predicted_harm, actual_drop, alpha=0.3, s=10, color="#2196F3")
            # Perfect calibration line
            lims = [0, max(max(predicted_harm, default=0), max(actual_drop, default=0))]
            ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
            ax.set_xlabel("Predicted Harm (AP Drop)")
            ax.set_ylabel("Actual AP Drop")
            ax.set_title("Harm Predictor Calibration")
            fig.tight_layout()
            path = self.output_dir / save_name
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return str(path)


def create_comparison_figure(
    results_dir: str,
    output_dir: str = "figures",
    shift_schedule: Optional[list[tuple[int, int]]] = None,
) -> list[str]:
    """
    Generate all paper figures from experiment results.

    Args:
        results_dir: Directory containing strategy subdirectories.
        output_dir: Where to save figures.
        shift_schedule: List of (onset, end) for shift periods.

    Returns:
        List of saved figure paths.
    """
    from .results_analyzer import ResultsAnalyzer

    analyzer = ResultsAnalyzer(results_dir)
    n = analyzer.load_all()
    if n == 0:
        return []

    plotter = StreamPlotter(output_dir)
    saved = []

    # 1. AP timeline
    if analyzer.frame_data and shift_schedule:
        path = plotter.plot_ap_timeline(
            analyzer.frame_data, shift_schedule,
            save_name="fig1_ap_timeline.pdf",
            title="Streaming Detection Performance Under Distribution Shift",
        )
        saved.append(path)

    # 2. Action distribution
    if analyzer.frame_data:
        path = plotter.plot_action_distribution(
            analyzer.frame_data,
            save_name="fig2_action_dist.pdf",
        )
        if path:
            saved.append(path)

    # 3. Drop & recovery
    if analyzer.frame_data and shift_schedule:
        onset, end = shift_schedule[0]
        path = plotter.plot_drop_recovery(
            analyzer.frame_data, onset, end,
            save_name="fig3_drop_recovery.pdf",
        )
        saved.append(path)

    # 4. LaTeX table
    table = analyzer.comparison_table()
    table_path = Path(output_dir) / "table1_main.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write(table.to_latex(
            caption="Comparison of adaptation strategies on COCO-Person streaming benchmark. "
                    "SDMR achieves the best trade-off between detection performance and "
                    "adaptation cost.",
            label="tab:main",
        ))
    saved.append(str(table_path))

    return saved
