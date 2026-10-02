"""
Results analysis for SDMR experiments.

Loads experiment outputs, computes summary statistics,
generates comparison tables (LaTeX), and ranks strategies.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class StrategyResult:
    """Summary of one strategy's results."""
    strategy: str
    mean_ap50: float = 0.0
    mean_map50_95: float = 0.0
    mean_f1: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    fps: float = 0.0
    total_time_s: float = 0.0
    n_frames: int = 0
    # Adaptation metrics
    drop_magnitude: float = 0.0
    recovery_time: int = 0
    detection_delay: float = 0.0
    # Per-segment (clean vs shift)
    clean_ap50: float = 0.0
    shift_ap50: float = 0.0
    post_shift_ap50: float = 0.0
    # Adaptation stats
    total_adaptations: int = 0
    adaptation_ratio: float = 0.0  # fraction of frames with adaptation

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class ResultsAnalyzer:
    """
    Load and analyze experiment results across strategies.

    Usage:
        analyzer = ResultsAnalyzer("outputs/")
        analyzer.load_all()
        table = analyzer.comparison_table()
        print(table.to_latex())
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.results: dict[str, StrategyResult] = {}
        self.frame_data: dict[str, pd.DataFrame] = {}

    def load_strategy(self, strategy_dir: str | Path) -> Optional[StrategyResult]:
        """Load results for a single strategy."""
        d = Path(strategy_dir)
        results_file = d / "results.json"
        frames_file = d / "frame_results.json"

        if not results_file.exists():
            return None

        with open(results_file) as f:
            data = json.load(f)

        sr = StrategyResult(strategy=data.get("strategy", d.name))
        agg = data.get("aggregate_metrics", {})
        sr.mean_ap50 = agg.get("mean_ap50", 0)
        sr.mean_map50_95 = agg.get("mean_map50_95", 0)
        sr.mean_f1 = agg.get("mean_f1", 0)
        sr.mean_precision = agg.get("mean_precision", 0)
        sr.mean_recall = agg.get("mean_recall", 0)
        sr.fps = data.get("fps", 0)
        sr.total_time_s = data.get("total_time_s", 0)
        sr.n_frames = data.get("n_frames", 0)

        adapt = data.get("adaptation_metrics", {})
        sr.drop_magnitude = adapt.get("drop_magnitude", 0)
        sr.recovery_time = adapt.get("recovery_time", 0)
        sr.detection_delay = adapt.get("detection_delay", 0)

        # Load frame-level results
        if frames_file.exists():
            with open(frames_file) as f:
                frames = json.load(f)
            df = pd.DataFrame(frames)
            self.frame_data[sr.strategy] = df

            # Compute per-segment metrics
            if "is_shifted" in df.columns and "ap50" in df.columns:
                clean_mask = ~df["is_shifted"]
                shift_mask = df["is_shifted"]
                sr.clean_ap50 = float(df.loc[clean_mask, "ap50"].mean()) if clean_mask.any() else 0
                sr.shift_ap50 = float(df.loc[shift_mask, "ap50"].mean()) if shift_mask.any() else 0

            if "action" in df.columns:
                sr.total_adaptations = int((df["action"] != "none").sum())
                sr.adaptation_ratio = sr.total_adaptations / max(len(df), 1)

        self.results[sr.strategy] = sr
        return sr

    def load_all(self) -> int:
        """Load all strategy results from output directory."""
        count = 0
        if not self.output_dir.exists():
            return 0

        for d in sorted(self.output_dir.iterdir()):
            if d.is_dir() and (d / "results.json").exists():
                r = self.load_strategy(d)
                if r:
                    count += 1
        return count

    def comparison_table(self) -> "ComparisonTable":
        """Generate a comparison table of all strategies."""
        return ComparisonTable(self.results)

    def rank_by(self, metric: str = "mean_ap50") -> list[StrategyResult]:
        """Rank strategies by a metric (descending)."""
        return sorted(
            self.results.values(),
            key=lambda r: getattr(r, metric, 0),
            reverse=True,
        )

    def compute_relative_improvement(
        self, baseline: str = "no_adapt", target: str = "sdmr"
    ) -> dict[str, float]:
        """Compute relative improvement of target over baseline."""
        if baseline not in self.results or target not in self.results:
            return {}

        base = self.results[baseline]
        targ = self.results[target]

        improvements = {}
        for metric in ["mean_ap50", "mean_map50_95", "mean_f1"]:
            bv = getattr(base, metric, 0)
            tv = getattr(targ, metric, 0)
            if abs(bv) > 1e-6:
                improvements[metric] = (tv - bv) / abs(bv) * 100
            else:
                improvements[metric] = 0

        return improvements

    def get_frame_dataframe(self) -> pd.DataFrame:
        """Combine all frame-level data into a single DataFrame."""
        dfs = []
        for strategy, df in self.frame_data.items():
            df = df.copy()
            df["strategy"] = strategy
            dfs.append(df)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()


class ComparisonTable:
    """
    Generate LaTeX comparison tables for the paper.

    Formats:
    - Main results table (Table 1 in paper)
    - Per-scenario breakdown
    - Adaptation cost analysis
    """

    def __init__(self, results: dict[str, StrategyResult]):
        self.results = results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        rows = []
        for strategy, r in sorted(self.results.items()):
            rows.append({
                "Strategy": strategy.replace("_", " ").title(),
                "AP@50": r.mean_ap50,
                "mAP": r.mean_map50_95,
                "F1": r.mean_f1,
                "Clean AP": r.clean_ap50,
                "Shift AP": r.shift_ap50,
                "Drop": r.drop_magnitude,
                "Recovery": r.recovery_time,
                "FPS": r.fps,
                "Adapt%": r.adaptation_ratio * 100,
            })
        return pd.DataFrame(rows)

    def to_latex(self, caption: str = "", label: str = "tab:main") -> str:
        """Generate LaTeX table for NeurIPS format."""
        df = self.to_dataframe()
        if df.empty:
            return "% No results to display"

        # Find best values for bolding
        best = {}
        for col in ["AP@50", "mAP", "F1", "Clean AP", "Shift AP", "FPS"]:
            if col in df.columns:
                best[col] = df[col].max()

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + caption + "}")
        lines.append(r"\label{" + label + "}")
        lines.append(r"\resizebox{\textwidth}{!}{")
        lines.append(r"\begin{tabular}{l" + "c" * 8 + "}")
        lines.append(r"\toprule")
        lines.append(
            r"Strategy & AP@50$\uparrow$ & mAP$\uparrow$ & F1$\uparrow$ "
            r"& Clean$\uparrow$ & Shift$\uparrow$ & Drop$\downarrow$ "
            r"& Recovery$\downarrow$ & FPS$\uparrow$ \\"
        )
        lines.append(r"\midrule")

        for _, row in df.iterrows():
            vals = []
            vals.append(str(row["Strategy"]))
            for col in ["AP@50", "mAP", "F1", "Clean AP", "Shift AP"]:
                v = row.get(col, 0)
                s = f"{v:.3f}"
                if col in best and abs(v - best[col]) < 1e-6:
                    s = r"\textbf{" + s + "}"
                vals.append(s)
            vals.append(f"{row.get('Drop', 0):.3f}")
            vals.append(f"{int(row.get('Recovery', 0))}")
            vals.append(f"{row.get('FPS', 0):.1f}")
            lines.append(" & ".join(vals) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate Markdown table for quick inspection."""
        df = self.to_dataframe()
        if df.empty:
            return "No results"

        lines = []
        headers = list(df.columns)
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for _, row in df.iterrows():
            vals = []
            for col in headers:
                v = row[col]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")

        return "\n".join(lines)
