"""
Analysis and visualization module for SDMR experiments.

Generates publication-quality figures and LaTeX tables for NeurIPS paper.
"""

from .results_analyzer import ResultsAnalyzer, ComparisonTable
from .plot_utils import StreamPlotter, create_comparison_figure

__all__ = [
    "ResultsAnalyzer",
    "ComparisonTable",
    "StreamPlotter",
    "create_comparison_figure",
]
