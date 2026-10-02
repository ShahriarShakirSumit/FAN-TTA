"""
Stream evaluator module for end-to-end streaming detection experiments.

Re-exports from metrics.py for convenience.
"""

from .metrics import (
    StreamEvaluator,
    DetectionMetrics,
    AdaptationMetrics,
    FrameRecord,
    compute_frame_metrics,
    compute_iou,
    compute_iou_matrix,
    compute_ap_at_iou,
    match_predictions_to_gt,
)

__all__ = [
    "StreamEvaluator",
    "DetectionMetrics",
    "AdaptationMetrics",
    "FrameRecord",
    "compute_frame_metrics",
    "compute_iou",
    "compute_iou_matrix",
    "compute_ap_at_iou",
    "match_predictions_to_gt",
]
