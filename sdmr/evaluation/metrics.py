"""
Evaluation module for streaming object detection under distribution shift.

Provides:
- StreamEvaluator: tracks per-frame and per-window metrics
- Metrics: AP, mAP, F1, precision, recall for detection
- Shift-aware metrics: recovery time, adaptation cost, drop magnitude
- Computation-aware metrics: GFLOPs, latency tracking
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionMetrics:
    """Per-frame or aggregated detection metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ap50: float = 0.0
    ap75: float = 0.0
    map50_95: float = 0.0
    num_predictions: int = 0
    num_ground_truths: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}precision": self.precision,
            f"{p}recall": self.recall,
            f"{p}f1": self.f1,
            f"{p}ap50": self.ap50,
            f"{p}ap75": self.ap75,
            f"{p}map50_95": self.map50_95,
            f"{p}num_preds": float(self.num_predictions),
            f"{p}num_gts": float(self.num_ground_truths),
            f"{p}tp": float(self.true_positives),
            f"{p}fp": float(self.false_positives),
            f"{p}fn": float(self.false_negatives),
        }


@dataclass
class AdaptationMetrics:
    """Metrics specific to adaptation behavior under shift."""
    # Performance drop at shift onset
    drop_magnitude: float = 0.0         # max performance drop
    recovery_time: int = 0              # frames to recover to 90% of pre-shift
    adaptation_cost_flops: float = 0.0  # extra FLOPs from adaptation
    adaptation_cost_time: float = 0.0   # extra wall-time from adaptation
    steady_state_gap: float = 0.0       # gap between adapted and oracle
    num_shifts_detected: int = 0
    false_alarm_rate: float = 0.0
    detection_delay: float = 0.0        # avg delay in detecting true shifts

    def to_dict(self, prefix: str = "adapt") -> dict[str, float]:
        return {
            f"{prefix}_drop_mag": self.drop_magnitude,
            f"{prefix}_recovery_time": float(self.recovery_time),
            f"{prefix}_cost_flops": self.adaptation_cost_flops,
            f"{prefix}_cost_time": self.adaptation_cost_time,
            f"{prefix}_steady_gap": self.steady_state_gap,
            f"{prefix}_n_shifts": float(self.num_shifts_detected),
            f"{prefix}_false_alarm": self.false_alarm_rate,
            f"{prefix}_detect_delay": self.detection_delay,
        }


@dataclass
class FrameRecord:
    """Record for a single frame evaluation."""
    frame_idx: int
    timestamp: float
    metrics: DetectionMetrics
    action_taken: str = "none"  # e.g., "none", "bn_refresh", "head_only", "full"
    shift_detected: bool = False
    true_shift: bool = False
    latency_ms: float = 0.0
    adaptation_time_ms: float = 0.0
    signal_summary: Optional[dict[str, float]] = None


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / (union + 1e-8)


def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between prediction and GT boxes.

    Args:
        pred_boxes: (M, 4) array [x1, y1, x2, y2]
        gt_boxes: (N, 4) array [x1, y1, x2, y2]

    Returns:
        (M, N) IoU matrix
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    x1 = np.maximum(pred_boxes[:, 0:1], gt_boxes[:, 0:1].T)
    y1 = np.maximum(pred_boxes[:, 1:2], gt_boxes[:, 1:2].T)
    x2 = np.minimum(pred_boxes[:, 2:3], gt_boxes[:, 2:3].T)
    y2 = np.minimum(pred_boxes[:, 3:4], gt_boxes[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_pred[:, None] + area_gt[None, :] - inter

    return inter / (union + 1e-8)


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """
    Match predictions to ground truth using greedy IoU matching.

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    # Sort predictions by score (descending)
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    tp, fp = 0, 0
    for i in range(len(pred_boxes)):
        ious = iou_matrix[i]
        # Mask already matched GTs
        ious = ious * (~gt_matched).astype(float)
        best_gt = np.argmax(ious)
        if ious[best_gt] >= iou_threshold:
            tp += 1
            gt_matched[best_gt] = True
        else:
            fp += 1

    fn = int(np.sum(~gt_matched))
    return tp, fp, fn


def compute_ap_at_iou(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision at a given IoU threshold."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0:
        return 0.0
    if len(pred_boxes) == 0:
        return 0.0

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    tp_list = []
    fp_list = []

    for i in range(len(pred_boxes)):
        ious = iou_matrix[i] * (~gt_matched).astype(float)
        best_gt = np.argmax(ious)
        if ious[best_gt] >= iou_threshold:
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[best_gt] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (len(gt_boxes) + 1e-8)

    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if np.any(mask):
            ap += np.max(precision[mask])
    ap /= 11.0
    return float(ap)


def compute_frame_metrics(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    score_threshold: float = 0.25,
) -> DetectionMetrics:
    """Compute detection metrics for a single frame."""
    # Filter by score threshold
    mask = pred_scores >= score_threshold
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]

    tp, fp, fn = match_predictions_to_gt(pred_boxes, gt_boxes, pred_scores, 0.5)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    ap50 = compute_ap_at_iou(pred_boxes, pred_scores, gt_boxes, 0.5)
    ap75 = compute_ap_at_iou(pred_boxes, pred_scores, gt_boxes, 0.75)

    # mAP50:95
    aps = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        aps.append(compute_ap_at_iou(pred_boxes, pred_scores, gt_boxes, iou_thresh))
    map50_95 = float(np.mean(aps))

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        ap50=ap50,
        ap75=ap75,
        map50_95=map50_95,
        num_predictions=len(pred_boxes),
        num_ground_truths=len(gt_boxes),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


class StreamEvaluator:
    """
    Evaluator for streaming detection experiments.

    Tracks frame-level metrics, detects performance drops,
    and computes adaptation-aware aggregate metrics.
    """

    def __init__(
        self,
        score_threshold: float = 0.25,
        recovery_threshold: float = 0.9,
    ):
        self.score_threshold = score_threshold
        self.recovery_threshold = recovery_threshold  # fraction of pre-shift perf
        self.records: list[FrameRecord] = []
        self._pre_shift_perf: Optional[float] = None
        self._shift_onset_idx: Optional[int] = None
        self._recovered = True

    def record_frame(
        self,
        frame_idx: int,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        action_taken: str = "none",
        shift_detected: bool = False,
        true_shift: bool = False,
        latency_ms: float = 0.0,
        adaptation_time_ms: float = 0.0,
        signal_summary: Optional[dict[str, float]] = None,
    ) -> DetectionMetrics:
        """Record evaluation for one frame."""
        metrics = compute_frame_metrics(
            pred_boxes, pred_scores, gt_boxes, self.score_threshold
        )

        record = FrameRecord(
            frame_idx=frame_idx,
            timestamp=time.time(),
            metrics=metrics,
            action_taken=action_taken,
            shift_detected=shift_detected,
            true_shift=true_shift,
            latency_ms=latency_ms,
            adaptation_time_ms=adaptation_time_ms,
            signal_summary=signal_summary,
        )
        self.records.append(record)
        return metrics

    def get_aggregate_metrics(self) -> dict[str, float]:
        """Compute aggregate metrics over all recorded frames."""
        if not self.records:
            return {}

        metrics_keys = ["precision", "recall", "f1", "ap50", "ap75", "map50_95"]
        agg: dict[str, float] = {}

        for key in metrics_keys:
            values = [getattr(r.metrics, key) for r in self.records]
            agg[f"mean_{key}"] = float(np.mean(values))
            agg[f"std_{key}"] = float(np.std(values))
            agg[f"min_{key}"] = float(np.min(values))

        # Latency stats
        latencies = [r.latency_ms for r in self.records]
        agg["mean_latency_ms"] = float(np.mean(latencies))
        agg["p95_latency_ms"] = float(np.percentile(latencies, 95))

        # Adaptation stats
        adaptation_times = [r.adaptation_time_ms for r in self.records]
        agg["total_adaptation_ms"] = float(np.sum(adaptation_times))
        agg["mean_adaptation_ms"] = float(np.mean(adaptation_times))

        # Actions taken
        actions = [r.action_taken for r in self.records]
        action_set = set(actions)
        for a in action_set:
            agg[f"action_count_{a}"] = float(actions.count(a))
        agg["action_diversity"] = float(len(action_set))

        # Shift detection accuracy
        true_shifts = [r.true_shift for r in self.records]
        detected_shifts = [r.shift_detected for r in self.records]
        if any(true_shifts):
            # True positive: shift detected when true shift present
            tp_shift = sum(1 for t, d in zip(true_shifts, detected_shifts) if t and d)
            fn_shift = sum(1 for t, d in zip(true_shifts, detected_shifts) if t and not d)
            fp_shift = sum(1 for t, d in zip(true_shifts, detected_shifts) if not t and d)
            agg["shift_precision"] = tp_shift / (tp_shift + fp_shift + 1e-8)
            agg["shift_recall"] = tp_shift / (tp_shift + fn_shift + 1e-8)

        agg["total_frames"] = float(len(self.records))
        return agg

    def get_window_metrics(
        self, window_start: int, window_end: int
    ) -> dict[str, float]:
        """Compute metrics for a specific frame window."""
        window_records = [
            r for r in self.records
            if window_start <= r.frame_idx < window_end
        ]
        if not window_records:
            return {}

        metrics_keys = ["precision", "recall", "f1", "ap50", "map50_95"]
        result: dict[str, float] = {}
        for key in metrics_keys:
            values = [getattr(r.metrics, key) for r in window_records]
            result[f"window_{key}"] = float(np.mean(values))

        return result

    def compute_adaptation_metrics(
        self, shift_schedule: list[tuple[int, int]]
    ) -> AdaptationMetrics:
        """
        Compute adaptation-specific metrics given known shift schedule.

        Args:
            shift_schedule: List of (onset_frame, end_frame) for each shift.
        """
        if not self.records or not shift_schedule:
            return AdaptationMetrics()

        # Build frame index -> metrics mapping
        frame_map = {r.frame_idx: r for r in self.records}
        all_frames = sorted(frame_map.keys())

        drops = []
        recovery_times = []
        detection_delays = []
        false_alarms = 0
        total_non_shift_frames = 0

        for onset, end in shift_schedule:
            # Pre-shift performance (last 20 frames before onset)
            pre_frames = [
                frame_map[f] for f in all_frames
                if onset - 20 <= f < onset and f in frame_map
            ]
            if not pre_frames:
                continue
            pre_perf = float(np.mean([r.metrics.map50_95 for r in pre_frames]))

            # Find max drop during shift
            shift_frames = [
                frame_map[f] for f in all_frames
                if onset <= f < end and f in frame_map
            ]
            if shift_frames:
                min_perf = min(r.metrics.map50_95 for r in shift_frames)
                drops.append(pre_perf - min_perf)

            # Recovery time
            recovery_target = pre_perf * self.recovery_threshold
            recovered_at = None
            for r in shift_frames:
                if r.metrics.map50_95 >= recovery_target:
                    recovered_at = r.frame_idx
                    break
            if recovered_at is not None:
                recovery_times.append(recovered_at - onset)

            # Detection delay: first shift_detected at or after onset
            for r in shift_frames:
                if r.shift_detected:
                    detection_delays.append(r.frame_idx - onset)
                    break

        # False alarms: shift detected during non-shift periods
        shift_ranges = set()
        for onset, end in shift_schedule:
            for f in range(onset, end):
                shift_ranges.add(f)
        for r in self.records:
            if r.frame_idx not in shift_ranges:
                total_non_shift_frames += 1
                if r.shift_detected:
                    false_alarms += 1

        return AdaptationMetrics(
            drop_magnitude=float(np.mean(drops)) if drops else 0.0,
            recovery_time=int(np.mean(recovery_times)) if recovery_times else 0,
            adaptation_cost_time=sum(r.adaptation_time_ms for r in self.records),
            num_shifts_detected=sum(1 for r in self.records if r.shift_detected),
            false_alarm_rate=false_alarms / (total_non_shift_frames + 1e-8),
            detection_delay=float(np.mean(detection_delays)) if detection_delays else 0.0,
        )

    def export_records(self) -> list[dict]:
        """Export all frame records as list of dicts for logging."""
        return [
            {
                "frame_idx": r.frame_idx,
                "timestamp": r.timestamp,
                **r.metrics.to_dict(),
                "action": r.action_taken,
                "shift_detected": r.shift_detected,
                "true_shift": r.true_shift,
                "latency_ms": r.latency_ms,
                "adaptation_ms": r.adaptation_time_ms,
            }
            for r in self.records
        ]
