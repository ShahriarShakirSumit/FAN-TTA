"""
Output-space shift diagnosis signals.

Computes signals from detector outputs (predictions) to detect behavioral drift:
- Confidence distribution statistics
- Detection count drift
- Entropy and margin summaries
- NMS conflict proxy
- Box geometry statistics (area, aspect ratio)
- Temporal detection consistency (box jitter between adjacent frames)
- Calibration proxies

Usage:
    from sdmr.models.detector_wrappers.base_detector import DetectionResult
    signals = compute_output_signals(current_result)
    signals = compute_output_signals_temporal(current_result, prev_result)
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


def _safe_mean(arr, default=0.0):
    """Safe mean for potentially empty arrays."""
    if len(arr) == 0:
        return default
    return float(np.mean(arr))


def _safe_std(arr, default=0.0):
    if len(arr) == 0:
        return default
    return float(np.std(arr))


def confidence_stats(scores: np.ndarray) -> dict[str, float]:
    """
    Compute statistics of detection confidence scores.

    Args:
        scores: 1D array of confidence scores [0, 1].

    Returns:
        confidence_mean, confidence_std, confidence_max, confidence_min,
        confidence_median, confidence_q25, confidence_q75
    """
    if len(scores) == 0:
        return {
            "confidence_mean": 0.0,
            "confidence_std": 0.0,
            "confidence_max": 0.0,
            "confidence_min": 0.0,
            "confidence_median": 0.0,
            "confidence_q25": 0.0,
            "confidence_q75": 0.0,
        }

    return {
        "confidence_mean": float(np.mean(scores)),
        "confidence_std": float(np.std(scores)),
        "confidence_max": float(np.max(scores)),
        "confidence_min": float(np.min(scores)),
        "confidence_median": float(np.median(scores)),
        "confidence_q25": float(np.percentile(scores, 25)),
        "confidence_q75": float(np.percentile(scores, 75)),
    }


def detection_count_features(
    num_detections: int,
    expected_count: Optional[float] = None,
) -> dict[str, float]:
    """
    Detection count features.

    Args:
        num_detections: Number of detections in current frame.
        expected_count: Expected count from source statistics.

    Returns:
        detection_count, detection_count_delta (if expected given)
    """
    features = {"detection_count": float(num_detections)}
    if expected_count is not None:
        features["detection_count_delta"] = float(num_detections - expected_count)
        features["detection_count_ratio"] = float(num_detections / (expected_count + 1e-8))
    return features


def entropy_and_margin(scores: np.ndarray) -> dict[str, float]:
    """
    Compute entropy and margin of confidence scores.

    These are useful proxies for model uncertainty under shift.

    Returns:
        score_entropy: Entropy of normalized score distribution
        score_margin_mean: Mean margin (distance between score and 0.5)
        low_confidence_ratio: Fraction of detections with conf < 0.5
    """
    if len(scores) == 0:
        return {
            "score_entropy": 0.0,
            "score_margin_mean": 0.0,
            "low_confidence_ratio": 0.0,
        }

    # Entropy of the score distribution (binned)
    hist, _ = np.histogram(scores, bins=20, range=(0.0, 1.0), density=True)
    hist = hist + 1e-10
    hist = hist / hist.sum()
    entropy = -float(np.sum(hist * np.log2(hist)))

    # Margin: how far scores are from 0.5
    margins = np.abs(scores - 0.5)
    margin_mean = float(np.mean(margins))

    # Low confidence ratio
    low_conf = float(np.mean(scores < 0.5))

    return {
        "score_entropy": entropy,
        "score_margin_mean": margin_mean,
        "low_confidence_ratio": low_conf,
    }


def nms_conflict_proxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Estimate NMS conflict level — how many near-duplicate detections exist.
    High conflict suggests model is less decisive (possible shift effect).

    Args:
        boxes: [N, 4] boxes in xyxy format.
        scores: [N] confidence scores.
        iou_threshold: IoU threshold to count as conflict.

    Returns:
        nms_conflict_ratio: Fraction of box pairs with IoU > threshold
        nms_max_overlap: Maximum pairwise IoU
    """
    n = len(boxes)
    if n < 2:
        return {
            "nms_conflict_ratio": 0.0,
            "nms_max_overlap": 0.0,
        }

    # Compute pairwise IoU for a subset (cap at 50 to keep fast)
    if n > 50:
        idx = np.argsort(-scores)[:50]
        boxes = boxes[idx]
        n = 50

    # IoU computation
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    conflicts = 0
    total_pairs = 0
    max_iou = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            ix1 = max(x1[i], x1[j])
            iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j])
            iy2 = min(y2[i], y2[j])

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            union = areas[i] + areas[j] - inter
            iou = inter / (union + 1e-8)

            if iou > iou_threshold:
                conflicts += 1
            if iou > max_iou:
                max_iou = iou
            total_pairs += 1

    conflict_ratio = conflicts / (total_pairs + 1e-8)
    return {
        "nms_conflict_ratio": float(conflict_ratio),
        "nms_max_overlap": float(max_iou),
    }


def box_geometry_stats(boxes: np.ndarray) -> dict[str, float]:
    """
    Compute bounding box geometry statistics.

    Args:
        boxes: [N, 4] boxes in xyxy format.

    Returns:
        box_area_mean, box_area_std, box_aspect_ratio_mean, box_aspect_ratio_std,
        box_center_x_std, box_center_y_std
    """
    if len(boxes) == 0:
        return {
            "box_area_mean": 0.0,
            "box_area_std": 0.0,
            "box_aspect_ratio_mean": 0.0,
            "box_aspect_ratio_std": 0.0,
            "box_center_x_std": 0.0,
            "box_center_y_std": 0.0,
        }

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    aspect_ratios = widths / (heights + 1e-8)
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2

    return {
        "box_area_mean": float(np.mean(areas)),
        "box_area_std": float(np.std(areas)),
        "box_aspect_ratio_mean": float(np.mean(aspect_ratios)),
        "box_aspect_ratio_std": float(np.std(aspect_ratios)),
        "box_center_x_std": float(np.std(centers_x)),
        "box_center_y_std": float(np.std(centers_y)),
    }


def temporal_box_jitter(
    current_boxes: np.ndarray,
    previous_boxes: np.ndarray,
    iou_match_threshold: float = 0.3,
) -> dict[str, float]:
    """
    Measure box jitter between consecutive frames.
    Matches boxes by IoU and measures center displacement.

    High jitter under stable scenes suggests model instability (shift).

    Args:
        current_boxes: [N, 4] current frame boxes.
        previous_boxes: [M, 4] previous frame boxes.
        iou_match_threshold: Minimum IoU to consider a match.

    Returns:
        box_jitter_mean: Mean center displacement of matched boxes
        box_jitter_max: Max center displacement
        box_match_rate: Fraction of current boxes that matched
    """
    if len(current_boxes) == 0 or len(previous_boxes) == 0:
        return {
            "box_jitter_mean": 0.0,
            "box_jitter_max": 0.0,
            "box_match_rate": 0.0,
        }

    # Simple greedy matching by IoU
    n_curr = len(current_boxes)
    n_prev = len(previous_boxes)

    curr_centers = np.stack([
        (current_boxes[:, 0] + current_boxes[:, 2]) / 2,
        (current_boxes[:, 1] + current_boxes[:, 3]) / 2,
    ], axis=1)
    prev_centers = np.stack([
        (previous_boxes[:, 0] + previous_boxes[:, 2]) / 2,
        (previous_boxes[:, 1] + previous_boxes[:, 3]) / 2,
    ], axis=1)

    displacements = []
    matched = 0

    used_prev = set()
    for i in range(n_curr):
        best_iou = 0
        best_j = -1
        for j in range(n_prev):
            if j in used_prev:
                continue
            # Compute IoU
            ix1 = max(current_boxes[i, 0], previous_boxes[j, 0])
            iy1 = max(current_boxes[i, 1], previous_boxes[j, 1])
            ix2 = min(current_boxes[i, 2], previous_boxes[j, 2])
            iy2 = min(current_boxes[i, 3], previous_boxes[j, 3])
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            a1 = (current_boxes[i, 2] - current_boxes[i, 0]) * (current_boxes[i, 3] - current_boxes[i, 1])
            a2 = (previous_boxes[j, 2] - previous_boxes[j, 0]) * (previous_boxes[j, 3] - previous_boxes[j, 1])
            union = a1 + a2 - inter
            iou = inter / (union + 1e-8)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_match_threshold and best_j >= 0:
            disp = np.sqrt(np.sum((curr_centers[i] - prev_centers[best_j])**2))
            displacements.append(disp)
            used_prev.add(best_j)
            matched += 1

    return {
        "box_jitter_mean": _safe_mean(displacements),
        "box_jitter_max": float(max(displacements)) if displacements else 0.0,
        "box_match_rate": float(matched / (n_curr + 1e-8)),
    }


def calibration_proxy(scores: np.ndarray) -> dict[str, float]:
    """
    Compute calibration proxy features.
    Without ground truth, these estimate overconfidence patterns.

    Returns:
        calibration_sharpness: Concentration of scores near 0 or 1
        calibration_overconfidence: Fraction of scores > 0.9
    """
    if len(scores) == 0:
        return {
            "calibration_sharpness": 0.0,
            "calibration_overconfidence": 0.0,
        }

    # Sharpness: how peaked the score distribution is
    sharpness = float(np.mean(np.abs(scores - 0.5)) / 0.5)

    # Overconfidence proxy
    overconf = float(np.mean(scores > 0.9))

    return {
        "calibration_sharpness": sharpness,
        "calibration_overconfidence": overconf,
    }


def compute_output_signals(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    expected_count: Optional[float] = None,
) -> dict[str, float]:
    """
    Compute all output-space signals from a single frame's detections.

    Args:
        boxes: [N, 4] boxes in xyxy format (numpy).
        scores: [N] confidence scores (numpy).
        labels: [N] class labels (numpy), optional.
        expected_count: Expected detection count from source stats.

    Returns:
        Dict of signal_name -> value.
    """
    signals = {}
    signals.update(confidence_stats(scores))
    signals.update(detection_count_features(len(scores), expected_count))
    signals.update(entropy_and_margin(scores))
    signals.update(box_geometry_stats(boxes))
    signals.update(calibration_proxy(scores))

    if len(boxes) >= 2:
        signals.update(nms_conflict_proxy(boxes, scores))
    else:
        signals["nms_conflict_ratio"] = 0.0
        signals["nms_max_overlap"] = 0.0

    return signals


def compute_output_signals_temporal(
    current_boxes: np.ndarray,
    current_scores: np.ndarray,
    previous_boxes: np.ndarray,
    previous_scores: np.ndarray,
    expected_count: Optional[float] = None,
) -> dict[str, float]:
    """
    Compute all output-space signals including temporal features.

    Args:
        current_boxes, current_scores: Current frame detections.
        previous_boxes, previous_scores: Previous frame detections.
        expected_count: Expected detection count from source.

    Returns:
        Dict of all output signals including temporal box jitter.
    """
    signals = compute_output_signals(
        current_boxes, current_scores,
        expected_count=expected_count,
    )
    signals.update(temporal_box_jitter(current_boxes, previous_boxes))

    # Detection count change
    signals["detection_count_change"] = float(
        len(current_scores) - len(previous_scores)
    )

    # Confidence distribution shift (simple KL proxy)
    if len(current_scores) > 0 and len(previous_scores) > 0:
        # Histogram-based KL divergence proxy
        bins = np.linspace(0, 1, 21)
        hist_curr, _ = np.histogram(current_scores, bins=bins, density=True)
        hist_prev, _ = np.histogram(previous_scores, bins=bins, density=True)
        hist_curr = hist_curr + 1e-10
        hist_prev = hist_prev + 1e-10
        hist_curr = hist_curr / hist_curr.sum()
        hist_prev = hist_prev / hist_prev.sum()
        kl = float(np.sum(hist_curr * np.log(hist_curr / hist_prev)))
        signals["confidence_kl_divergence"] = kl
    else:
        signals["confidence_kl_divergence"] = 0.0

    return signals


# List of all output signal names for reference
OUTPUT_SIGNAL_NAMES = [
    "confidence_mean", "confidence_std", "confidence_max", "confidence_min",
    "confidence_median", "confidence_q25", "confidence_q75",
    "detection_count", "detection_count_delta", "detection_count_ratio",
    "score_entropy", "score_margin_mean", "low_confidence_ratio",
    "nms_conflict_ratio", "nms_max_overlap",
    "box_area_mean", "box_area_std",
    "box_aspect_ratio_mean", "box_aspect_ratio_std",
    "box_center_x_std", "box_center_y_std",
    "calibration_sharpness", "calibration_overconfidence",
    "box_jitter_mean", "box_jitter_max", "box_match_rate",
    "detection_count_change", "confidence_kl_divergence",
]
