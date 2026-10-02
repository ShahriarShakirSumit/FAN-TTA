"""Tests for evaluation metrics."""

import numpy as np
import pytest

from sdmr.evaluation.metrics import (
    compute_iou,
    compute_iou_matrix,
    match_predictions_to_gt,
    compute_ap_at_iou,
    compute_frame_metrics,
    StreamEvaluator,
)


class TestIoU:
    def test_perfect_overlap(self):
        box = np.array([0, 0, 10, 10])
        assert compute_iou(box, box) == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert compute_iou(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 5, 15, 15])
        # Intersection: 5x5=25, Union: 100+100-25=175
        assert compute_iou(a, b) == pytest.approx(25 / 175, abs=1e-5)


class TestIoUMatrix:
    def test_shape(self):
        pred = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=np.float32)
        iou = compute_iou_matrix(pred, gt)
        assert iou.shape == (2, 3)

    def test_empty(self):
        pred = np.zeros((0, 4), dtype=np.float32)
        gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
        iou = compute_iou_matrix(pred, gt)
        assert iou.shape == (0, 1)


class TestMatching:
    def test_perfect_match(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
        tp, fp, fn = match_predictions_to_gt(boxes, gt, scores)
        assert tp == 1
        assert fp == 0
        assert fn == 0

    def test_no_predictions(self):
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
        tp, fp, fn = match_predictions_to_gt(boxes, gt, scores)
        assert tp == 0
        assert fp == 0
        assert fn == 1

    def test_false_positive(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
        tp, fp, fn = match_predictions_to_gt(boxes, gt, scores)
        assert tp == 1
        assert fp == 1
        assert fn == 0


class TestFrameMetrics:
    def test_basic(self):
        pred_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        pred_scores = np.array([0.9], dtype=np.float32)
        gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        metrics = compute_frame_metrics(pred_boxes, pred_scores, gt_boxes)
        assert metrics.precision == pytest.approx(1.0, abs=0.1)
        assert metrics.recall == pytest.approx(1.0, abs=0.1)
        assert metrics.ap50 > 0.5


class TestStreamEvaluator:
    def test_record_and_aggregate(self):
        evaluator = StreamEvaluator()
        for i in range(10):
            pred = np.array([[0, 0, 10, 10]], dtype=np.float32)
            scores = np.array([0.9], dtype=np.float32)
            gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
            evaluator.record_frame(
                frame_idx=i,
                pred_boxes=pred,
                pred_scores=scores,
                gt_boxes=gt,
            )
        agg = evaluator.get_aggregate_metrics()
        assert "mean_ap50" in agg
        assert agg["total_frames"] == 10

    def test_export(self):
        evaluator = StreamEvaluator()
        pred = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
        evaluator.record_frame(0, pred, scores, gt)
        records = evaluator.export_records()
        assert len(records) == 1
        assert "frame_idx" in records[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
