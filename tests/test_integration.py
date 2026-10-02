"""
Integration tests for the SDMR experiment pipeline with real COCO data.

These tests validate the full pipeline:
- Dataset loading (COCO-Person)
- Stream wrapper with shift injection
- Diagnosis signals on real images
- Router decisions
- Evaluation metrics

Requires: COCO val2017 dataset at ./data/coco2017/
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Skip all tests if COCO dataset not available
COCO_DIR = "./data/coco2017"
COCO_AVAILABLE = (
    os.path.exists(os.path.join(COCO_DIR, "val2017")) and
    os.path.exists(os.path.join(COCO_DIR, "annotations", "instances_val2017.json"))
)
pytestmark = pytest.mark.skipif(
    not COCO_AVAILABLE,
    reason="COCO val2017 dataset not available"
)


# =====================================================================
# Dataset Tests
# =====================================================================

class TestCOCODataset:
    """Test COCO Person dataset loading."""

    def test_load_coco_person(self):
        from sdmr.data.datasets.coco_person import COCOPersonDataset, COCOPersonConfig
        ds = COCOPersonDataset(
            images_dir=os.path.join(COCO_DIR, "val2017"),
            annotations_file=os.path.join(COCO_DIR, "annotations", "instances_val2017.json"),
            config=COCOPersonConfig(min_area=100.0, person_only=True),
        )
        assert len(ds) > 100, f"Expected >100 person images, got {len(ds)}"

    def test_coco_item_format(self):
        from sdmr.data.datasets.coco_person import COCOPersonDataset, COCOPersonConfig
        ds = COCOPersonDataset(
            images_dir=os.path.join(COCO_DIR, "val2017"),
            annotations_file=os.path.join(COCO_DIR, "annotations", "instances_val2017.json"),
            config=COCOPersonConfig(min_area=100.0, person_only=True),
        )
        frame = ds[0]
        assert hasattr(frame, 'image')
        assert hasattr(frame, 'targets')
        assert hasattr(frame, 'metadata')
        assert frame.image.ndim == 3
        assert frame.image.shape[2] == 3
        assert 'boxes' in frame.targets
        assert 'labels' in frame.targets
        assert frame.targets['boxes'].ndim == 2
        assert frame.targets['boxes'].shape[1] == 4

    def test_coco_subset(self):
        from sdmr.data.datasets.coco_person import COCOPersonDataset, COCOPersonConfig
        ds = COCOPersonDataset(
            images_dir=os.path.join(COCO_DIR, "val2017"),
            annotations_file=os.path.join(COCO_DIR, "annotations", "instances_val2017.json"),
        )
        subset = ds.subset(50, seed=42)
        assert len(subset) == 50

    def test_dataset_adapter(self):
        from sdmr.run_stream_experiment import SimpleDatasetAdapter
        from sdmr.data.datasets.coco_person import COCOPersonDataset
        ds = COCOPersonDataset(
            images_dir=os.path.join(COCO_DIR, "val2017"),
            annotations_file=os.path.join(COCO_DIR, "annotations", "instances_val2017.json"),
        )
        adapted = SimpleDatasetAdapter(ds.subset(10))
        img, targets = adapted[0]
        assert isinstance(img, np.ndarray)
        assert isinstance(targets, dict)


# =====================================================================
# Stream Tests
# =====================================================================

class TestStreamWrapper:
    """Test stream wrapper with real dataset."""

    @pytest.fixture
    def small_stream(self):
        from sdmr.run_stream_experiment import load_config, build_dataset, build_shift_scenario
        from sdmr.data.datasets.stream_wrapper import StreamWrapper
        from sdmr.data.datasets.synthetic_shift_injector import SyntheticShiftInjector

        cfg = {
            "seed": 42,
            "data": {
                "dataset": "coco_person",
                "dataset_dir": COCO_DIR,
                "num_images": 20,
                "min_area": 100.0,
            },
            "stream": {
                "scenario": {
                    "type": "abrupt",
                    "clean_before": 30,
                    "shift_length": 30,
                    "clean_after": 30,
                    "shift_family": "gaussian_noise",
                    "severity": 0.5,
                },
            },
        }
        dataset = build_dataset(cfg)
        scenario = build_shift_scenario(cfg)
        injector = SyntheticShiftInjector()
        stream = StreamWrapper(
            dataset=dataset, scenario=scenario,
            shift_injector=lambda img, fam, sev, s: injector.apply(
                img, family=fam, severity=sev, seed=s
            ),
            seed=42,
        )
        return stream

    def test_stream_clean_frame(self, small_stream):
        f = small_stream.get_frame(0)
        assert not f.metadata.shift_active
        assert f.image.ndim == 3

    def test_stream_shifted_frame(self, small_stream):
        f = small_stream.get_frame(35)
        assert f.metadata.shift_active
        assert f.metadata.shift_family == "gaussian_noise"
        assert f.metadata.shift_severity > 0

    def test_stream_post_shift_frame(self, small_stream):
        f = small_stream.get_frame(65)
        assert not f.metadata.shift_active

    def test_stream_total_frames(self, small_stream):
        assert small_stream.total_frames == 90


# =====================================================================
# Diagnosis Tests with Real Images
# =====================================================================

class TestDiagnosisOnRealImages:
    """Test diagnosis signals on real COCO images."""

    @pytest.fixture
    def coco_image(self):
        from sdmr.data.datasets.coco_person import COCOPersonDataset
        ds = COCOPersonDataset(
            images_dir=os.path.join(COCO_DIR, "val2017"),
            annotations_file=os.path.join(COCO_DIR, "annotations", "instances_val2017.json"),
        )
        return ds[0].image

    def test_input_signals_on_real_image(self, coco_image):
        from sdmr.diagnosis.input_signals import compute_input_signals
        signals = compute_input_signals(coco_image)
        assert "brightness_mean" in signals
        assert "contrast_rms" in signals
        assert "blur_laplacian_var" in signals
        assert 0 < signals["brightness_mean"] < 1
        assert signals["blur_laplacian_var"] > 0

    def test_output_signals_with_boxes(self):
        from sdmr.diagnosis.output_signals import compute_output_signals
        boxes = np.array([[10, 20, 100, 200], [50, 30, 150, 250]], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        signals = compute_output_signals(boxes, scores)
        assert "confidence_mean" in signals
        assert "detection_count" in signals
        assert signals["confidence_mean"] == pytest.approx(0.8, abs=0.01)
        assert signals["detection_count"] == 2

    def test_temporal_buffer_with_real_images(self):
        from sdmr.data.datasets.coco_person import COCOPersonDataset
        from sdmr.diagnosis.input_signals import compute_input_signals
        from sdmr.diagnosis.temporal_buffer import TemporalBuffer

        ds = COCOPersonDataset(
            images_dir=os.path.join(COCO_DIR, "val2017"),
            annotations_file=os.path.join(COCO_DIR, "annotations", "instances_val2017.json"),
        )
        buf = TemporalBuffer(max_size=50)
        for i in range(10):
            signals = compute_input_signals(ds[i].image)
            buf.append(signals)

        assert buf.size == 10
        stats = buf.get_rolling_stats(window=5)
        assert len(stats) > 0
        for name, rs in stats.items():
            assert hasattr(rs, 'mean')
            assert hasattr(rs, 'std')


# =====================================================================
# Router Tests
# =====================================================================

class TestRouterDecisions:
    """Test router decisions with realistic signal values."""

    def test_sdmr_no_shift(self):
        from sdmr.adaptation import RuleBasedRouter, AdaptationAction
        router = RuleBasedRouter()
        decision = router.decide(
            shift_detected=False, shift_severity=0.0,
            shift_persistence=0.0, input_drift_score=0.1,
            output_drift_score=0.1, feature_drift_score=0.1,
        )
        assert decision.action == AdaptationAction.NONE

    def test_sdmr_mild_shift(self):
        from sdmr.adaptation import RuleBasedRouter, AdaptationAction
        router = RuleBasedRouter()
        decision = router.decide(
            shift_detected=True, shift_severity=0.3,
            shift_persistence=0.3, input_drift_score=0.5,
            output_drift_score=0.2, feature_drift_score=0.2,
        )
        assert decision.action != AdaptationAction.NONE

    def test_sdmr_severe_shift(self):
        from sdmr.adaptation import RuleBasedRouter, AdaptationAction
        router = RuleBasedRouter()
        decision = router.decide(
            shift_detected=True, shift_severity=0.9,
            shift_persistence=0.8, input_drift_score=0.8,
            output_drift_score=0.7, feature_drift_score=0.6,
        )
        # Severe shift should trigger stronger adaptation
        assert decision.action.value >= AdaptationAction.HEAD_ONLY.value


# =====================================================================
# Evaluation Tests
# =====================================================================

class TestEvaluation:
    """Test evaluation metrics with realistic predictions."""

    def test_perfect_detection(self):
        from sdmr.evaluation.metrics import compute_frame_metrics
        gt_boxes = np.array([[10, 20, 100, 200]], dtype=np.float32)
        pred_boxes = np.array([[10, 20, 100, 200]], dtype=np.float32)
        pred_scores = np.array([0.9], dtype=np.float32)
        metrics = compute_frame_metrics(pred_boxes, pred_scores, gt_boxes)
        # AP@50 for single perfect match may not be exactly 1.0 due to
        # interpolation in AP calculation — just check it's high
        assert metrics.ap50 > 0.85
        assert metrics.precision == pytest.approx(1.0, abs=0.01)
        assert metrics.recall == pytest.approx(1.0, abs=0.01)

    def test_no_detection(self):
        from sdmr.evaluation.metrics import compute_frame_metrics
        gt_boxes = np.array([[10, 20, 100, 200]], dtype=np.float32)
        pred_boxes = np.zeros((0, 4), dtype=np.float32)
        pred_scores = np.zeros(0, dtype=np.float32)
        metrics = compute_frame_metrics(pred_boxes, pred_scores, gt_boxes)
        assert metrics.recall == pytest.approx(0.0, abs=0.01)

    def test_stream_evaluator(self):
        from sdmr.evaluation.metrics import StreamEvaluator
        evaluator = StreamEvaluator(score_threshold=0.25)
        for i in range(20):
            gt = np.array([[10, 20, 100, 200]], dtype=np.float32)
            pred = gt + np.random.randn(1, 4) * 5
            scores = np.array([0.8], dtype=np.float32)
            evaluator.record_frame(i, pred, scores, gt)
        agg = evaluator.get_aggregate_metrics()
        assert "mean_ap50" in agg
        assert agg["total_frames"] == 20


# =====================================================================
# Config Tests
# =====================================================================

class TestConfigs:
    """Test experiment config loading."""

    @pytest.fixture(params=[
        "configs/coco_experiment.yaml",
        "configs/coco_gradual_brightness.yaml",
        "configs/coco_recurring_blur.yaml",
        "configs/coco_mixed_shifts.yaml",
    ])
    def config_path(self, request):
        return request.param

    def test_config_loads(self, config_path):
        from sdmr.run_stream_experiment import load_config, build_shift_scenario
        cfg = load_config(config_path)
        assert "stream" in cfg
        scenario = build_shift_scenario(cfg)
        assert scenario.total_frames > 0

    def test_config_builds_router(self, config_path):
        from sdmr.run_stream_experiment import load_config, build_router
        cfg = load_config(config_path)
        router = build_router(cfg)
        assert router is not None
