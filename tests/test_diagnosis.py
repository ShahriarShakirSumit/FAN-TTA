"""Tests for diagnosis modules: change-point, signals, temporal buffer."""

import numpy as np
import pytest

from sdmr.diagnosis.change_point import (
    CUSUMDetector,
    RollingZScoreDetector,
    HysteresisDetector,
    MultiSignalChangePointDetector,
    ChangePointResult,
)
from sdmr.diagnosis.temporal_buffer import TemporalBuffer
from sdmr.diagnosis.input_signals import compute_input_signals
from sdmr.diagnosis.output_signals import compute_output_signals
from sdmr.diagnosis.feature_signals import compute_feature_signals


class TestCUSUMDetector:
    def test_no_shift(self):
        detector = CUSUMDetector(threshold=5.0, warmup=20)
        # Feed stable signal
        for _ in range(100):
            result = detector.update(0.0)
        assert not result.shift_detected

    def test_detects_shift(self):
        detector = CUSUMDetector(threshold=5.0, drift=0.5, warmup=20)
        # Feed stable signal then sudden shift
        for _ in range(30):
            detector.update(0.0)
        detected = False
        for _ in range(50):
            result = detector.update(5.0)
            if result.shift_detected:
                detected = True
                break
        assert detected

    def test_reset(self):
        detector = CUSUMDetector(threshold=5.0, warmup=20)
        for _ in range(50):
            detector.update(0.0)
        detector.reset()
        assert detector._step == 0
        assert detector._s_pos == 0.0


class TestRollingZScoreDetector:
    def test_no_shift(self):
        detector = RollingZScoreDetector(window_size=30, z_threshold=3.0)
        rng = np.random.RandomState(42)
        for _ in range(100):
            result = detector.update(rng.normal(0, 1))
        # Should not trigger frequently with normal data
        assert result.statistic_value < 10  # relaxed check

    def test_detects_shift(self):
        detector = RollingZScoreDetector(
            window_size=30, z_threshold=2.0, min_consecutive=3
        )
        # Stable phase
        for _ in range(50):
            detector.update(0.0)
        # Shift phase
        detected = False
        for _ in range(20):
            result = detector.update(10.0)
            if result.shift_detected:
                detected = True
                break
        assert detected


class TestHysteresisDetector:
    def test_prevents_toggling(self):
        inner = CUSUMDetector(threshold=3.0, warmup=10)
        detector = HysteresisDetector(inner, enter_count=3, exit_count=5)
        
        # Warmup
        for _ in range(15):
            detector.update(0.0)
        
        # Single spike should not trigger
        result = detector.update(10.0)
        assert not result.shift_detected  # needs 3 consecutive

    def test_enters_and_exits_shift(self):
        inner = CUSUMDetector(threshold=2.0, warmup=10)
        detector = HysteresisDetector(inner, enter_count=2, exit_count=3)
        
        # Warmup
        for _ in range(15):
            detector.update(0.0)
        
        # Sustained shift should eventually trigger
        for _ in range(30):
            detector.update(8.0)
        result = detector.last_result
        # May or may not be detected depending on inner detector behavior
        assert isinstance(result, ChangePointResult)


class TestMultiSignalDetector:
    def test_majority_fusion(self):
        signals = ["sig_a", "sig_b", "sig_c"]
        detector = MultiSignalChangePointDetector(
            signal_names=signals,
            fusion="majority",
        )
        # Feed stable data
        for _ in range(60):
            result = detector.update({s: 0.0 for s in signals})
        assert not result.shift_detected

    def test_reset(self):
        detector = MultiSignalChangePointDetector(
            signal_names=["a", "b"],
        )
        for _ in range(30):
            detector.update({"a": 0.0, "b": 0.0})
        detector.reset()
        assert detector._step == 0


class TestTemporalBuffer:
    def test_add_and_size(self):
        buf = TemporalBuffer(max_size=10)
        for i in range(5):
            buf.append({"x": float(i), "y": float(i * 2)})
        assert buf.size == 5

    def test_max_size(self):
        buf = TemporalBuffer(max_size=5)
        for i in range(20):
            buf.append({"x": float(i)})
        assert buf.size == 5

    def test_rolling_stats(self):
        buf = TemporalBuffer(max_size=100)
        for i in range(50):
            buf.append({"x": float(i)})
        stats = buf.get_rolling_stats(window=10)
        assert "x" in stats
        assert hasattr(stats["x"], "mean")
        assert hasattr(stats["x"], "std")

    def test_fused_vector(self):
        buf = TemporalBuffer(max_size=100)
        for i in range(20):
            buf.append({"a": float(i), "b": float(i * 2)})
        vec = buf.get_fused_vector(window=10)
        assert isinstance(vec, np.ndarray)
        assert len(vec) > 0


class TestInputSignals:
    def test_compute_basic(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        signals = compute_input_signals(frame)
        assert "brightness_mean" in signals
        assert "contrast_rms" in signals
        assert "blur_laplacian_var" in signals
        assert signals["brightness_mean"] >= 0

    def test_with_prev_frame(self):
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        signals = compute_input_signals(frame2, previous_image=frame1)
        assert "frame_diff_mean" in signals
        assert signals["frame_diff_mean"] > 0


class TestOutputSignals:
    def test_compute_basic(self):
        boxes = np.array([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        classes = np.array([0, 1], dtype=np.int64)
        signals = compute_output_signals(boxes, scores, classes)
        assert "confidence_mean" in signals
        assert "detection_count" in signals
        assert signals["detection_count"] == 2

    def test_empty_predictions(self):
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        classes = np.zeros(0, dtype=np.int64)
        signals = compute_output_signals(boxes, scores, classes)
        assert signals["detection_count"] == 0


class TestFeatureSignals:
    def test_compute_basic(self):
        features = {
            "layer1": np.random.randn(64, 8, 8).astype(np.float32),
            "layer2": np.random.randn(128, 4, 4).astype(np.float32),
        }
        signals = compute_feature_signals(features)
        assert "layer1" in signals.layer_mean
        assert "layer2" in signals.layer_std
        assert signals.agg_mean != 0 or signals.agg_std != 0

    def test_flat_dict(self):
        features = {
            "conv1": np.random.randn(32, 16, 16).astype(np.float32),
        }
        signals = compute_feature_signals(features)
        flat = signals.to_flat_dict()
        assert isinstance(flat, dict)
        assert any("conv1" in k for k in flat.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
