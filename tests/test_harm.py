"""Tests for the harm predictor module."""

import pytest
import numpy as np
from sdmr.harm import (
    HarmPredictor,
    HarmEstimate,
    HistogramHarmPredictor,
    GradientHarmPredictor,
    EnsembleHarmPredictor,
)


# =====================================================================
# HarmEstimate tests
# =====================================================================

class TestHarmEstimate:
    def test_default(self):
        h = HarmEstimate()
        assert h.expected_ap_drop == 0.0
        assert h.should_adapt is False  # CBR = 0

    def test_should_adapt_true(self):
        h = HarmEstimate(expected_ap_drop=0.3, cost_benefit_ratio=2.5)
        assert h.should_adapt is True

    def test_should_adapt_false(self):
        h = HarmEstimate(expected_ap_drop=0.01, cost_benefit_ratio=0.3)
        assert h.should_adapt is False

    def test_to_dict(self):
        h = HarmEstimate(expected_ap_drop=0.1234, confidence=0.8, horizon=50)
        d = h.to_dict()
        assert "expected_ap_drop" in d
        assert d["expected_ap_drop"] == 0.1234
        assert d["horizon"] == 50
        assert "should_adapt" in d


# =====================================================================
# HistogramHarmPredictor tests
# =====================================================================

class TestHistogramHarmPredictor:
    @pytest.fixture
    def predictor(self):
        return HistogramHarmPredictor(n_bins=5, horizon=50)

    @pytest.fixture
    def calibrated_predictor(self):
        hp = HistogramHarmPredictor(n_bins=5, horizon=50)
        # Simulate clean-stream calibration
        clean = [
            {
                "brightness_mean": 0.5 + np.random.randn() * 0.02,
                "contrast_rms": 0.3 + np.random.randn() * 0.01,
                "blur_laplacian_var": 100 + np.random.randn() * 5,
                "confidence_mean": 0.7 + np.random.randn() * 0.05,
                "detection_count": 5 + np.random.randn() * 1,
            }
            for _ in range(100)
        ]
        hp.calibrate(clean)
        return hp

    def test_predict_uncalibrated(self, predictor):
        signals = {"confidence_mean": 0.3, "detection_count": 2, "blur_laplacian_var": 10}
        estimate = predictor.predict(signals)
        assert isinstance(estimate, HarmEstimate)
        assert 0 <= estimate.expected_ap_drop <= 1
        assert 0 <= estimate.severity_bucket < 5

    def test_predict_calibrated_clean(self, calibrated_predictor):
        signals = {
            "brightness_mean": 0.5,
            "contrast_rms": 0.3,
            "blur_laplacian_var": 100,
            "confidence_mean": 0.7,
            "detection_count": 5,
        }
        estimate = calibrated_predictor.predict(signals)
        assert estimate.severity_bucket <= 1  # Clean should be low severity
        assert estimate.expected_ap_drop < 0.15

    def test_predict_calibrated_severe(self, calibrated_predictor):
        signals = {
            "brightness_mean": 0.1,  # Very dark
            "contrast_rms": 0.05,    # Low contrast
            "blur_laplacian_var": 5,  # Very blurry
            "confidence_mean": 0.1,  # Low confidence
            "detection_count": 0,
        }
        estimate = calibrated_predictor.predict(signals)
        assert estimate.severity_bucket >= 2  # Severe
        assert estimate.expected_ap_drop > 0.05

    def test_update(self, calibrated_predictor):
        signals = {"brightness_mean": 0.2, "confidence_mean": 0.2}
        calibrated_predictor.update(signals, actual_ap_drop=0.3)
        stats = calibrated_predictor.get_stats()
        assert stats["total_observations"] >= 1

    def test_auto_calibrate(self, predictor):
        """Auto-calibration after enough samples."""
        for i in range(110):
            signals = {
                "brightness_mean": 0.5 + np.random.randn() * 0.02,
                "confidence_mean": 0.7 + np.random.randn() * 0.05,
            }
            predictor.predict(signals)

        assert predictor._calibrated is True

    def test_cost_benefit_ratio(self, calibrated_predictor):
        signals = {
            "brightness_mean": 0.1,
            "confidence_mean": 0.1,
        }
        estimate = calibrated_predictor.predict(signals, adaptation_cost=0.01)
        # Severe shift + low cost → high CBR
        assert estimate.cost_benefit_ratio > 0

    def test_get_stats(self, predictor):
        stats = predictor.get_stats()
        assert "n_bins" in stats
        assert "calibrated" in stats
        assert "bin_counts" in stats
        assert stats["n_bins"] == 5

    def test_signal_contributions_empty_uncalibrated(self, predictor):
        signals = {"brightness_mean": 0.3}
        estimate = predictor.predict(signals)
        # Uncalibrated → empty contributions
        assert isinstance(estimate.signal_contributions, dict)


# =====================================================================
# GradientHarmPredictor tests
# =====================================================================

class TestGradientHarmPredictor:
    @pytest.fixture
    def predictor(self):
        return GradientHarmPredictor(horizon=50)

    def test_predict_low_entropy(self, predictor):
        signals = {"confidence_mean": 0.9, "grad_norm": 0.1}
        estimate = predictor.predict(signals)
        assert estimate.expected_ap_drop < 0.3

    def test_predict_high_entropy(self, predictor):
        signals = {"confidence_mean": 0.05, "grad_norm": 5.0}
        estimate = predictor.predict(signals)
        assert estimate.expected_ap_drop > 0

    def test_predict_with_explicit_entropy(self, predictor):
        signals = {"prediction_entropy": 2.5, "grad_norm": 3.0}
        estimate = predictor.predict(signals)
        assert isinstance(estimate, HarmEstimate)

    def test_confidence_increases_with_data(self, predictor):
        estimates = []
        for i in range(60):
            signals = {"confidence_mean": 0.5 + np.random.randn() * 0.1}
            e = predictor.predict(signals)
            estimates.append(e.confidence)

        # Confidence should be higher later
        assert estimates[-1] >= estimates[0]

    def test_update_does_not_crash(self, predictor):
        signals = {"confidence_mean": 0.3}
        predictor.update(signals, actual_ap_drop=0.2)  # Should not crash


# =====================================================================
# EnsembleHarmPredictor tests
# =====================================================================

class TestEnsembleHarmPredictor:
    @pytest.fixture
    def predictor(self):
        return EnsembleHarmPredictor(histogram_weight=0.6, horizon=50)

    def test_predict(self, predictor):
        signals = {
            "brightness_mean": 0.3,
            "confidence_mean": 0.2,
            "detection_count": 1,
        }
        estimate = predictor.predict(signals)
        assert isinstance(estimate, HarmEstimate)
        assert 0 <= estimate.expected_ap_drop <= 1

    def test_weighted_combination(self):
        """Ensemble should weight components correctly."""
        hp = EnsembleHarmPredictor(histogram_weight=0.5)
        signals = {
            "brightness_mean": 0.3,
            "confidence_mean": 0.2,
            "detection_count": 1,
        }
        e = hp.predict(signals)
        assert isinstance(e.signal_contributions, dict)

    def test_calibrate(self, predictor):
        clean = [{"brightness_mean": 0.5, "confidence_mean": 0.7} for _ in range(100)]
        predictor.calibrate(clean)
        assert predictor.histogram._calibrated is True

    def test_update(self, predictor):
        signals = {"brightness_mean": 0.2, "confidence_mean": 0.3}
        predictor.update(signals, actual_ap_drop=0.25)

    def test_get_stats(self, predictor):
        stats = predictor.get_stats()
        assert "alpha" in stats
        assert "histogram" in stats


# =====================================================================
# Integration: harm-based adaptation decision
# =====================================================================

class TestHarmBasedDecision:
    """Test that harm predictions lead to sensible adaptation decisions."""

    def test_no_harm_no_adapt(self):
        hp = HistogramHarmPredictor()
        hp.calibrate([
            {"brightness_mean": 0.5, "confidence_mean": 0.7, "detection_count": 5}
            for _ in range(50)
        ])

        # Clean signal → low harm
        estimate = hp.predict(
            {"brightness_mean": 0.5, "confidence_mean": 0.7, "detection_count": 5},
            adaptation_cost=0.5,
        )
        assert not estimate.should_adapt

    def test_high_harm_should_adapt(self):
        hp = HistogramHarmPredictor()
        hp.calibrate([
            {"brightness_mean": 0.5, "confidence_mean": 0.7, "detection_count": 5}
            for _ in range(50)
        ])

        # Severe shift → high harm
        estimate = hp.predict(
            {"brightness_mean": 0.1, "confidence_mean": 0.05, "detection_count": 0},
            adaptation_cost=0.01,
        )
        assert estimate.cost_benefit_ratio > 1
        assert estimate.should_adapt
