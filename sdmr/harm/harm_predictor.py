"""
Harm Predictor — estimates performance degradation risk.

Given the current diagnosis signals (input drift, output instability,
feature divergence), predicts how much detection performance will drop
if no adaptation is performed.

Three implementations:
1. HistogramHarmPredictor — non-parametric, bins signal values into
   severity levels and looks up empirical AP drops.
2. GradientHarmPredictor — uses gradient norms and prediction entropy
   as proxy for model-data misalignment.
3. EnsembleHarmPredictor — combines both for robust estimates.

Theory (from Section 7.2 of paper):
  Theorem 2 (Regret Bound): For shift detector with delay d and
  adaptation with convergence in k steps:
    R(T) ≤ L_max · (d + k) · S + C_adapt · S
  where S = number of shift events, L_max = max per-frame loss.

  Harm predicts the L_max · (d + k) term to decide if C_adapt is worth paying.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod


@dataclass
class HarmEstimate:
    """Estimated harm from not adapting at this frame.

    Attributes:
        expected_ap_drop: Expected AP@50 drop if no adaptation [0, 1].
        expected_map_drop: Expected mAP@50:95 drop [0, 1].
        confidence: Confidence of the estimate [0, 1].
        horizon: Number of frames the estimate covers.
        severity_bucket: Discretized severity level (0-4).
        signal_contributions: Which signals contributed most to the estimate.
        cost_benefit_ratio: Estimated harm / adaptation cost. > 1 means adapt.
    """
    expected_ap_drop: float = 0.0
    expected_map_drop: float = 0.0
    confidence: float = 0.0
    horizon: int = 50
    severity_bucket: int = 0
    signal_contributions: dict[str, float] = field(default_factory=dict)
    cost_benefit_ratio: float = 0.0

    @property
    def should_adapt(self) -> bool:
        """Whether adaptation is recommended based on cost-benefit."""
        return self.cost_benefit_ratio > 1.0

    def to_dict(self) -> dict:
        return {
            "expected_ap_drop": round(self.expected_ap_drop, 4),
            "expected_map_drop": round(self.expected_map_drop, 4),
            "confidence": round(self.confidence, 4),
            "horizon": self.horizon,
            "severity_bucket": self.severity_bucket,
            "cost_benefit_ratio": round(self.cost_benefit_ratio, 4),
            "should_adapt": self.should_adapt,
        }


class HarmPredictor(ABC):
    """Abstract base class for harm predictors."""

    @abstractmethod
    def predict(
        self,
        signals: dict[str, float],
        adaptation_cost: float = 0.1,
    ) -> HarmEstimate:
        """
        Predict expected harm from not adapting.

        Args:
            signals: Current diagnosis signal dict.
            adaptation_cost: Cost of the cheapest useful adaptation [0,1].

        Returns:
            HarmEstimate with expected performance drop.
        """
        ...

    @abstractmethod
    def update(
        self,
        signals: dict[str, float],
        actual_ap_drop: float,
    ) -> None:
        """
        Update predictor with observed outcome.

        Args:
            signals: Signals at the time of the observation.
            actual_ap_drop: Observed AP drop at this frame.
        """
        ...


class HistogramHarmPredictor(HarmPredictor):
    """
    Non-parametric harm predictor using signal histograms.

    Bins each signal dimension into severity levels and maintains
    running estimates of AP drop for each bin combination.
    Uses a simplified approach: discretize the key "summary" signal
    (max drift across input/output/feature) into N bins and track
    empirical AP drops per bin.

    This is interpretable (good for paper) and data-efficient.
    """

    # Key signals used for harm prediction
    KEY_SIGNALS = [
        "brightness_mean", "contrast_rms", "blur_laplacian_var",
        "confidence_mean", "detection_count",
    ]

    # Empirical AP drop estimates per severity bucket (from literature/calibration)
    # Index = severity bucket (0=clean, 1=mild, 2=moderate, 3=severe, 4=catastrophic)
    DEFAULT_AP_DROPS = [0.0, 0.03, 0.10, 0.25, 0.45]
    DEFAULT_MAP_DROPS = [0.0, 0.05, 0.15, 0.30, 0.50]

    def __init__(
        self,
        n_bins: int = 5,
        ema_alpha: float = 0.1,
        horizon: int = 50,
        min_samples_per_bin: int = 10,
    ):
        """
        Args:
            n_bins: Number of severity bins.
            ema_alpha: EMA smoothing for updating empirical estimates.
            horizon: Prediction horizon in frames.
            min_samples_per_bin: Min observations before trusting empirical.
        """
        self.n_bins = n_bins
        self.ema_alpha = ema_alpha
        self.horizon = horizon
        self.min_samples = min_samples_per_bin

        # Per-bin estimates (start with priors, refine with data)
        self.ap_drop_per_bin = list(self.DEFAULT_AP_DROPS[:n_bins])
        self.map_drop_per_bin = list(self.DEFAULT_MAP_DROPS[:n_bins])
        self.bin_counts = [0] * n_bins

        # Reference statistics (clean-stream calibration)
        self._ref_means: dict[str, float] = {}
        self._ref_stds: dict[str, float] = {}
        self._calibrated = False

        # History for online calibration
        self._signal_history: list[dict[str, float]] = []
        self._calibration_window = 100

    def calibrate(self, clean_signals: list[dict[str, float]]) -> None:
        """
        Calibrate reference statistics from clean-stream data.

        Should be called during warmup phase with signals from
        un-shifted frames to establish the baseline.
        """
        if not clean_signals:
            return

        for key in self.KEY_SIGNALS:
            vals = [s.get(key, 0.0) for s in clean_signals if key in s]
            if vals:
                self._ref_means[key] = float(np.mean(vals))
                self._ref_stds[key] = float(np.std(vals)) + 1e-8

        self._calibrated = True

    def _auto_calibrate(self, signals: dict[str, float]) -> None:
        """Auto-calibrate from early frames if not explicitly calibrated."""
        self._signal_history.append(signals)
        if len(self._signal_history) >= self._calibration_window and not self._calibrated:
            self.calibrate(self._signal_history[:self._calibration_window])

    def _compute_severity(self, signals: dict[str, float]) -> float:
        """
        Compute overall severity score from signals [0, 1].

        Uses z-score deviation from reference statistics.
        """
        if not self._calibrated:
            # Fallback: use absolute signal magnitudes
            return self._compute_severity_heuristic(signals)

        z_scores = []
        for key in self.KEY_SIGNALS:
            if key in signals and key in self._ref_means:
                z = abs(signals[key] - self._ref_means[key]) / self._ref_stds[key]
                z_scores.append(z)

        if not z_scores:
            return 0.0

        # Max z-score as severity proxy
        max_z = max(z_scores)
        # Normalize to [0, 1] with soft saturation
        severity = 1 - np.exp(-max_z / 3.0)
        return float(np.clip(severity, 0, 1))

    def _compute_severity_heuristic(self, signals: dict[str, float]) -> float:
        """Heuristic severity when no calibration is available."""
        # Use confidence drop as primary indicator
        conf = signals.get("confidence_mean", 0.5)
        det_count = signals.get("detection_count", 5)
        blur = signals.get("blur_laplacian_var", 100)

        severity = 0.0
        # Low confidence → model struggling
        if conf < 0.3:
            severity = max(severity, 0.7)
        elif conf < 0.5:
            severity = max(severity, 0.4)

        # Few detections → missing objects
        if det_count < 2:
            severity = max(severity, 0.5)

        # Low blur variance → very blurry
        if blur < 20:
            severity = max(severity, 0.6)

        return float(np.clip(severity, 0, 1))

    def _severity_to_bucket(self, severity: float) -> int:
        """Map continuous severity to discrete bucket."""
        thresholds = np.linspace(0, 1, self.n_bins + 1)[1:]
        for i, t in enumerate(thresholds):
            if severity <= t:
                return i
        return self.n_bins - 1

    def _compute_signal_contributions(
        self, signals: dict[str, float]
    ) -> dict[str, float]:
        """Quantify each signal's contribution to harm estimate."""
        contribs = {}
        if not self._calibrated:
            return contribs

        for key in self.KEY_SIGNALS:
            if key in signals and key in self._ref_means:
                z = abs(signals[key] - self._ref_means[key]) / self._ref_stds[key]
                contribs[key] = float(np.clip(z / 5.0, 0, 1))  # Normalize

        return contribs

    def predict(
        self,
        signals: dict[str, float],
        adaptation_cost: float = 0.1,
    ) -> HarmEstimate:
        """Predict harm from current signals."""
        self._auto_calibrate(signals)

        severity = self._compute_severity(signals)
        bucket = self._severity_to_bucket(severity)

        # Look up empirical AP drop for this bucket
        ap_drop = self.ap_drop_per_bin[bucket]
        map_drop = self.map_drop_per_bin[bucket]

        # Confidence based on data availability
        n = self.bin_counts[bucket]
        confidence = min(1.0, n / self.min_samples) if n > 0 else 0.3

        # If not calibrated, lower confidence
        if not self._calibrated:
            confidence *= 0.5

        # Cost-benefit: harm (AP drop * horizon) vs cost
        effective_harm = ap_drop * self.horizon
        cbr = effective_harm / max(adaptation_cost, 1e-6)

        signal_contribs = self._compute_signal_contributions(signals)

        return HarmEstimate(
            expected_ap_drop=ap_drop,
            expected_map_drop=map_drop,
            confidence=confidence,
            horizon=self.horizon,
            severity_bucket=bucket,
            signal_contributions=signal_contribs,
            cost_benefit_ratio=cbr,
        )

    def update(
        self,
        signals: dict[str, float],
        actual_ap_drop: float,
    ) -> None:
        """Update empirical estimates with observed outcome."""
        severity = self._compute_severity(signals)
        bucket = self._severity_to_bucket(severity)

        n = self.bin_counts[bucket]
        alpha = self.ema_alpha if n >= self.min_samples else 0.5

        self.ap_drop_per_bin[bucket] = (
            (1 - alpha) * self.ap_drop_per_bin[bucket] +
            alpha * actual_ap_drop
        )

        self.bin_counts[bucket] += 1

    def get_stats(self) -> dict:
        """Return predictor statistics."""
        return {
            "n_bins": self.n_bins,
            "calibrated": self._calibrated,
            "bin_counts": list(self.bin_counts),
            "ap_drops": [round(d, 4) for d in self.ap_drop_per_bin],
            "total_observations": sum(self.bin_counts),
        }


class GradientHarmPredictor(HarmPredictor):
    """
    Gradient-based harm predictor.

    Uses model-side signals (prediction entropy, gradient norms)
    to estimate harm. More informed than signal-only but requires
    model access.

    Key insight: High entropy on test inputs indicates model-data
    misalignment ≈ high harm from not adapting.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        grad_norm_threshold: float = 1.0,
        calibration_window: int = 50,
        horizon: int = 50,
    ):
        self.entropy_thresh = entropy_threshold
        self.grad_norm_thresh = grad_norm_threshold
        self.horizon = horizon

        # Running statistics for normalization
        self._entropy_history: list[float] = []
        self._grad_norm_history: list[float] = []
        self._cal_window = calibration_window
        self._mean_entropy = 0.0
        self._std_entropy = 1.0
        self._mean_grad = 0.0
        self._std_grad = 1.0

    def _update_stats(self, entropy: float, grad_norm: float) -> None:
        """Update running normalization statistics."""
        self._entropy_history.append(entropy)
        self._grad_norm_history.append(grad_norm)

        if len(self._entropy_history) >= self._cal_window:
            recent = self._entropy_history[-self._cal_window:]
            self._mean_entropy = float(np.mean(recent))
            self._std_entropy = float(np.std(recent)) + 1e-8
            recent_g = self._grad_norm_history[-self._cal_window:]
            self._mean_grad = float(np.mean(recent_g))
            self._std_grad = float(np.std(recent_g)) + 1e-8

    def predict(
        self,
        signals: dict[str, float],
        adaptation_cost: float = 0.1,
    ) -> HarmEstimate:
        """
        Predict harm from gradient/entropy signals.

        Expected keys in signals:
        - 'prediction_entropy' or 'confidence_mean' (inverted)
        - 'grad_norm' (if available from model)
        """
        # Extract entropy from signals
        if "prediction_entropy" in signals:
            entropy = signals["prediction_entropy"]
        elif "confidence_mean" in signals:
            # Entropy proxy: lower confidence ≈ higher entropy
            conf = max(signals["confidence_mean"], 1e-6)
            entropy = -np.log(conf)
        else:
            entropy = 0.0

        # Gradient norm (if available)
        grad_norm = signals.get("grad_norm", 0.0)

        self._update_stats(entropy, grad_norm)

        # Compute normalized deviations
        z_entropy = (entropy - self._mean_entropy) / self._std_entropy
        z_grad = (grad_norm - self._mean_grad) / self._std_grad

        # Combined severity
        severity = float(np.clip(
            0.4 * (1 - np.exp(-max(z_entropy, 0) / 2)) +
            0.6 * (1 - np.exp(-max(z_grad, 0) / 2)),
            0, 1
        ))

        # Map to expected AP drop (learned from calibration or prior)
        ap_drop = severity * 0.5  # Linear model as prior
        map_drop = severity * 0.6

        # Cost-benefit
        effective_harm = ap_drop * self.horizon
        cbr = effective_harm / max(adaptation_cost, 1e-6)

        bucket = min(4, int(severity * 5))

        return HarmEstimate(
            expected_ap_drop=ap_drop,
            expected_map_drop=map_drop,
            confidence=min(0.8, len(self._entropy_history) / self._cal_window),
            horizon=self.horizon,
            severity_bucket=bucket,
            signal_contributions={"entropy": entropy, "grad_norm": grad_norm},
            cost_benefit_ratio=cbr,
        )

    def update(
        self,
        signals: dict[str, float],
        actual_ap_drop: float,
    ) -> None:
        """Update predictor with observed AP drop."""
        # Could adjust the linear mapping coefficient here
        # For now, just track for analysis
        pass


class EnsembleHarmPredictor(HarmPredictor):
    """
    Ensemble of histogram and gradient harm predictors.

    Combines both for robust estimates: histogram predictor uses
    domain-level signals (brightness, contrast), gradient predictor
    uses model-level signals (entropy, gradients).

    Weighting: α · histogram + (1-α) · gradient
    where α is learned from calibration data.
    """

    def __init__(
        self,
        histogram_weight: float = 0.6,
        horizon: int = 50,
        **kwargs,
    ):
        self.alpha = histogram_weight
        self.histogram = HistogramHarmPredictor(horizon=horizon, **kwargs)
        self.gradient = GradientHarmPredictor(horizon=horizon)

    def calibrate(self, clean_signals: list[dict[str, float]]) -> None:
        """Calibrate the histogram predictor."""
        self.histogram.calibrate(clean_signals)

    def predict(
        self,
        signals: dict[str, float],
        adaptation_cost: float = 0.1,
    ) -> HarmEstimate:
        """Ensemble prediction."""
        h_pred = self.histogram.predict(signals, adaptation_cost)
        g_pred = self.gradient.predict(signals, adaptation_cost)

        # Weighted combination
        ap_drop = self.alpha * h_pred.expected_ap_drop + (1 - self.alpha) * g_pred.expected_ap_drop
        map_drop = self.alpha * h_pred.expected_map_drop + (1 - self.alpha) * g_pred.expected_map_drop

        # Use higher confidence estimator
        confidence = max(h_pred.confidence, g_pred.confidence)

        # Merge contributions
        contribs = dict(h_pred.signal_contributions)
        contribs.update(g_pred.signal_contributions)

        # Cost-benefit
        effective_harm = ap_drop * self.histogram.horizon
        cbr = effective_harm / max(adaptation_cost, 1e-6)

        # Bucket from weighted severity
        weighted_bucket = self.alpha * h_pred.severity_bucket + (1 - self.alpha) * g_pred.severity_bucket

        return HarmEstimate(
            expected_ap_drop=ap_drop,
            expected_map_drop=map_drop,
            confidence=confidence,
            horizon=self.histogram.horizon,
            severity_bucket=int(round(weighted_bucket)),
            signal_contributions=contribs,
            cost_benefit_ratio=cbr,
        )

    def update(
        self,
        signals: dict[str, float],
        actual_ap_drop: float,
    ) -> None:
        """Update both sub-predictors."""
        self.histogram.update(signals, actual_ap_drop)
        self.gradient.update(signals, actual_ap_drop)

    def get_stats(self) -> dict:
        return {
            "alpha": self.alpha,
            "histogram": self.histogram.get_stats(),
        }
