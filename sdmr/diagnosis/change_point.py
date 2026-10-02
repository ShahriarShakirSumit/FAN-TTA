"""
Change-point detection for streaming diagnosis signals.

Implements multiple change-point detection methods:
- CUSUM (Cumulative Sum) detector
- Rolling z-score threshold detector
- Hysteresis-based detector (avoids rapid toggling)

All detectors operate on 1D signal sequences and output:
- shift_event flag
- estimated onset index
- severity score
- persistence score (transient vs persistent)
- confidence of detection

Usage:
    detector = CUSUMDetector(threshold=5.0)
    for value in signal_stream:
        result = detector.update(value)
        if result.shift_detected:
            print(f"Shift at step {result.onset_index}, severity={result.severity}")
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod


@dataclass
class ChangePointResult:
    """Result from a change-point detection update."""
    shift_detected: bool = False
    onset_index: Optional[int] = None
    severity: float = 0.0
    persistence: float = 0.0   # 0 = transient, 1 = persistent
    confidence: float = 0.0
    statistic_value: float = 0.0
    threshold: float = 0.0
    detector_name: str = ""

    def to_dict(self) -> dict:
        return {
            "shift_detected": self.shift_detected,
            "onset_index": self.onset_index,
            "severity": self.severity,
            "persistence": self.persistence,
            "confidence": self.confidence,
            "statistic_value": self.statistic_value,
            "threshold": self.threshold,
            "detector_name": self.detector_name,
        }


class BaseChangePointDetector(ABC):
    """Abstract base class for change-point detectors."""

    def __init__(self, name: str = "base"):
        self.name = name
        self._step = 0
        self._last_result = ChangePointResult(detector_name=name)

    @abstractmethod
    def update(self, value: float) -> ChangePointResult:
        """Process a new value and return detection result."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        ...

    @property
    def last_result(self) -> ChangePointResult:
        return self._last_result


class CUSUMDetector(BaseChangePointDetector):
    """
    CUSUM (Cumulative Sum) change-point detector.

    Detects shifts in the mean of a signal by accumulating
    deviations from a reference level. Signals both positive
    and negative shifts.

    Args:
        threshold: Detection threshold for the CUSUM statistic.
        drift: Minimum shift magnitude to detect (slack parameter).
        reference_mean: Expected mean under no-shift. If None, estimated online.
        reference_std: Expected std under no-shift. If None, estimated online.
        warmup: Number of initial samples for estimating reference stats.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.5,
        reference_mean: Optional[float] = None,
        reference_std: Optional[float] = None,
        warmup: int = 50,
    ):
        super().__init__(name="cusum")
        self.threshold = threshold
        self.drift = drift
        self.warmup = warmup

        self._ref_mean = reference_mean
        self._ref_std = reference_std
        self._s_pos = 0.0      # positive CUSUM
        self._s_neg = 0.0      # negative CUSUM
        self._warmup_values: list[float] = []
        self._onset_pos: Optional[int] = None
        self._onset_neg: Optional[int] = None
        self._consecutive_detections = 0

    def update(self, value: float) -> ChangePointResult:
        self._step += 1

        # Warmup phase: collect stats
        if self._ref_mean is None:
            self._warmup_values.append(value)
            if len(self._warmup_values) >= self.warmup:
                self._ref_mean = float(np.mean(self._warmup_values))
                self._ref_std = float(np.std(self._warmup_values)) + 1e-8
            result = ChangePointResult(
                shift_detected=False,
                detector_name=self.name,
            )
            self._last_result = result
            return result

        # Normalize
        z = (value - self._ref_mean) / self._ref_std

        # Update CUSUM statistics
        self._s_pos = max(0, self._s_pos + z - self.drift)
        self._s_neg = max(0, self._s_neg - z - self.drift)

        # Track onset
        if self._s_pos > 0 and self._onset_pos is None:
            self._onset_pos = self._step
        if self._s_pos == 0:
            self._onset_pos = None

        if self._s_neg > 0 and self._onset_neg is None:
            self._onset_neg = self._step
        if self._s_neg == 0:
            self._onset_neg = None

        # Detection
        detected = self._s_pos > self.threshold or self._s_neg > self.threshold
        stat_value = max(self._s_pos, self._s_neg)

        if detected:
            self._consecutive_detections += 1
            onset = self._onset_pos if self._s_pos > self._s_neg else self._onset_neg
            severity = min(1.0, stat_value / (self.threshold * 3))
            persistence = min(1.0, self._consecutive_detections / 10)
            confidence = min(1.0, stat_value / self.threshold)
        else:
            self._consecutive_detections = 0
            onset = None
            severity = 0.0
            persistence = 0.0
            confidence = 0.0

        result = ChangePointResult(
            shift_detected=detected,
            onset_index=onset,
            severity=severity,
            persistence=persistence,
            confidence=confidence,
            statistic_value=stat_value,
            threshold=self.threshold,
            detector_name=self.name,
        )
        self._last_result = result
        return result

    def reset(self) -> None:
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._onset_pos = None
        self._onset_neg = None
        self._consecutive_detections = 0
        self._step = 0
        if self._ref_mean is None:
            self._warmup_values.clear()


class RollingZScoreDetector(BaseChangePointDetector):
    """
    Rolling z-score based change-point detector.

    Computes z-scores against a rolling window and detects
    when z-score exceeds a threshold.

    Args:
        window_size: Size of the rolling window for baseline stats.
        z_threshold: Z-score threshold for detection.
        min_consecutive: Minimum consecutive detections to confirm shift.
    """

    def __init__(
        self,
        window_size: int = 50,
        z_threshold: float = 3.0,
        min_consecutive: int = 3,
    ):
        super().__init__(name="rolling_zscore")
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_consecutive = min_consecutive

        self._buffer: list[float] = []
        self._consecutive_count = 0
        self._onset_step: Optional[int] = None

    def update(self, value: float) -> ChangePointResult:
        self._step += 1
        self._buffer.append(value)

        # Need enough history
        if len(self._buffer) < self.window_size + 1:
            result = ChangePointResult(
                shift_detected=False,
                detector_name=self.name,
            )
            self._last_result = result
            return result

        # Compute z-score against trailing window (excluding most recent values)
        baseline = self._buffer[-(self.window_size + 10):-(10)]
        if len(baseline) < 10:
            baseline = self._buffer[:self.window_size]

        mean = float(np.mean(baseline))
        std = float(np.std(baseline)) + 1e-8
        z = abs(value - mean) / std

        detected_raw = z > self.z_threshold

        if detected_raw:
            self._consecutive_count += 1
            if self._onset_step is None:
                self._onset_step = self._step
        else:
            self._consecutive_count = 0
            self._onset_step = None

        confirmed = self._consecutive_count >= self.min_consecutive

        severity = min(1.0, z / (self.z_threshold * 3)) if confirmed else 0.0
        persistence = min(1.0, self._consecutive_count / 20) if confirmed else 0.0
        confidence = min(1.0, z / self.z_threshold) if z > 0 else 0.0

        result = ChangePointResult(
            shift_detected=confirmed,
            onset_index=self._onset_step,
            severity=severity,
            persistence=persistence,
            confidence=confidence,
            statistic_value=z,
            threshold=self.z_threshold,
            detector_name=self.name,
        )
        self._last_result = result
        return result

    def reset(self) -> None:
        self._buffer.clear()
        self._consecutive_count = 0
        self._onset_step = None
        self._step = 0


class HysteresisDetector(BaseChangePointDetector):
    """
    Hysteresis-based change-point detector.

    Uses high/low thresholds to avoid rapid toggling:
    - Enters "shift" state when signal exceeds high threshold
    - Exits "shift" state only when signal drops below low threshold

    Args:
        high_threshold: Upper threshold to enter shift state.
        low_threshold: Lower threshold to exit shift state.
        stat_fn: Function to compute the statistic from the wrapped detector.
    """

    def __init__(
        self,
        inner_detector: BaseChangePointDetector,
        enter_count: int = 3,
        exit_count: int = 5,
    ):
        super().__init__(name=f"hysteresis_{inner_detector.name}")
        self.inner = inner_detector
        self.enter_count = enter_count
        self.exit_count = exit_count

        self._in_shift = False
        self._enter_counter = 0
        self._exit_counter = 0
        self._onset_step: Optional[int] = None

    def update(self, value: float) -> ChangePointResult:
        self._step += 1
        inner_result = self.inner.update(value)

        if not self._in_shift:
            # Not in shift state — count consecutive inner detections
            if inner_result.shift_detected:
                self._enter_counter += 1
                if self._enter_counter >= self.enter_count:
                    self._in_shift = True
                    self._onset_step = inner_result.onset_index or self._step
                    self._exit_counter = 0
            else:
                self._enter_counter = 0
        else:
            # In shift state — count consecutive non-detections to exit
            if not inner_result.shift_detected:
                self._exit_counter += 1
                if self._exit_counter >= self.exit_count:
                    self._in_shift = False
                    self._onset_step = None
                    self._enter_counter = 0
            else:
                self._exit_counter = 0

        result = ChangePointResult(
            shift_detected=self._in_shift,
            onset_index=self._onset_step,
            severity=inner_result.severity if self._in_shift else 0.0,
            persistence=inner_result.persistence if self._in_shift else 0.0,
            confidence=inner_result.confidence,
            statistic_value=inner_result.statistic_value,
            threshold=inner_result.threshold,
            detector_name=self.name,
        )
        self._last_result = result
        return result

    def reset(self) -> None:
        self.inner.reset()
        self._in_shift = False
        self._enter_counter = 0
        self._exit_counter = 0
        self._onset_step = None
        self._step = 0


class MultiSignalChangePointDetector:
    """
    Fused change-point detector over multiple signals.

    Runs independent detectors on each signal and fuses their outputs
    using a voting / aggregation scheme.

    Args:
        signal_names: List of signal names to monitor.
        detector_factory: Factory function returning a detector for each signal.
        fusion: Fusion method ("any", "majority", "weighted").
        weights: Optional per-signal weights for weighted fusion.
    """

    def __init__(
        self,
        signal_names: list[str],
        detector_factory=None,
        fusion: str = "majority",
        weights: Optional[dict[str, float]] = None,
    ):
        self.signal_names = signal_names
        self.fusion = fusion
        self.weights = weights or {n: 1.0 for n in signal_names}

        if detector_factory is None:
            detector_factory = lambda name: HysteresisDetector(
                CUSUMDetector(threshold=5.0, drift=0.5, warmup=30)
            )

        self.detectors: dict[str, BaseChangePointDetector] = {
            name: detector_factory(name) for name in signal_names
        }
        self._step = 0

    def update(self, signals: dict[str, float]) -> ChangePointResult:
        """
        Update all detectors with new signal values and fuse results.

        Args:
            signals: Dict of signal_name -> value.

        Returns:
            Fused ChangePointResult.
        """
        self._step += 1
        results = {}

        for name in self.signal_names:
            if name in signals:
                results[name] = self.detectors[name].update(signals[name])

        if not results:
            return ChangePointResult(shift_detected=False, detector_name="multi_fused")

        # Fuse detections
        detected_signals = [
            (name, r) for name, r in results.items() if r.shift_detected
        ]

        if self.fusion == "any":
            detected = len(detected_signals) > 0
        elif self.fusion == "majority":
            detected = len(detected_signals) > len(results) / 2
        elif self.fusion == "weighted":
            total_weight = sum(self.weights.get(n, 1.0) for n in results)
            detected_weight = sum(
                self.weights.get(n, 1.0) for n, _ in detected_signals
            )
            detected = detected_weight > total_weight / 2
        else:
            detected = len(detected_signals) > 0

        # Aggregate severity, confidence, persistence
        if detected_signals:
            severity = float(np.mean([r.severity for _, r in detected_signals]))
            confidence = float(np.mean([r.confidence for _, r in detected_signals]))
            persistence = float(np.mean([r.persistence for _, r in detected_signals]))
            onset = min(
                (r.onset_index for _, r in detected_signals if r.onset_index is not None),
                default=None,
            )
        else:
            severity = 0.0
            confidence = 0.0
            persistence = 0.0
            onset = None

        return ChangePointResult(
            shift_detected=detected,
            onset_index=onset,
            severity=severity,
            persistence=persistence,
            confidence=confidence,
            statistic_value=float(len(detected_signals) / (len(results) + 1e-8)),
            threshold=0.5,
            detector_name="multi_fused",
        )

    def reset(self) -> None:
        for d in self.detectors.values():
            d.reset()
        self._step = 0
