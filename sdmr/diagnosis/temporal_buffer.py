"""
Temporal buffer for streaming diagnosis signals.

Maintains a fixed-size rolling buffer of signal values with support for:
- Rolling statistics (mean, std, z-score)
- Delta from a source anchor
- Change detection readiness
- Efficient append/evict operations

Usage:
    buffer = TemporalBuffer(max_size=100, signal_names=["brightness", "blur_score"])
    buffer.append({"brightness": 0.45, "blur_score": 120.5})
    stats = buffer.get_rolling_stats(window=20)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RollingStats:
    """Rolling statistics for a single signal."""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    current: float
    delta_from_anchor: Optional[float] = None
    z_score: Optional[float] = None
    window_size: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "current": self.current,
            "delta_from_anchor": self.delta_from_anchor,
            "z_score": self.z_score,
            "window_size": self.window_size,
        }


class TemporalBuffer:
    """
    Fixed-size rolling buffer for streaming signal values.

    Stores a history of signal dictionaries and provides efficient
    computation of rolling statistics for change detection.

    Args:
        max_size: Maximum number of entries to store.
        signal_names: List of expected signal names (for validation).
    """

    def __init__(
        self,
        max_size: int = 200,
        signal_names: Optional[list[str]] = None,
    ):
        self.max_size = max_size
        self.signal_names = signal_names or []
        self._buffer: deque[dict[str, float]] = deque(maxlen=max_size)
        self._timestamps: deque[int] = deque(maxlen=max_size)
        self._step_counter: int = 0

        # Source anchor statistics (set from clean/source data)
        self._anchor_mean: dict[str, float] = {}
        self._anchor_std: dict[str, float] = {}

    @property
    def size(self) -> int:
        """Current number of entries in the buffer."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return self.size >= self.max_size

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def append(self, signals: dict[str, float], timestamp: Optional[int] = None) -> None:
        """
        Append a new signal measurement to the buffer.

        Args:
            signals: Dict of signal_name -> value.
            timestamp: Optional explicit timestamp. Auto-incremented if None.
        """
        if timestamp is None:
            timestamp = self._step_counter
        self._buffer.append(signals)
        self._timestamps.append(timestamp)
        self._step_counter += 1

    def get_recent(self, n: int = 1) -> list[dict[str, float]]:
        """Get the n most recent entries."""
        n = min(n, self.size)
        return list(self._buffer)[-n:]

    def get_signal_history(
        self,
        signal_name: str,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get history of a single signal as a numpy array.

        Args:
            signal_name: Name of the signal.
            window: Number of recent entries to include. None = all.

        Returns:
            1D numpy array of signal values.
        """
        entries = list(self._buffer)
        if window is not None:
            entries = entries[-window:]

        values = []
        for entry in entries:
            if signal_name in entry:
                values.append(entry[signal_name])
            else:
                values.append(np.nan)
        return np.array(values, dtype=np.float64)

    def get_all_signals_matrix(
        self,
        signal_names: Optional[list[str]] = None,
        window: Optional[int] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Get multi-signal history as a 2D matrix.

        Returns:
            (matrix [T, D], signal_names)
        """
        if signal_names is None:
            # Auto-detect from first entry
            if self.is_empty:
                return np.zeros((0, 0)), []
            signal_names = sorted(list(self._buffer[0].keys()))

        columns = []
        for name in signal_names:
            columns.append(self.get_signal_history(name, window))

        if not columns:
            return np.zeros((0, 0)), []

        return np.column_stack(columns), signal_names

    def set_anchor(
        self,
        anchor_mean: dict[str, float],
        anchor_std: dict[str, float],
    ) -> None:
        """
        Set source anchor statistics for drift measurement.

        Args:
            anchor_mean: Mean of each signal on clean source data.
            anchor_std: Std of each signal on clean source data.
        """
        self._anchor_mean = anchor_mean
        self._anchor_std = anchor_std

    def compute_anchor_from_buffer(self, window: Optional[int] = None) -> None:
        """
        Compute anchor statistics from the current buffer contents.
        Call this on clean/source data to establish baseline.
        """
        if self.is_empty:
            return

        entries = list(self._buffer)
        if window is not None:
            entries = entries[-window:]

        # Collect all signal names
        all_names: set[str] = set()
        for entry in entries:
            all_names.update(entry.keys())

        for name in all_names:
            values = [e.get(name, np.nan) for e in entries]
            values = [v for v in values if not np.isnan(v)]
            if values:
                self._anchor_mean[name] = float(np.mean(values))
                self._anchor_std[name] = float(np.std(values)) + 1e-8

    def get_rolling_stats(
        self,
        window: int = 20,
        signal_names: Optional[list[str]] = None,
    ) -> dict[str, RollingStats]:
        """
        Compute rolling statistics for all tracked signals.

        Args:
            window: Number of recent entries for stats computation.
            signal_names: Specific signals to compute. None = all.

        Returns:
            Dict of signal_name -> RollingStats.
        """
        if self.is_empty:
            return {}

        if signal_names is None:
            signal_names = sorted(list(self._buffer[-1].keys()))

        stats = {}
        for name in signal_names:
            values = self.get_signal_history(name, window)
            valid = values[~np.isnan(values)]

            if len(valid) == 0:
                continue

            current = valid[-1]
            mean = float(np.mean(valid))
            std = float(np.std(valid)) + 1e-8

            # Delta from anchor
            delta = None
            z_score = None
            if name in self._anchor_mean:
                delta = current - self._anchor_mean[name]
                if name in self._anchor_std and self._anchor_std[name] > 1e-8:
                    z_score = delta / self._anchor_std[name]

            stats[name] = RollingStats(
                name=name,
                mean=mean,
                std=std,
                min_val=float(np.min(valid)),
                max_val=float(np.max(valid)),
                current=current,
                delta_from_anchor=delta,
                z_score=z_score,
                window_size=len(valid),
            )

        return stats

    def compute_drift_score(
        self,
        prefixes: list[str],
        window: int = 10,
    ) -> float:
        """
        Compute drift as |window_mean - anchor_mean| / anchor_std.

        Uses z-score of the window mean vs the anchor mean for all
        signals matching any of the given prefixes. This is much more
        robust than CV-based drift which is always ~0.2 on diverse images.

        Args:
            prefixes: List of signal name prefixes to include.
            window: Number of recent frames for window mean.

        Returns:
            Mean absolute z-score across matched signals. 0.0 if no anchor.
        """
        if not self._anchor_mean:
            return 0.0

        stats = self.get_rolling_stats(window=window)
        z_scores = []
        for key, rs in stats.items():
            if not any(p in key for p in prefixes):
                continue
            if key not in self._anchor_mean:
                continue
            anchor_mean = self._anchor_mean[key]
            anchor_std = self._anchor_std.get(key, 1.0)
            # Use robust minimum std: max of anchor_std, 10% of |anchor_mean|,
            # and 0.01 to prevent huge z-scores from near-zero variance signals
            min_std = max(0.01, 0.1 * abs(anchor_mean))
            effective_std = max(anchor_std, min_std)
            # z-score of window mean vs anchor mean
            z = abs(rs.mean - anchor_mean) / effective_std
            z_scores.append(z)
        return float(np.mean(z_scores)) if z_scores else 0.0

    def get_fused_vector(
        self,
        signal_names: Optional[list[str]] = None,
        window: int = 20,
        include_stats: bool = True,
    ) -> np.ndarray:
        """
        Build a fused feature vector from rolling stats.

        For each signal, includes: current, mean, std, z_score, delta_from_anchor.
        This vector can be input to a diagnosis classifier or harm predictor.

        Returns:
            1D numpy array of features.
        """
        stats = self.get_rolling_stats(window, signal_names)
        if not stats:
            return np.zeros(0)

        features = []
        for name in sorted(stats.keys()):
            s = stats[name]
            features.extend([
                s.current,
                s.mean,
                s.std,
                s.z_score if s.z_score is not None else 0.0,
                s.delta_from_anchor if s.delta_from_anchor is not None else 0.0,
            ])

        return np.array(features, dtype=np.float64)

    def clear(self) -> None:
        """Clear all entries (keeps anchor)."""
        self._buffer.clear()
        self._timestamps.clear()
        self._step_counter = 0

    def summary(self) -> dict:
        """Return buffer state summary."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "has_anchor": bool(self._anchor_mean),
            "anchor_signals": list(self._anchor_mean.keys()),
            "step_counter": self._step_counter,
        }
