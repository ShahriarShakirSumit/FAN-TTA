"""
Stream wrapper for converting detection datasets into ordered evaluation streams.

Supports:
- Sliding or chunked windows
- Shift scenario schedules (clean→shifted→clean, gradual, abrupt, mixed)
- Per-frame metadata for diagnosis supervision

Usage:
    stream = StreamWrapper(dataset, scenario=scenario, window_size=32)
    for window in stream:
        # window.frames: list of (image, targets, metadata)
        ...
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Iterator, Any
import numpy as np
from enum import Enum


class ShiftPattern(Enum):
    """Supported shift schedule patterns."""
    CLEAN = "clean"
    ABRUPT = "abrupt"                # clean -> shifted -> clean
    GRADUAL = "gradual"              # smooth severity ramp
    RECURRING = "recurring"          # periodic abrupt shifts
    MIXED = "mixed"                  # multiple shift types in sequence
    CUSTOM = "custom"                # user-defined schedule


@dataclass
class FrameMetadata:
    """Metadata attached to each frame in the stream."""
    stream_index: int
    global_index: int
    shift_active: bool = False
    shift_family: str = "none"       # e.g., "blur", "noise", "brightness"
    shift_severity: float = 0.0      # 0.0 = clean, 1.0 = max corruption
    scenario_id: str = ""
    window_id: int = -1
    is_warmup: bool = False

    def to_dict(self) -> dict:
        return {
            "stream_index": self.stream_index,
            "global_index": self.global_index,
            "shift_active": self.shift_active,
            "shift_family": self.shift_family,
            "shift_severity": self.shift_severity,
            "scenario_id": self.scenario_id,
            "window_id": self.window_id,
            "is_warmup": self.is_warmup,
        }


@dataclass
class StreamFrame:
    """A single frame in the stream with its data and metadata."""
    image: Any                       # PIL Image or np.ndarray or Tensor
    targets: Optional[dict] = None   # detection targets (boxes, labels, etc.)
    metadata: Optional[FrameMetadata] = None


@dataclass
class StreamWindow:
    """A window of frames for batch processing."""
    frames: list[StreamFrame]
    window_id: int
    start_index: int
    end_index: int

    @property
    def size(self) -> int:
        return len(self.frames)

    @property
    def images(self) -> list:
        return [f.image for f in self.frames]

    @property
    def metadata_list(self) -> list[FrameMetadata]:
        return [f.metadata for f in self.frames]

    @property
    def has_shift(self) -> bool:
        return any(f.metadata.shift_active for f in self.frames if f.metadata)

    @property
    def dominant_shift_family(self) -> str:
        """Most common shift family in this window."""
        families = [f.metadata.shift_family for f in self.frames
                    if f.metadata and f.metadata.shift_active]
        if not families:
            return "none"
        from collections import Counter
        return Counter(families).most_common(1)[0][0]


@dataclass
class ShiftScenario:
    """
    Defines a shift schedule over a stream.

    Example scenarios:
    - 500 clean, 200 fog at severity 0.7, 300 clean
    - Gradual brightness ramp from 0 to 0.8 over 400 frames
    - Recurring noise every 200 frames for 50 frames
    """
    pattern: ShiftPattern = ShiftPattern.CLEAN
    total_frames: int = 1000
    segments: list[dict] = field(default_factory=list)
    seed: int = 42

    def get_shift_info(self, frame_index: int) -> tuple[bool, str, float]:
        """
        Get shift status for a given frame index.

        Returns:
            (shift_active, shift_family, severity)
        """
        if self.pattern == ShiftPattern.CLEAN:
            return False, "none", 0.0

        cumulative = 0
        for seg in self.segments:
            seg_len = seg.get("length", 0)
            if cumulative <= frame_index < cumulative + seg_len:
                family = seg.get("family", "none")
                severity = seg.get("severity", 0.0)

                if seg.get("gradual", False):
                    # Linear ramp within segment
                    progress = (frame_index - cumulative) / max(seg_len - 1, 1)
                    start_sev = seg.get("start_severity", 0.0)
                    end_sev = seg.get("end_severity", severity)
                    severity = start_sev + progress * (end_sev - start_sev)

                active = family != "none" and severity > 0.0
                return active, family, severity
            cumulative += seg_len

        return False, "none", 0.0

    @classmethod
    def abrupt(
        cls,
        clean_before: int = 300,
        shift_length: int = 200,
        clean_after: int = 300,
        shift_family: str = "gaussian_noise",
        severity: float = 0.5,
        seed: int = 42,
    ) -> "ShiftScenario":
        """Create a clean → shift → clean scenario."""
        segments = [
            {"length": clean_before, "family": "none", "severity": 0.0},
            {"length": shift_length, "family": shift_family, "severity": severity},
            {"length": clean_after, "family": "none", "severity": 0.0},
        ]
        total = clean_before + shift_length + clean_after
        return cls(
            pattern=ShiftPattern.ABRUPT,
            total_frames=total,
            segments=segments,
            seed=seed,
        )

    @classmethod
    def gradual(
        cls,
        clean_before: int = 200,
        ramp_length: int = 200,
        hold_length: int = 200,
        ramp_down_length: int = 200,
        clean_after: int = 200,
        shift_family: str = "brightness",
        max_severity: float = 0.8,
        seed: int = 42,
    ) -> "ShiftScenario":
        """Create a gradual drift scenario."""
        segments = [
            {"length": clean_before, "family": "none", "severity": 0.0},
            {"length": ramp_length, "family": shift_family,
             "severity": max_severity, "gradual": True,
             "start_severity": 0.0, "end_severity": max_severity},
            {"length": hold_length, "family": shift_family,
             "severity": max_severity},
            {"length": ramp_down_length, "family": shift_family,
             "severity": 0.0, "gradual": True,
             "start_severity": max_severity, "end_severity": 0.0},
            {"length": clean_after, "family": "none", "severity": 0.0},
        ]
        total = clean_before + ramp_length + hold_length + ramp_down_length + clean_after
        return cls(
            pattern=ShiftPattern.GRADUAL,
            total_frames=total,
            segments=segments,
            seed=seed,
        )

    @classmethod
    def recurring(
        cls,
        num_cycles: int = 4,
        clean_length: int = 150,
        shift_length: int = 100,
        shift_family: str = "gaussian_blur",
        severity: float = 0.6,
        seed: int = 42,
    ) -> "ShiftScenario":
        """Create a recurring shift scenario."""
        segments = []
        for _ in range(num_cycles):
            segments.append({"length": clean_length, "family": "none", "severity": 0.0})
            segments.append({"length": shift_length, "family": shift_family, "severity": severity})
        segments.append({"length": clean_length, "family": "none", "severity": 0.0})
        total = num_cycles * (clean_length + shift_length) + clean_length
        return cls(
            pattern=ShiftPattern.RECURRING,
            total_frames=total,
            segments=segments,
            seed=seed,
        )

    @classmethod
    def mixed(
        cls,
        segment_specs: list[dict],
        seed: int = 42,
    ) -> "ShiftScenario":
        """Create a mixed-shift scenario from explicit segment specs."""
        total = sum(s.get("length", 0) for s in segment_specs)
        return cls(
            pattern=ShiftPattern.MIXED,
            total_frames=total,
            segments=segment_specs,
            seed=seed,
        )


class StreamWrapper:
    """
    Wraps a dataset into a streaming evaluation pipeline.

    The dataset must support __len__ and __getitem__ returning (image, targets).
    Images are repeated/cycled if total_frames > len(dataset).

    Args:
        dataset: Base dataset (must support indexing).
        scenario: ShiftScenario defining the corruption schedule.
        window_size: Number of frames per window.
        window_stride: Stride between windows (default = window_size for chunked).
        shift_injector: Optional callable (image, family, severity, seed) -> image.
        warmup_frames: Number of initial frames to mark as warmup (excluded from metrics).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset,
        scenario: Optional[ShiftScenario] = None,
        window_size: int = 32,
        window_stride: Optional[int] = None,
        shift_injector: Optional[Callable] = None,
        warmup_frames: int = 0,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.scenario = scenario or ShiftScenario()
        self.window_size = window_size
        self.window_stride = window_stride or window_size
        self.shift_injector = shift_injector
        self.warmup_frames = warmup_frames
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self._total_frames = self.scenario.total_frames
        self._dataset_len = len(dataset) if hasattr(dataset, '__len__') else self._total_frames

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def num_windows(self) -> int:
        """Number of windows in the stream."""
        if self._total_frames <= self.window_size:
            return 1
        return max(1, (self._total_frames - self.window_size) // self.window_stride + 1)

    def _get_dataset_item(self, stream_index: int):
        """Get item from dataset, cycling if needed."""
        dataset_index = stream_index % self._dataset_len
        return self.dataset[dataset_index]

    def get_frame(self, stream_index: int) -> StreamFrame:
        """Get a single frame from the stream with shift applied."""
        image, targets = self._get_dataset_item(stream_index)

        # Get shift info from scenario
        shift_active, shift_family, severity = self.scenario.get_shift_info(stream_index)

        # Apply shift if active and injector available
        if shift_active and self.shift_injector is not None and severity > 0:
            frame_seed = self.seed + stream_index
            image = self.shift_injector(image, shift_family, severity, frame_seed)

        metadata = FrameMetadata(
            stream_index=stream_index,
            global_index=stream_index,
            shift_active=shift_active,
            shift_family=shift_family,
            shift_severity=severity,
            scenario_id=f"{self.scenario.pattern.value}_{self.seed}",
            is_warmup=stream_index < self.warmup_frames,
        )

        return StreamFrame(image=image, targets=targets, metadata=metadata)

    def get_window(self, window_id: int) -> StreamWindow:
        """Get a window of frames."""
        start = window_id * self.window_stride
        end = min(start + self.window_size, self._total_frames)

        frames = []
        for i in range(start, end):
            frame = self.get_frame(i)
            frame.metadata.window_id = window_id
            frames.append(frame)

        return StreamWindow(
            frames=frames,
            window_id=window_id,
            start_index=start,
            end_index=end,
        )

    def iter_frames(self) -> Iterator[StreamFrame]:
        """Iterate over all frames in the stream."""
        for i in range(self._total_frames):
            yield self.get_frame(i)

    def iter_windows(self) -> Iterator[StreamWindow]:
        """Iterate over all windows in the stream."""
        for w in range(self.num_windows):
            yield self.get_window(w)

    def __len__(self) -> int:
        return self.num_windows

    def __iter__(self) -> Iterator[StreamWindow]:
        return self.iter_windows()

    def summary(self) -> dict:
        """Return a summary of the stream configuration."""
        return {
            "total_frames": self._total_frames,
            "dataset_length": self._dataset_len,
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "num_windows": self.num_windows,
            "warmup_frames": self.warmup_frames,
            "scenario_pattern": self.scenario.pattern.value,
            "num_segments": len(self.scenario.segments),
            "seed": self.seed,
        }
