"""
Cross-Domain Dataset for evaluating TTA methods under real distribution shift.

Concatenates a SOURCE dataset (e.g., COCO person) with a TARGET dataset
(e.g., NightOwls, ECP Night) into a single streaming sequence:

  [Source frames 0..n_source-1] → [Target frames n_source..n_source+n_target-1]

The source frames form the "clean" anchor period (daytime, in-distribution).
The target frames represent genuine domain shift (nighttime, out-of-distribution).
No synthetic corruption is applied — the shift is real.

Usage with run_stream_experiment.py:
  - dataset: "cross_domain"
  - scenario type: "abrupt" with clean_before=n_source, shift_length=n_target
  - shift_injector: None (no synthetic corruption needed)
"""

import logging
from typing import Optional

import numpy as np

from .stream_wrapper import StreamFrame, FrameMetadata

logger = logging.getLogger(__name__)


class CrossDomainDataset:
    """
    Concatenation of a source and target dataset for cross-domain streaming.

    The source dataset provides clean, in-distribution images for the anchor
    period. The target dataset provides naturally shifted images (e.g., nighttime
    pedestrians) for the evaluation period.

    Both datasets must support __len__() and __getitem__(idx) returning either:
    - (image, targets) tuple, or
    - StreamFrame object

    The combined dataset returns (image, targets) tuples with consistent format.
    """

    def __init__(
        self,
        source_dataset,
        target_dataset,
        n_source: int = 200,
        n_target: Optional[int] = None,
        source_name: str = "coco",
        target_name: str = "target",
    ):
        """
        Args:
            source_dataset: Clean/in-distribution dataset (e.g., COCOPersonDataset)
            target_dataset: Shifted/OOD dataset (e.g., NightOwlsDataset, ECPDataset)
            n_source: Number of source frames to use for anchor period
            n_target: Number of target frames (None = use all available)
            source_name: Name identifier for source domain
            target_name: Name identifier for target domain
        """
        self.source = source_dataset
        self.target = target_dataset
        self.source_name = source_name
        self.target_name = target_name

        # Determine actual sizes
        self.n_source = min(n_source, len(source_dataset))
        self.n_target = n_target if n_target is not None else len(target_dataset)
        self.n_target = min(self.n_target, len(target_dataset))

        self._total = self.n_source + self.n_target

        logger.info(
            f"CrossDomainDataset: {self.n_source} source ({source_name}) + "
            f"{self.n_target} target ({target_name}) = {self._total} frames"
        )

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        """
        Get frame by index.

        Args:
            idx: Stream index (0..total-1)

        Returns:
            (image, targets) tuple where image is np.ndarray (H, W, 3)
            and targets is dict with 'boxes', 'labels', etc.
        """
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index {idx} out of range [0, {self._total})")

        if idx < self.n_source:
            # Source domain (clean, in-distribution)
            item = self.source[idx]
        else:
            # Target domain (naturally shifted, OOD)
            target_idx = idx - self.n_source
            item = self.target[target_idx]

        # Handle both StreamFrame and (image, targets) return types
        if isinstance(item, StreamFrame):
            return item.image, item.targets
        elif isinstance(item, tuple) and len(item) == 2:
            return item
        else:
            raise TypeError(
                f"Dataset returned unexpected type: {type(item)}. "
                f"Expected StreamFrame or (image, targets) tuple."
            )

    @property
    def total_frames(self) -> int:
        return self._total

    def get_domain_info(self, idx: int) -> dict:
        """Get domain information for a frame index."""
        is_source = idx < self.n_source
        return {
            'domain': self.source_name if is_source else self.target_name,
            'is_source': is_source,
            'domain_index': idx if is_source else idx - self.n_source,
        }


class SimpleAdapter:
    """Adapt a dataset that returns StreamFrame to return (image, targets) tuples."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, StreamFrame):
            return item.image, item.targets
        return item
