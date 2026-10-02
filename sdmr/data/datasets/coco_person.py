"""
COCO Person Subset Dataset Loader.

Filters COCO 2017 annotations to only include images with at least one 'person'
instance. Provides streaming-compatible interface for SDMR evaluation.

Standard COCO 2017:
- train2017: ~118K images (person appears in ~64K instances)
- val2017: 5K images (~2,700 contain person, ~10K person instances)

Usage:
    dataset = COCOPersonDataset(
        images_dir="/path/to/coco/val2017",
        annotations_file="/path/to/coco/annotations/instances_val2017.json",
    )
    frame = dataset[0]  # returns StreamFrame
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np

from .stream_wrapper import StreamFrame, FrameMetadata

logger = logging.getLogger(__name__)

# COCO category ID for 'person' is 1
COCO_PERSON_CATEGORY_ID = 1


@dataclass
class COCOPersonConfig:
    """Configuration for COCO Person dataset."""
    images_dir: str = ""
    annotations_file: str = ""
    min_area: float = 0.0          # min bbox area in pixels^2
    min_visibility: float = 0.0    # min fraction visible (0=fully occluded, 1=fully visible)
    max_crowd: bool = False        # include iscrowd annotations
    person_only: bool = True       # filter to person class only
    sort_by_id: bool = True        # sort images by ID for deterministic ordering


class COCOPersonDataset:
    """
    COCO dataset filtered to person-class images only.

    Provides a list-like interface compatible with StreamWrapper.
    Each item returns a StreamFrame with:
    - image: loaded as numpy array (H, W, 3) in RGB
    - targets: dict with 'boxes' (N, 4) in [x1, y1, x2, y2], 'labels' (N,), 'areas' (N,)
    - metadata: FrameMetadata with global_index set
    """

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        config: Optional[COCOPersonConfig] = None,
        transform: Optional[callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.config = config or COCOPersonConfig()
        self.transform = transform

        # Load and filter annotations
        self._load_annotations()

    def _load_annotations(self):
        """Load COCO annotations and filter for person class."""
        logger.info(f"Loading COCO annotations from {self.annotations_file}")

        if not self.annotations_file.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {self.annotations_file}\n"
                f"Download from: http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            )

        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)

        # Build category name map
        self.categories = {
            cat['id']: cat['name'] for cat in coco_data['categories']
        }

        # Build image info map
        self.image_info = {
            img['id']: img for img in coco_data['images']
        }

        # Filter annotations for person class
        person_annotations = {}
        for ann in coco_data['annotations']:
            # Skip if not person
            if self.config.person_only and ann['category_id'] != COCO_PERSON_CATEGORY_ID:
                continue

            # Skip crowd annotations if configured
            if not self.config.max_crowd and ann.get('iscrowd', 0):
                continue

            # Skip small boxes
            if ann['area'] < self.config.min_area:
                continue

            img_id = ann['image_id']
            if img_id not in person_annotations:
                person_annotations[img_id] = []
            person_annotations[img_id].append(ann)

        # Get the list of image IDs that have person annotations
        self.person_image_ids = sorted(person_annotations.keys()) if self.config.sort_by_id else list(person_annotations.keys())
        self.person_annotations = person_annotations

        # Statistics
        total_instances = sum(len(v) for v in person_annotations.values())
        logger.info(
            f"COCO Person subset: {len(self.person_image_ids)} images, "
            f"{total_instances} person instances "
            f"(from {len(coco_data['images'])} total images)"
        )

    def __len__(self) -> int:
        return len(self.person_image_ids)

    def __getitem__(self, idx: int) -> StreamFrame:
        """Get a single frame by index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        img_id = self.person_image_ids[idx]
        img_info = self.image_info[img_id]

        # Load image
        image = self._load_image(img_info['file_name'])

        # Build targets
        anns = self.person_annotations[img_id]
        targets = self._build_targets(anns, img_info)

        # Apply transform if any
        if self.transform:
            image, targets = self.transform(image, targets)

        # Build metadata
        metadata = FrameMetadata(
            stream_index=idx,
            global_index=img_id,
            shift_active=False,
            shift_family="none",
            shift_severity=0.0,
            scenario_id="coco_person_source",
        )

        return StreamFrame(image=image, targets=targets, metadata=metadata)

    def _load_image(self, file_name: str) -> np.ndarray:
        """Load image as RGB numpy array."""
        from PIL import Image

        img_path = self.images_dir / file_name
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Download from: http://images.cocodataset.org/zips/val2017.zip"
            )

        img = Image.open(img_path).convert("RGB")
        return np.array(img)

    def _build_targets(self, annotations: list, img_info: dict) -> dict:
        """Convert COCO annotations to standard target format."""
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            # COCO bbox is [x, y, width, height] -> convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_info['width'], x2)
            y2 = min(img_info['height'], y2)

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(0)  # person = class 0 in our framework
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        targets = {
            'boxes': np.array(boxes, dtype=np.float32).reshape(-1, 4),
            'labels': np.array(labels, dtype=np.int64),
            'areas': np.array(areas, dtype=np.float32),
            'iscrowd': np.array(iscrowd, dtype=np.int64),
            'image_id': img_info['id'],
            'image_width': img_info['width'],
            'image_height': img_info['height'],
        }

        return targets

    def get_image_ids(self) -> list[int]:
        """Return ordered list of image IDs."""
        return list(self.person_image_ids)

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        total_instances = sum(len(v) for v in self.person_annotations.values())
        instances_per_image = [
            len(self.person_annotations[img_id])
            for img_id in self.person_image_ids
        ]
        return {
            'num_images': len(self.person_image_ids),
            'num_instances': total_instances,
            'instances_per_image_mean': np.mean(instances_per_image) if instances_per_image else 0,
            'instances_per_image_std': np.std(instances_per_image) if instances_per_image else 0,
            'instances_per_image_max': max(instances_per_image) if instances_per_image else 0,
        }

    def subset(self, num_images: int, seed: int = 42) -> "COCOPersonDataset":
        """
        Return a subset of this dataset with num_images randomly selected images.
        Useful for creating a smaller evaluation set.
        """
        rng = np.random.RandomState(seed)
        n = min(num_images, len(self.person_image_ids))
        selected_indices = rng.choice(len(self.person_image_ids), size=n, replace=False)
        selected_ids = [self.person_image_ids[i] for i in sorted(selected_indices)]

        # Create a shallow copy with subset
        import copy
        subset_ds = copy.copy(self)
        subset_ds.person_image_ids = selected_ids
        subset_ds.person_annotations = {
            img_id: self.person_annotations[img_id] for img_id in selected_ids
        }
        return subset_ds
