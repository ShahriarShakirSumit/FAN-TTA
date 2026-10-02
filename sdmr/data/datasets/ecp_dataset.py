"""
EuroCity Persons (ECP) Dataset Loader.

The ECP dataset contains 47,300 images with 238,200 person instances from 31
European cities. Critically for SDMR, it includes BOTH day and night subsets,
enabling natural day→night distribution shift evaluation.

Dataset: https://eurocity-dataset.tudelft.nl/
Paper: Braun et al., "The EuroCity Persons Dataset", IEEE TPAMI 2019

ECP annotation format uses JSON with per-image annotation files.
Each annotation file contains objects with class, bounding box, occlusion, etc.

Usage:
    dataset = ECPDataset(
        images_dir="/path/to/ecp/day/img/val",
        annotations_dir="/path/to/ecp/day/labels/val",
        time_of_day="night",  # "day" or "night"
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

# ECP class mapping
ECP_CLASSES = {
    'pedestrian': 0,
    'rider': 1,
    'rider+vehicle-group-bounding-box': -1,  # ignore
    'bicycle-group': -1,
    'motorbike-group': -1,
    'scooter-group': -1,
    'buggy-group': -1,
    'wheelchair-group': -1,
    'tricycle-group': -1,
}


@dataclass
class ECPConfig:
    """Configuration for ECP dataset."""
    images_dir: str = ""
    annotations_dir: str = ""
    time_of_day: str = "night"       # "day" or "night"
    split: str = "val"               # "train" or "val"
    person_only: bool = True         # only load pedestrian class
    min_height: int = 20             # minimum bbox height in pixels
    max_occlusion: float = 0.8       # max occlusion ratio (0=visible, 1=fully occluded)
    include_riders: bool = False     # include rider class
    sort_files: bool = True          # sort by filename for deterministic ordering


class ECPDataset:
    """
    EuroCity Persons dataset loader.

    Supports separate loading of day and night subsets for
    evaluating domain shift adaptation.

    Each item returns a StreamFrame with:
    - image: numpy array (H, W, 3) in RGB
    - targets: dict with 'boxes' (N, 4) in [x1, y1, x2, y2], 'labels' (N,)
    - metadata: FrameMetadata with scenario_id indicating time_of_day
    """

    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        config: Optional[ECPConfig] = None,
        transform: Optional[callable] = None,
    ):
        self.config = config or ECPConfig()
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform

        # Discover and load annotation files
        self._discover_files()

    def _discover_files(self):
        """Discover matching image and annotation files."""
        # ECP organizes by city subdirectories
        # Structure: {split}/{city}/{filename}.png and labels/{split}/{city}/{filename}.json

        self.image_paths = []
        self.annotation_paths = []
        self.file_identifiers = []

        if not self.images_dir.exists():
            logger.warning(
                f"ECP images directory not found: {self.images_dir}\n"
                f"Download from: https://eurocity-dataset.tudelft.nl/eval/downloads/detection"
            )
            return

        # Walk through the images directory
        image_files = []
        for root, dirs, files in os.walk(self.images_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(Path(root) / f)

        if self.config.sort_files:
            image_files.sort()

        for img_path in image_files:
            # Derive annotation path from image path
            relative = img_path.relative_to(self.images_dir)
            ann_path = self.annotations_dir / relative.with_suffix('.json')

            if ann_path.exists():
                self.image_paths.append(img_path)
                self.annotation_paths.append(ann_path)
                self.file_identifiers.append(str(relative))

        logger.info(
            f"ECP {self.config.time_of_day} {self.config.split}: "
            f"found {len(self.image_paths)} images with annotations"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> StreamFrame:
        """Get a single frame by index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Load image
        image = self._load_image(self.image_paths[idx])

        # Load annotations
        targets = self._load_annotations(self.annotation_paths[idx], image.shape)

        # Apply transform if any
        if self.transform:
            image, targets = self.transform(image, targets)

        # Build metadata
        metadata = FrameMetadata(
            stream_index=idx,
            global_index=idx,
            shift_active=(self.config.time_of_day == "night"),
            shift_family="illumination" if self.config.time_of_day == "night" else "none",
            shift_severity=0.8 if self.config.time_of_day == "night" else 0.0,
            scenario_id=f"ecp_{self.config.time_of_day}_{self.config.split}",
        )

        return StreamFrame(image=image, targets=targets, metadata=metadata)

    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load image as RGB numpy array."""
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        return np.array(img)

    def _load_annotations(self, ann_path: Path, img_shape: tuple) -> dict:
        """Load ECP JSON annotation and convert to standard format."""
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)

        h, w = img_shape[:2]

        boxes = []
        labels = []
        occlusions = []
        orientations = []

        # ECP annotations structure:
        # {"children": [{"identity": "pedestrian", "x0": ..., "y0": ..., "x1": ..., "y1": ..., ...}]}
        objects = ann_data.get('children', [])
        if not objects and 'objects' in ann_data:
            objects = ann_data['objects']

        for obj in objects:
            identity = obj.get('identity', obj.get('class', '')).lower()

            # Filter by class
            if self.config.person_only and identity != 'pedestrian':
                if not (self.config.include_riders and identity == 'rider'):
                    continue

            # Map class name to ID
            class_id = ECP_CLASSES.get(identity, -1)
            if class_id < 0:
                continue

            # Get bbox - ECP may use x0/y0/x1/y1 or x/y/w/h format
            if 'x0' in obj:
                x1, y1 = obj['x0'], obj['y0']
                x2, y2 = obj['x1'], obj['y1']
            elif 'bbox' in obj:
                x1, y1, bw, bh = obj['bbox']
                x2, y2 = x1 + bw, y1 + bh
            else:
                continue

            # Clip to image bounds
            x1 = max(0, float(x1))
            y1 = max(0, float(y1))
            x2 = min(w, float(x2))
            y2 = min(h, float(y2))

            # Skip degenerate or too-small boxes
            box_h = y2 - y1
            if x2 <= x1 or y2 <= y1:
                continue
            if box_h < self.config.min_height:
                continue

            # Check occlusion
            occ = obj.get('occlusion', obj.get('occluded', 0.0))
            if isinstance(occ, str):
                occ = {'none': 0.0, 'partial': 0.3, 'heavy': 0.7, 'full': 1.0}.get(occ.lower(), 0.0)
            if float(occ) > self.config.max_occlusion:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
            occlusions.append(float(occ))

            # Orientation if available
            orientation = obj.get('orientation', -1)
            orientations.append(orientation)

        targets = {
            'boxes': np.array(boxes, dtype=np.float32).reshape(-1, 4),
            'labels': np.array(labels, dtype=np.int64),
            'occlusions': np.array(occlusions, dtype=np.float32),
            'orientations': np.array(orientations, dtype=np.float32),
            'image_width': w,
            'image_height': h,
        }

        return targets

    def get_stats(self) -> dict:
        """Return dataset statistics (lazy evaluation of a sample)."""
        return {
            'num_images': len(self),
            'time_of_day': self.config.time_of_day,
            'split': self.config.split,
        }

    def get_city_distribution(self) -> dict:
        """Return count of images per city."""
        city_counts = {}
        for fid in self.file_identifiers:
            city = fid.split('/')[0] if '/' in fid else 'unknown'
            city_counts[city] = city_counts.get(city, 0) + 1
        return city_counts
