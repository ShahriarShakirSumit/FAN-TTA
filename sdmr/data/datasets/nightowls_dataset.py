"""
NightOwls Dataset Loader.

The NightOwls dataset is designed for pedestrian detection at night.
279K frames from 40 sequences across 3 European countries.
Resolution: 1024 × 640, frame rate: 5.33fps

Dataset: https://www.nightowls-dataset.org/
Paper: Neumann et al., "NightOwls: A Pedestrians at Night Dataset", ACCV 2018

NightOwls uses COCO-format JSON annotations.

Usage:
    dataset = NightOwlsDataset(
        images_dir="/path/to/nightowls/nightowls_validation",
        annotations_file="/path/to/nightowls/nightowls_validation.json",
    )
    frame = dataset[0]  # returns StreamFrame
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np

from .stream_wrapper import StreamFrame, FrameMetadata

logger = logging.getLogger(__name__)

# NightOwls class mapping
NIGHTOWLS_CLASSES = {
    'pedestrian': 0,
    'bicycledriver': 1,
    'motorbikedriver': 2,
    'ignore': -1,
}

NIGHTOWLS_CAT_ID_MAP = {
    1: 'pedestrian',
    2: 'bicycledriver',
    3: 'motorbikedriver',
}


@dataclass
class NightOwlsConfig:
    """Configuration for NightOwls dataset."""
    images_dir: str = ""
    annotations_file: str = ""
    split: str = "validation"        # "training", "validation", or "test"
    person_only: bool = True         # only load pedestrian class
    min_height: int = 20             # minimum bbox height in pixels
    min_area: float = 0.0            # min bbox area
    max_occlusion: float = 0.8       # max occlusion ratio
    include_cyclists: bool = False   # include bicycle/motorbike driver
    sort_by_id: bool = True          # deterministic ordering
    sequence_mode: bool = False      # preserve sequence ordering for streaming
    max_images: int = -1             # limit number of images (-1 = all)


class NightOwlsDataset:
    """
    NightOwls nighttime pedestrian detection dataset loader.

    The dataset is entirely recorded at night, providing a natural target
    domain for day→night distribution shift experiments.

    Each item returns a StreamFrame with:
    - image: numpy array (H, W, 3) in RGB (native: 1024×640)
    - targets: dict with 'boxes' (N, 4) in [x1, y1, x2, y2], 'labels' (N,)
    - metadata: FrameMetadata with shift_active=True (always night)
    """

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        config: Optional[NightOwlsConfig] = None,
        transform: Optional[callable] = None,
    ):
        self.config = config or NightOwlsConfig()
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.transform = transform

        self._load_annotations()

    def _load_annotations(self):
        """Load NightOwls COCO-format annotations."""
        logger.info(f"Loading NightOwls annotations from {self.annotations_file}")

        if not self.annotations_file.exists():
            logger.warning(
                f"NightOwls annotations not found: {self.annotations_file}\n"
                f"Download from: https://www.nightowls-dataset.org/download/\n"
                f"Validation annotations: http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.json"
            )
            self.image_ids = []
            self.image_info = {}
            self.annotations_by_image = {}
            return

        with open(self.annotations_file, 'r') as f:
            data = json.load(f)

        # Build category map
        self.categories = {}
        if 'categories' in data:
            for cat in data['categories']:
                self.categories[cat['id']] = cat['name']

        # Build image info map
        self.image_info = {
            img['id']: img for img in data['images']
        }

        # Group annotations by image, filtering for person class
        self.annotations_by_image = {}
        total_instances = 0

        for ann in data.get('annotations', []):
            cat_id = ann.get('category_id', -1)
            cat_name = self.categories.get(cat_id, NIGHTOWLS_CAT_ID_MAP.get(cat_id, 'unknown')).lower()

            # Class filter
            if self.config.person_only and cat_name != 'pedestrian':
                if not (self.config.include_cyclists and cat_name in ('bicycledriver', 'motorbikedriver')):
                    continue

            # Area filter
            if ann.get('area', float('inf')) < self.config.min_area:
                continue

            # Height filter (from bbox)
            bbox = ann.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4 and bbox[3] < self.config.min_height:
                continue

            # Occlusion filter
            occ = ann.get('occlusion', ann.get('occluded', 0.0))
            if isinstance(occ, (int, float)) and float(occ) > self.config.max_occlusion:
                continue

            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
            total_instances += 1

        # Get image IDs
        if self.config.sequence_mode:
            # Preserve original ordering for streaming evaluation
            self.image_ids = [img['id'] for img in data['images']
                              if img['id'] in self.annotations_by_image]
        else:
            self.image_ids = sorted(self.annotations_by_image.keys()) if self.config.sort_by_id else list(self.annotations_by_image.keys())

        # Apply max_images limit
        if self.config.max_images > 0:
            self.image_ids = self.image_ids[:self.config.max_images]

        logger.info(
            f"NightOwls {self.config.split}: {len(self.image_ids)} images, "
            f"{total_instances} person instances"
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> StreamFrame:
        """Get a single frame by index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        img_id = self.image_ids[idx]
        img_info = self.image_info[img_id]

        # Load image
        image = self._load_image(img_info)

        # Build targets
        anns = self.annotations_by_image.get(img_id, [])
        targets = self._build_targets(anns, img_info)

        # Apply transform
        if self.transform:
            image, targets = self.transform(image, targets)

        # Build metadata — NightOwls is ALWAYS night, so shift_active=True
        # when used as target domain
        metadata = FrameMetadata(
            stream_index=idx,
            global_index=img_id,
            shift_active=True,
            shift_family="illumination_night",
            shift_severity=0.9,  # high severity — fully night
            scenario_id=f"nightowls_{self.config.split}",
        )

        return StreamFrame(image=image, targets=targets, metadata=metadata)

    def _load_image(self, img_info: dict) -> np.ndarray:
        """Load image as RGB numpy array."""
        from PIL import Image

        file_name = img_info.get('file_name', '')

        # NightOwls may have paths like "images/nightowls_validation/000001.png"
        # or just "000001.png"
        img_path = self.images_dir / file_name
        if not img_path.exists():
            # Try just the basename
            img_path = self.images_dir / Path(file_name).name
        if not img_path.exists():
            raise FileNotFoundError(
                f"NightOwls image not found: {img_path}\n"
                f"Download from: https://www.nightowls-dataset.org/download/"
            )

        img = Image.open(img_path).convert("RGB")
        return np.array(img)

    def _build_targets(self, annotations: list, img_info: dict) -> dict:
        """Convert NightOwls annotations to standard target format."""
        w = img_info.get('width', 1024)
        h = img_info.get('height', 640)

        boxes = []
        labels = []
        areas = []
        difficulties = []

        for ann in annotations:
            # COCO bbox format: [x, y, width, height] -> [x1, y1, x2, y2]
            bbox = ann.get('bbox', [0, 0, 0, 0])
            if len(bbox) < 4:
                continue

            x, y, bw, bh = bbox[:4]
            x1, y1 = float(x), float(y)
            x2, y2 = x1 + float(bw), y1 + float(bh)

            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cat_id = ann.get('category_id', 1)
            cat_name = self.categories.get(cat_id, NIGHTOWLS_CAT_ID_MAP.get(cat_id, 'unknown')).lower()
            class_id = NIGHTOWLS_CLASSES.get(cat_name, 0)

            if class_id < 0:  # skip ignore regions
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
            areas.append(ann.get('area', bw * bh))
            difficulties.append(ann.get('difficulty', 0))

        targets = {
            'boxes': np.array(boxes, dtype=np.float32).reshape(-1, 4),
            'labels': np.array(labels, dtype=np.int64),
            'areas': np.array(areas, dtype=np.float32),
            'difficulties': np.array(difficulties, dtype=np.int64),
            'image_id': img_info['id'],
            'image_width': w,
            'image_height': h,
        }

        return targets

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        total_instances = sum(len(v) for v in self.annotations_by_image.values())
        instances_per_image = [
            len(self.annotations_by_image.get(img_id, []))
            for img_id in self.image_ids
        ]
        return {
            'num_images': len(self.image_ids),
            'num_instances': total_instances,
            'split': self.config.split,
            'instances_per_image_mean': np.mean(instances_per_image) if instances_per_image else 0,
            'instances_per_image_max': max(instances_per_image) if instances_per_image else 0,
        }

    def get_sequence_info(self) -> dict:
        """Return sequence-level information for streaming evaluation."""
        sequences = {}
        for img_id in self.image_ids:
            info = self.image_info.get(img_id, {})
            seq_id = info.get('seq_id', info.get('video_id', 'unknown'))
            if seq_id not in sequences:
                sequences[seq_id] = []
            sequences[seq_id].append(img_id)
        return {
            'num_sequences': len(sequences),
            'sequence_lengths': {k: len(v) for k, v in sequences.items()},
        }
