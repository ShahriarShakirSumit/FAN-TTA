"""
Tests for dataset loaders.

Tests cover:
- COCO Person dataset loader
- ECP dataset loader
- NightOwls dataset loader
- Dataset factory function
- StreamFrame compatibility

These tests use mock data to avoid requiring large dataset downloads.
"""

import json
import os
import tempfile
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from sdmr.data.datasets import (
    create_dataset,
    COCOPersonDataset, COCOPersonConfig,
    ECPDataset, ECPConfig,
    NightOwlsDataset, NightOwlsConfig,
    StreamFrame, FrameMetadata,
)


# ─── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_coco_dir(tmp_path):
    """Create a minimal mock COCO dataset."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()

    # Create mock images
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img.save(images_dir / f"{i:012d}.jpg")

    # Create mock COCO annotations
    coco_ann = {
        "images": [
            {"id": i, "file_name": f"{i:012d}.jpg", "width": 640, "height": 480}
            for i in range(5)
        ],
        "annotations": [
            # Person annotations
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [100, 100, 50, 120], "area": 6000, "iscrowd": 0},
            {"id": 2, "image_id": 0, "category_id": 1, "bbox": [200, 150, 40, 100], "area": 4000, "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [50, 80, 60, 140], "area": 8400, "iscrowd": 0},
            {"id": 4, "image_id": 2, "category_id": 1, "bbox": [300, 200, 45, 110], "area": 4950, "iscrowd": 0},
            # Non-person annotations (should be filtered)
            {"id": 5, "image_id": 3, "category_id": 3, "bbox": [10, 10, 100, 100], "area": 10000, "iscrowd": 0},  # car
            # Image 4 has no person — should be excluded
            {"id": 6, "image_id": 4, "category_id": 2, "bbox": [20, 20, 80, 80], "area": 6400, "iscrowd": 0},  # bicycle
            # Crowd annotation
            {"id": 7, "image_id": 1, "category_id": 1, "bbox": [300, 100, 100, 200], "area": 20000, "iscrowd": 1},
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "bicycle"},
            {"id": 3, "name": "car"},
        ],
    }

    ann_file = ann_dir / "instances_val2017.json"
    with open(ann_file, 'w') as f:
        json.dump(coco_ann, f)

    return {"images_dir": str(images_dir), "annotations_file": str(ann_file)}


@pytest.fixture
def mock_ecp_dir(tmp_path):
    """Create a minimal mock ECP dataset."""
    # ECP structure: images_dir/city/frame.png, annotations_dir/city/frame.json
    images_dir = tmp_path / "images"
    ann_dir = tmp_path / "labels"

    city = "berlin"
    (images_dir / city).mkdir(parents=True)
    (ann_dir / city).mkdir(parents=True)

    for i in range(4):
        # Create mock images
        img = Image.fromarray(np.random.randint(0, 255, (1024, 1920, 3), dtype=np.uint8))
        img.save(images_dir / city / f"frame_{i:06d}.png")

        # Create mock ECP annotations
        ann = {
            "children": [
                {
                    "identity": "pedestrian",
                    "x0": 100 + i * 50,
                    "y0": 200,
                    "x1": 150 + i * 50,
                    "y1": 450,
                    "occlusion": "none",
                    "orientation": 90,
                },
                {
                    "identity": "pedestrian",
                    "x0": 500,
                    "y0": 300,
                    "x1": 550,
                    "y1": 500,
                    "occlusion": "partial",
                    "orientation": 180,
                },
            ]
        }

        # Add a rider that should be filtered in person_only mode
        if i == 2:
            ann["children"].append({
                "identity": "rider",
                "x0": 800, "y0": 400, "x1": 860, "y1": 600,
                "occlusion": "none",
            })

        with open(ann_dir / city / f"frame_{i:06d}.json", 'w') as f:
            json.dump(ann, f)

    return {"images_dir": str(images_dir), "annotations_dir": str(ann_dir)}


@pytest.fixture
def mock_nightowls_dir(tmp_path):
    """Create a minimal mock NightOwls dataset."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create mock images
    for i in range(6):
        img = Image.fromarray(np.random.randint(0, 255, (640, 1024, 3), dtype=np.uint8))
        img.save(images_dir / f"{i:06d}.png")

    # Create mock NightOwls COCO-format annotations
    nightowls_ann = {
        "images": [
            {"id": i, "file_name": f"{i:06d}.png", "width": 1024, "height": 640}
            for i in range(6)
        ],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [100, 100, 50, 120], "area": 6000},
            {"id": 2, "image_id": 0, "category_id": 1, "bbox": [300, 200, 40, 100], "area": 4000},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [500, 150, 60, 140], "area": 8400},
            {"id": 4, "image_id": 2, "category_id": 1, "bbox": [200, 300, 45, 110], "area": 4950},
            {"id": 5, "image_id": 3, "category_id": 2, "bbox": [400, 100, 80, 160], "area": 12800},  # cyclist
            {"id": 6, "image_id": 4, "category_id": 1, "bbox": [150, 250, 55, 130], "area": 7150},
            {"id": 7, "image_id": 5, "category_id": 1, "bbox": [600, 100, 35, 90], "area": 3150},
        ],
        "categories": [
            {"id": 1, "name": "pedestrian"},
            {"id": 2, "name": "bicycledriver"},
            {"id": 3, "name": "motorbikedriver"},
        ],
    }

    ann_file = tmp_path / "nightowls_validation.json"
    with open(ann_file, 'w') as f:
        json.dump(nightowls_ann, f)

    return {"images_dir": str(images_dir), "annotations_file": str(ann_file)}


# ─── COCO Person Tests ────────────────────────────────────────────────

class TestCOCOPersonDataset:
    """Tests for COCO Person subset dataset loader."""

    def test_load_and_filter(self, mock_coco_dir):
        """Test that COCO is filtered to person-only images."""
        ds = COCOPersonDataset(**mock_coco_dir)
        # Images 0, 1, 2 have person annotations; 3 has car, 4 has bicycle
        assert len(ds) == 3

    def test_person_annotations_correct(self, mock_coco_dir):
        """Test that annotations are properly formatted."""
        ds = COCOPersonDataset(**mock_coco_dir)
        frame = ds[0]

        assert isinstance(frame, StreamFrame)
        assert isinstance(frame.image, np.ndarray)
        assert frame.image.shape == (480, 640, 3)

        targets = frame.targets
        assert 'boxes' in targets
        assert 'labels' in targets
        assert targets['boxes'].shape[1] == 4
        # Image 0 has 2 person annotations
        assert targets['boxes'].shape[0] == 2
        assert all(l == 0 for l in targets['labels'])  # person = class 0

    def test_bbox_format(self, mock_coco_dir):
        """Test COCO bbox is converted to [x1, y1, x2, y2]."""
        ds = COCOPersonDataset(**mock_coco_dir)
        frame = ds[0]

        boxes = frame.targets['boxes']
        # First annotation: [100, 100, 50, 120] → [100, 100, 150, 220]
        np.testing.assert_allclose(boxes[0], [100, 100, 150, 220])

    def test_crowd_filtering(self, mock_coco_dir):
        """Test crowd annotations are excluded by default."""
        ds = COCOPersonDataset(**mock_coco_dir)
        # Image 1 has 1 regular + 1 crowd annotation
        frame = ds[1]
        assert frame.targets['boxes'].shape[0] == 1  # crowd excluded

    def test_crowd_include(self, mock_coco_dir):
        """Test crowd annotations can be included."""
        config = COCOPersonConfig(max_crowd=True)
        ds = COCOPersonDataset(**mock_coco_dir, config=config)
        frame = ds[1]  # Image 1 has 1 regular + 1 crowd
        assert frame.targets['boxes'].shape[0] == 2

    def test_metadata(self, mock_coco_dir):
        """Test frame metadata is correctly set."""
        ds = COCOPersonDataset(**mock_coco_dir)
        frame = ds[0]

        assert frame.metadata is not None
        assert frame.metadata.stream_index == 0
        assert frame.metadata.shift_active is False
        assert frame.metadata.shift_family == "none"
        assert frame.metadata.scenario_id == "coco_person_source"

    def test_stats(self, mock_coco_dir):
        """Test dataset statistics computation."""
        ds = COCOPersonDataset(**mock_coco_dir)
        stats = ds.get_stats()
        assert stats['num_images'] == 3
        assert stats['num_instances'] == 4  # 2 + 1 + 1

    def test_subset(self, mock_coco_dir):
        """Test creating a random subset."""
        ds = COCOPersonDataset(**mock_coco_dir)
        sub = ds.subset(num_images=2, seed=42)
        assert len(sub) == 2

    def test_get_image_ids(self, mock_coco_dir):
        """Test image IDs are returned sorted."""
        ds = COCOPersonDataset(**mock_coco_dir)
        ids = ds.get_image_ids()
        assert ids == sorted(ids)
        assert len(ids) == 3

    def test_missing_annotations_file(self, tmp_path):
        """Test error on missing annotations file."""
        with pytest.raises(FileNotFoundError):
            COCOPersonDataset(
                images_dir=str(tmp_path),
                annotations_file=str(tmp_path / "nonexistent.json"),
            )

    def test_index_out_of_range(self, mock_coco_dir):
        """Test IndexError on invalid index."""
        ds = COCOPersonDataset(**mock_coco_dir)
        with pytest.raises(IndexError):
            ds[100]


# ─── ECP Tests ────────────────────────────────────────────────────────

class TestECPDataset:
    """Tests for EuroCity Persons dataset loader."""

    def test_discover_files(self, mock_ecp_dir):
        """Test that files are discovered from the city directory structure."""
        ds = ECPDataset(**mock_ecp_dir)
        assert len(ds) == 4

    def test_annotations_loaded(self, mock_ecp_dir):
        """Test that annotations are properly parsed."""
        ds = ECPDataset(**mock_ecp_dir)
        frame = ds[0]

        assert isinstance(frame, StreamFrame)
        assert isinstance(frame.image, np.ndarray)

        targets = frame.targets
        assert 'boxes' in targets
        assert 'labels' in targets
        assert targets['boxes'].shape[0] == 2  # 2 pedestrians per frame

    def test_person_only_filter(self, mock_ecp_dir):
        """Test that non-pedestrian classes are filtered."""
        config = ECPConfig(person_only=True)
        ds = ECPDataset(**mock_ecp_dir, config=config)
        frame = ds[2]  # Frame 2 has pedestrians + rider
        # With person_only=True, rider should be excluded
        assert all(l == 0 for l in frame.targets['labels'])

    def test_include_riders(self, mock_ecp_dir):
        """Test that riders can be included."""
        config = ECPConfig(person_only=True, include_riders=True)
        ds = ECPDataset(**mock_ecp_dir, config=config)
        frame = ds[2]
        # Now should have 3 annotations (2 pedestrians + 1 rider)
        assert frame.targets['boxes'].shape[0] == 3

    def test_night_metadata(self, mock_ecp_dir):
        """Test that night metadata is correctly set."""
        config = ECPConfig(time_of_day="night")
        ds = ECPDataset(**mock_ecp_dir, config=config)
        frame = ds[0]

        assert frame.metadata.shift_active is True
        assert frame.metadata.shift_family == "illumination"
        assert frame.metadata.shift_severity == 0.8

    def test_day_metadata(self, mock_ecp_dir):
        """Test that day metadata shows no shift."""
        config = ECPConfig(time_of_day="day")
        ds = ECPDataset(**mock_ecp_dir, config=config)
        frame = ds[0]

        assert frame.metadata.shift_active is False
        assert frame.metadata.shift_family == "none"
        assert frame.metadata.shift_severity == 0.0

    def test_city_distribution(self, mock_ecp_dir):
        """Test city distribution tracking."""
        ds = ECPDataset(**mock_ecp_dir)
        dist = ds.get_city_distribution()
        assert 'berlin' in dist
        assert dist['berlin'] == 4

    def test_min_height_filter(self, mock_ecp_dir):
        """Test minimum height filtering."""
        # Create annotation with very small box
        config = ECPConfig(min_height=300)  # very high threshold
        ds = ECPDataset(**mock_ecp_dir, config=config)
        # Some pedestrians have height 250 (y1=200,y2=450), some 200 (y1=300,y2=500)
        # With min_height=300, only boxes taller than 300px should remain
        # box1: height=250, box2: height=200 → none remain
        frame = ds[0]
        assert frame.targets['boxes'].shape[0] == 0

    def test_empty_dir_warning(self, tmp_path):
        """Test graceful handling of missing directory."""
        ds = ECPDataset(
            images_dir=str(tmp_path / "nonexistent"),
            annotations_dir=str(tmp_path / "also_nonexistent"),
        )
        assert len(ds) == 0


# ─── NightOwls Tests ──────────────────────────────────────────────────

class TestNightOwlsDataset:
    """Tests for NightOwls dataset loader."""

    def test_load_annotations(self, mock_nightowls_dir):
        """Test that annotations are loaded and filtered."""
        ds = NightOwlsDataset(**mock_nightowls_dir)
        # person_only=True by default, so image_id=3 (cyclist only) excluded
        assert len(ds) == 5  # images 0,1,2,4,5 have pedestrians

    def test_include_cyclists(self, mock_nightowls_dir):
        """Test including cyclist annotations."""
        config = NightOwlsConfig(person_only=True, include_cyclists=True)
        ds = NightOwlsDataset(**mock_nightowls_dir, config=config)
        assert len(ds) == 6  # all 6 images now included

    def test_frame_format(self, mock_nightowls_dir):
        """Test that frames have correct format."""
        ds = NightOwlsDataset(**mock_nightowls_dir)
        frame = ds[0]

        assert isinstance(frame, StreamFrame)
        assert isinstance(frame.image, np.ndarray)
        assert frame.image.shape == (640, 1024, 3)  # NightOwls resolution

        targets = frame.targets
        assert 'boxes' in targets
        assert 'labels' in targets
        assert targets['boxes'].shape[1] == 4

    def test_always_night_shift(self, mock_nightowls_dir):
        """Test that NightOwls frames always have shift_active=True."""
        ds = NightOwlsDataset(**mock_nightowls_dir)
        for i in range(min(3, len(ds))):
            frame = ds[i]
            assert frame.metadata.shift_active is True
            assert frame.metadata.shift_family == "illumination_night"
            assert frame.metadata.shift_severity == 0.9

    def test_max_images(self, mock_nightowls_dir):
        """Test limiting number of loaded images."""
        config = NightOwlsConfig(max_images=3)
        ds = NightOwlsDataset(**mock_nightowls_dir, config=config)
        assert len(ds) == 3

    def test_stats(self, mock_nightowls_dir):
        """Test statistics computation."""
        ds = NightOwlsDataset(**mock_nightowls_dir)
        stats = ds.get_stats()
        assert stats['num_images'] == 5
        assert stats['num_instances'] > 0
        assert stats['split'] == "validation"

    def test_sequence_info(self, mock_nightowls_dir):
        """Test sequence information extraction."""
        ds = NightOwlsDataset(**mock_nightowls_dir)
        seq_info = ds.get_sequence_info()
        assert 'num_sequences' in seq_info

    def test_missing_annotations_warning(self, tmp_path):
        """Test graceful handling of missing annotations."""
        ds = NightOwlsDataset(
            images_dir=str(tmp_path),
            annotations_file=str(tmp_path / "nonexistent.json"),
        )
        assert len(ds) == 0

    def test_bbox_conversion(self, mock_nightowls_dir):
        """Test COCO [x,y,w,h] → [x1,y1,x2,y2] conversion."""
        ds = NightOwlsDataset(**mock_nightowls_dir)
        frame = ds[0]
        boxes = frame.targets['boxes']
        # First ann: [100, 100, 50, 120] → [100, 100, 150, 220]
        np.testing.assert_allclose(boxes[0], [100, 100, 150, 220])


# ─── Factory Function Tests ──────────────────────────────────────────

class TestDatasetFactory:
    """Tests for the create_dataset factory function."""

    def test_create_coco_person(self, mock_coco_dir):
        """Test factory creates COCO Person dataset."""
        ds = create_dataset("coco_person", **mock_coco_dir)
        assert isinstance(ds, COCOPersonDataset)
        assert len(ds) == 3

    def test_create_ecp_night(self, mock_ecp_dir):
        """Test factory creates ECP night dataset."""
        ds = create_dataset("ecp_night", **mock_ecp_dir)
        assert isinstance(ds, ECPDataset)
        frame = ds[0]
        assert frame.metadata.shift_active is True

    def test_create_ecp_day(self, mock_ecp_dir):
        """Test factory creates ECP day dataset."""
        ds = create_dataset("ecp_day", **mock_ecp_dir)
        assert isinstance(ds, ECPDataset)
        frame = ds[0]
        assert frame.metadata.shift_active is False

    def test_create_nightowls(self, mock_nightowls_dir):
        """Test factory creates NightOwls dataset."""
        ds = create_dataset("nightowls", **mock_nightowls_dir)
        assert isinstance(ds, NightOwlsDataset)

    def test_unknown_dataset(self):
        """Test error on unknown dataset name."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            create_dataset("unknown_dataset")

    def test_case_insensitive(self, mock_coco_dir):
        """Test factory is case-insensitive."""
        ds = create_dataset("COCO_PERSON", **mock_coco_dir)
        assert isinstance(ds, COCOPersonDataset)


# ─── StreamFrame Compatibility Tests ──────────────────────────────────

class TestStreamFrameCompatibility:
    """Test that all datasets produce compatible StreamFrames."""

    def _check_stream_frame(self, frame):
        """Validate a StreamFrame has all required fields."""
        assert isinstance(frame, StreamFrame)
        assert frame.image is not None
        assert isinstance(frame.image, np.ndarray)
        assert len(frame.image.shape) == 3  # H, W, C
        assert frame.image.shape[2] == 3     # RGB

        assert frame.targets is not None
        assert 'boxes' in frame.targets
        assert 'labels' in frame.targets
        assert frame.targets['boxes'].ndim == 2
        if frame.targets['boxes'].shape[0] > 0:
            assert frame.targets['boxes'].shape[1] == 4

        assert frame.metadata is not None
        assert isinstance(frame.metadata, FrameMetadata)
        assert hasattr(frame.metadata, 'stream_index')
        assert hasattr(frame.metadata, 'shift_active')
        assert hasattr(frame.metadata, 'shift_family')
        assert hasattr(frame.metadata, 'shift_severity')

    def test_coco_compatibility(self, mock_coco_dir):
        """Test COCO frames are StreamFrame-compatible."""
        ds = COCOPersonDataset(**mock_coco_dir)
        for i in range(len(ds)):
            self._check_stream_frame(ds[i])

    def test_ecp_compatibility(self, mock_ecp_dir):
        """Test ECP frames are StreamFrame-compatible."""
        ds = ECPDataset(**mock_ecp_dir)
        for i in range(len(ds)):
            self._check_stream_frame(ds[i])

    def test_nightowls_compatibility(self, mock_nightowls_dir):
        """Test NightOwls frames are StreamFrame-compatible."""
        ds = NightOwlsDataset(**mock_nightowls_dir)
        for i in range(len(ds)):
            self._check_stream_frame(ds[i])
