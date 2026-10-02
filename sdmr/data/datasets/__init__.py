"""
SDMR dataset wrappers.

Supported datasets:
- COCO Person (source domain): Person-class subset of COCO 2017
- EuroCity Persons (ECP): Day and night subsets for domain shift evaluation
- NightOwls: Nighttime-only pedestrian detection for extreme shift evaluation

Factory function:
    dataset = create_dataset("coco_person", images_dir="...", annotations_file="...")
"""

from .stream_wrapper import StreamWrapper, StreamFrame, StreamWindow, FrameMetadata, ShiftScenario, ShiftPattern
from .synthetic_shift_injector import SyntheticShiftInjector
from .coco_person import COCOPersonDataset, COCOPersonConfig
from .ecp_dataset import ECPDataset, ECPConfig
from .nightowls_dataset import NightOwlsDataset, NightOwlsConfig
from .cross_domain import CrossDomainDataset


def create_dataset(name: str, **kwargs):
    """
    Factory function to create a dataset by name.

    Args:
        name: One of "coco_person", "ecp", "ecp_day", "ecp_night", "nightowls"
        **kwargs: Passed to the dataset constructor

    Returns:
        Dataset object with __len__ and __getitem__ returning StreamFrame
    """
    name = name.lower().strip()

    if name == "coco_person":
        return COCOPersonDataset(
            images_dir=kwargs.get("images_dir", ""),
            annotations_file=kwargs.get("annotations_file", ""),
            config=kwargs.get("config"),
            transform=kwargs.get("transform"),
        )
    elif name in ("ecp", "ecp_day"):
        config = kwargs.get("config") or ECPConfig(time_of_day="day")
        return ECPDataset(
            images_dir=kwargs.get("images_dir", ""),
            annotations_dir=kwargs.get("annotations_dir", ""),
            config=config,
            transform=kwargs.get("transform"),
        )
    elif name == "ecp_night":
        config = kwargs.get("config") or ECPConfig(time_of_day="night")
        return ECPDataset(
            images_dir=kwargs.get("images_dir", ""),
            annotations_dir=kwargs.get("annotations_dir", ""),
            config=config,
            transform=kwargs.get("transform"),
        )
    elif name == "nightowls":
        return NightOwlsDataset(
            images_dir=kwargs.get("images_dir", ""),
            annotations_file=kwargs.get("annotations_file", ""),
            config=kwargs.get("config"),
            transform=kwargs.get("transform"),
        )
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Supported: coco_person, ecp, ecp_day, ecp_night, nightowls"
        )


__all__ = [
    "StreamWrapper", "StreamFrame", "StreamWindow", "FrameMetadata",
    "ShiftScenario", "ShiftPattern",
    "SyntheticShiftInjector",
    "COCOPersonDataset", "COCOPersonConfig",
    "ECPDataset", "ECPConfig",
    "NightOwlsDataset", "NightOwlsConfig",
    "CrossDomainDataset",
    "create_dataset",
]

