"""
Seed management for reproducibility.

Usage:
    from sdmr.utils.seed import set_global_seed, get_seed_info
    set_global_seed(42)
"""

import os
import random
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class SeedInfo:
    """Records seed state for reproducibility logging."""
    seed: int
    python_hash_seed: str
    torch_deterministic: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "python_hash_seed": self.python_hash_seed,
            "torch_deterministic": self.torch_deterministic,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "timestamp": self.timestamp,
        }


def set_global_seed(seed: int, deterministic: bool = True) -> SeedInfo:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enforce deterministic CUDA operations
                      (slower but reproducible).

    Returns:
        SeedInfo with recorded state.
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 2.x deterministic mode
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    from datetime import datetime

    info = SeedInfo(
        seed=seed,
        python_hash_seed=str(seed),
        torch_deterministic=torch.backends.cudnn.deterministic,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        timestamp=datetime.now().isoformat(),
    )
    return info


def get_seed_info() -> dict:
    """Return current seed/determinism state (best effort)."""
    return {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not set"),
    }


def config_hash(config: dict) -> str:
    """
    Compute a deterministic hash of a config dict.
    Useful for tagging experiment outputs.
    """
    import json
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]
