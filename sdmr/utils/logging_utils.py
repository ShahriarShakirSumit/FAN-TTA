"""
Logging utilities for SDMR experiments.

Provides structured logging with file + console output,
experiment-level log management, and timing context managers.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager


def setup_logger(
    name: str = "sdmr",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_prefix: str = "sdmr",
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_dir: Directory for log files. If None, no file logging.
        level: Logging level.
        console: Whether to log to console.
        file_prefix: Prefix for log file name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"{file_prefix}_{timestamp}.log"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "sdmr") -> logging.Logger:
    """Get an existing logger by name."""
    return logging.getLogger(name)


@dataclass
class ExperimentMeta:
    """Metadata for an experiment run."""
    experiment_name: str
    timestamp: str
    config_hash: str
    seed: int
    hardware: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        """Save metadata to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def create(
        cls,
        name: str,
        config_hash: str,
        seed: int,
        notes: str = "",
    ) -> "ExperimentMeta":
        """Create with auto-detected hardware info."""
        import torch
        hw = "cpu"
        if torch.cuda.is_available():
            hw = torch.cuda.get_device_name(0)
        return cls(
            experiment_name=name,
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            seed=seed,
            hardware=hw,
            notes=notes,
        )


@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for timing code blocks.

    Usage:
        with timer("forward pass"):
            model(x)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"[TIMER] {label}: {elapsed:.4f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


class JsonLogger:
    """
    Append-only JSON-lines logger for structured experiment data.
    Each line is a valid JSON object.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log(self, data: dict) -> None:
        """Append a JSON line."""
        data["_timestamp"] = datetime.now().isoformat()
        with open(self.filepath, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def read_all(self) -> list[dict]:
        """Read all logged entries."""
        if not self.filepath.exists():
            return []
        entries = []
        with open(self.filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
