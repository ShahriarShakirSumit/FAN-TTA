"""
Base detector wrapper interface for SDMR.

Defines the abstract interface that all detector wrappers must implement.
Supports:
- Inference (predict)
- Feature extraction (hooks)
- Adaptation (bounded, reversible)
- State management (snapshots, rollback)

All detector wrappers must subclass BaseDetector.
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import torch
import torch.nn as nn


@dataclass
class DetectionResult:
    """Standard detection output format."""
    boxes: torch.Tensor          # [N, 4] xyxy format
    scores: torch.Tensor         # [N] confidence scores
    labels: torch.Tensor         # [N] class labels
    num_detections: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.num_detections = len(self.boxes) if self.boxes is not None else 0

    def filter_by_score(self, threshold: float) -> "DetectionResult":
        """Return detections above a confidence threshold."""
        if self.num_detections == 0:
            return self
        mask = self.scores >= threshold
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            labels=self.labels[mask],
        )

    def to_dict(self) -> dict:
        return {
            "boxes": self.boxes.cpu().numpy().tolist() if self.boxes is not None else [],
            "scores": self.scores.cpu().numpy().tolist() if self.scores is not None else [],
            "labels": self.labels.cpu().numpy().tolist() if self.labels is not None else [],
            "num_detections": self.num_detections,
        }


@dataclass
class AdaptationStats:
    """Statistics from an adaptation step."""
    action_name: str
    success: bool
    latency_ms: float = 0.0
    update_steps: int = 0
    memory_delta_mb: float = 0.0
    loss_before: float = 0.0
    loss_after: float = 0.0
    state_snapshot_id: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "action_name": self.action_name,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "update_steps": self.update_steps,
            "memory_delta_mb": self.memory_delta_mb,
            "loss_before": self.loss_before,
            "loss_after": self.loss_after,
            "state_snapshot_id": self.state_snapshot_id,
            "notes": self.notes,
        }


class BaseDetector(ABC):
    """
    Abstract base class for detector wrappers in SDMR.

    All methods that modify model state must be reversible through
    the snapshot/restore mechanism.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model: Optional[nn.Module] = None
        self._state_snapshots: dict[str, dict] = {}
        self._feature_hooks: dict[str, Any] = {}
        self._feature_cache: dict[str, torch.Tensor] = {}
        self._hook_handles: list = []

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """Load the detector model and move to device."""
        ...

    @abstractmethod
    def predict(
        self,
        images: list,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> list[DetectionResult]:
        """
        Run inference on a list of images.

        Args:
            images: List of images (PIL, numpy, or paths).
            conf_threshold: Confidence threshold for filtering.
            iou_threshold: NMS IoU threshold.

        Returns:
            List of DetectionResult, one per image.
        """
        ...

    @abstractmethod
    def extract_features(
        self,
        images: list,
        layer_names: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Extract intermediate features from specified layers.

        Args:
            images: List of images.
            layer_names: Names of layers to extract from.
                        If None, use default backbone/neck layers.

        Returns:
            Dict mapping layer names to feature tensors.
        """
        ...

    def adapt_step(
        self,
        batch: Any,
        mode: str = "head_only",
        params: Optional[dict] = None,
    ) -> AdaptationStats:
        """
        Perform one adaptation step.

        Args:
            batch: Input data for adaptation.
            mode: Adaptation mode ("bn_refresh", "head_only", "bounded_full").
            params: Mode-specific parameters.

        Returns:
            AdaptationStats with timing and diagnostics.
        """
        raise NotImplementedError(
            f"Adaptation mode '{mode}' not implemented for {self.__class__.__name__}"
        )

    # ---- State management ----

    def save_state(self, snapshot_id: Optional[str] = None) -> str:
        """
        Save current model state dict to memory.

        Args:
            snapshot_id: Optional name for the snapshot. Auto-generated if None.

        Returns:
            The snapshot ID.
        """
        if snapshot_id is None:
            snapshot_id = f"snapshot_{len(self._state_snapshots)}"
        self._state_snapshots[snapshot_id] = copy.deepcopy(
            self.model.state_dict()
        )
        return snapshot_id

    def load_state(self, snapshot_id: str) -> None:
        """Restore model state from a saved snapshot."""
        if snapshot_id not in self._state_snapshots:
            raise KeyError(f"Snapshot '{snapshot_id}' not found. "
                          f"Available: {list(self._state_snapshots.keys())}")
        # Use strict=False to handle Conv+BN fusion mismatches
        # (Ultralytics may fuse Conv+BN at load, changing state dict keys)
        self.model.load_state_dict(
            copy.deepcopy(self._state_snapshots[snapshot_id]),
            strict=False,
        )

    def reset_to_checkpoint(self, snapshot_id: str = "initial") -> None:
        """Reset model to a named checkpoint."""
        self.load_state(snapshot_id)

    def get_bn_modules(self) -> list[tuple[str, nn.BatchNorm2d]]:
        """Return all BatchNorm2d modules with their names."""
        bn_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_modules.append((name, module))
        return bn_modules

    def get_head_modules(self) -> list[tuple[str, nn.Module]]:
        """Return the detection head modules. Must be overridden per detector."""
        raise NotImplementedError("Subclass must identify head modules.")

    def get_trainable_submodules(
        self, mode: str = "head_only"
    ) -> list[nn.Parameter]:
        """
        Get parameters that should be trainable for a given adaptation mode.

        Args:
            mode: "head_only", "bn_affine", "bounded_full", "all"

        Returns:
            List of trainable parameters.
        """
        if mode == "all":
            return list(self.model.parameters())
        elif mode == "bn_affine":
            params = []
            for _, m in self.get_bn_modules():
                if m.weight is not None:
                    params.append(m.weight)
                if m.bias is not None:
                    params.append(m.bias)
            return params
        elif mode == "head_only":
            params = []
            for _, m in self.get_head_modules():
                params.extend(m.parameters())
            return params
        elif mode == "bounded_full":
            return list(self.model.parameters())
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def supports_action(self, action_name: str) -> bool:
        """Check if this detector supports a given repair action."""
        supported = {
            "no_op": True,
            "recalibrate_output": True,
            "refresh_bn_stats": len(self.get_bn_modules()) > 0,
            "head_only_adapt": True,
            "bounded_full_adapt": True,
            "rollback_safe_mode": True,
        }
        return supported.get(action_name, False)

    # ---- Feature hook management ----

    def _register_hook(self, layer_name: str, module: nn.Module) -> None:
        """Register a forward hook on a module to capture features."""
        def hook_fn(mod, inp, out, name=layer_name):
            self._feature_cache[name] = out

        handle = module.register_forward_hook(hook_fn)
        self._hook_handles.append(handle)

    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._feature_cache.clear()

    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get a module by its dotted name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    # ---- Utility ----

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def to_eval(self) -> None:
        """Set model to eval mode."""
        self.model.eval()

    def to_train(self) -> None:
        """Set model to train mode."""
        self.model.train()

    def summary(self) -> dict:
        """Return model summary info."""
        return {
            "class": self.__class__.__name__,
            "device": str(self.device),
            "total_params": self.num_parameters(),
            "trainable_params": self.num_parameters(trainable_only=True),
            "bn_modules": len(self.get_bn_modules()),
            "snapshots": list(self._state_snapshots.keys()),
        }
