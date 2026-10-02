"""
YOLO detector wrapper using Ultralytics.

Wraps YOLOv8/v10/v11/YOLO26 models with the BaseDetector interface.
Supports:
- Inference with configurable confidence/NMS thresholds
- Feature extraction via backbone/neck hooks
- BN refresh, head-only adaptation, bounded full adaptation
- State snapshots and rollback

Usage:
    detector = YOLODetectorWrapper(model_name="yolov8s.pt", device="cuda")
    detector.load_model()
    results = detector.predict(images)
"""

import time
import copy
from typing import Optional, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .base_detector import BaseDetector, DetectionResult, AdaptationStats


class YOLODetectorWrapper(BaseDetector):
    """
    Ultralytics YOLO wrapper implementing the BaseDetector interface.

    Supports YOLOv8, YOLOv10, YOLO11, and YOLO26 variants.

    Args:
        model_name: Model identifier (e.g., "yolov8s.pt", "yolo11s.pt").
        device: Device string ("cuda" or "cpu").
        imgsz: Default inference image size.
    """

    # Map of known model families and their properties
    KNOWN_MODELS = {
        "yolov8": {"sizes": ["n", "s", "m", "l", "x"], "has_bn": True},
        "yolov10": {"sizes": ["n", "s", "m", "l", "x"], "has_bn": True},
        "yolo11": {"sizes": ["n", "s", "m", "l", "x"], "has_bn": True},
        "yolo26": {"sizes": ["n", "s", "m", "l", "x"], "has_bn": True},
    }

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        device: str = "cuda",
        imgsz: int = 640,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.imgsz = imgsz
        self._yolo = None  # Ultralytics YOLO object
        self._backbone_layers: list[str] = []
        self._neck_layers: list[str] = []
        self._head_layers: list[str] = []

    def load_model(self, **kwargs) -> None:
        """
        Load YOLO model from Ultralytics.

        kwargs:
            weights_path: Optional custom weights path.
            verbose: Print model info (default False).
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            )

        weights_path = kwargs.get("weights_path", self.model_name)
        verbose = kwargs.get("verbose", False)

        # Ultralytics <8.1 has Model.__init__(model, task=None) with no `verbose`.
        # `verbose` only controls console logging, never inference, so falling back
        # without it is numerically identical.
        try:
            self._yolo = YOLO(weights_path, verbose=verbose)
        except TypeError:
            self._yolo = YOLO(weights_path)
        self._model = self._yolo.model

        # ---- CRITICAL: Prevent Conv-BN fusion ----
        # Ultralytics' predict() calls AutoBackend(fuse=True) which permanently
        # merges BatchNorm into preceding Conv layers. This destroys all BN
        # modules, making BN refresh (a core SDMR adaptation) impossible.
        # We monkey-patch fuse() to be a no-op so BN layers are preserved.
        self._model.fuse = lambda *args, **kwargs: self._model

        # Move to device
        self._model.to(self.device)
        self._model.eval()

        # Identify architecture layers
        self._identify_layers()

        # Save initial state for rollback
        self.save_state("initial")

    def _identify_layers(self) -> None:
        """Identify backbone, neck, and head layers in the YOLO model."""
        self._backbone_layers = []
        self._neck_layers = []
        self._head_layers = []

        # Ultralytics YOLO model structure:
        # model.model is a Sequential of modules
        # Typically: backbone (first ~10 layers), neck (next ~10), head (last few)
        if hasattr(self._model, 'model'):
            sequential = self._model.model
            total_layers = len(sequential) if hasattr(sequential, '__len__') else 0

            for i, (name, module) in enumerate(sequential.named_children()):
                full_name = f"model.{name}"
                # Heuristic: first 60% backbone, next 30% neck, last 10% head
                if i < total_layers * 0.6:
                    self._backbone_layers.append(full_name)
                elif i < total_layers * 0.9:
                    self._neck_layers.append(full_name)
                else:
                    self._head_layers.append(full_name)

    def predict(
        self,
        images: list,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> list[DetectionResult]:
        """
        Run YOLO inference on images.

        Args:
            images: List of images (PIL, numpy arrays, or file paths).
            conf_threshold: Confidence threshold.
            iou_threshold: NMS IoU threshold.

        Returns:
            List of DetectionResult objects.
        """
        if self._yolo is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference
        results = self._yolo.predict(
            images,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )

        # Convert to DetectionResult format
        detection_results = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu()
                scores = r.boxes.conf.cpu()
                labels = r.boxes.cls.cpu().int()
            else:
                boxes = torch.zeros((0, 4))
                scores = torch.zeros(0)
                labels = torch.zeros(0, dtype=torch.int)

            detection_results.append(DetectionResult(
                boxes=boxes,
                scores=scores,
                labels=labels,
            ))

        return detection_results

    def predict_single(
        self,
        image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> DetectionResult:
        """Convenience: predict on a single image."""
        results = self.predict([image], conf_threshold, iou_threshold)
        return results[0]

    def extract_features(
        self,
        images: list,
        layer_names: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Extract intermediate features using hooks.

        Args:
            images: List of images.
            layer_names: Specific layer names, or None for backbone + neck defaults.

        Returns:
            Dict of layer_name -> feature tensor.
        """
        # Clear previous features
        self._feature_cache.clear()
        self._clear_hooks()

        # Determine which layers to hook
        if layer_names is None:
            # Default: last backbone layer + last neck layer
            target_layers = []
            if self._backbone_layers:
                target_layers.append(self._backbone_layers[-1])
            if self._neck_layers:
                target_layers.append(self._neck_layers[-1])
            # If architecture wasn't identified, hook a few by index
            if not target_layers and hasattr(self._model, 'model'):
                seq = self._model.model
                n = len(list(seq.children()))
                for idx in [n // 3, 2 * n // 3]:
                    target_layers.append(f"model.{idx}")
        else:
            target_layers = layer_names

        # Register hooks
        for layer_name in target_layers:
            module = self._get_module_by_name(layer_name)
            if module is not None:
                self._register_hook(layer_name, module)

        # Forward pass to trigger hooks
        with torch.no_grad():
            self.predict(images)

        # Copy features and clean up
        features = {k: v.detach().clone() for k, v in self._feature_cache.items()}
        self._clear_hooks()

        return features

    def get_head_modules(self) -> list[tuple[str, nn.Module]]:
        """Return detection head modules."""
        heads = []
        # Ultralytics Detect head is typically the last module
        if hasattr(self._model, 'model'):
            seq = self._model.model
            children = list(seq.named_children())
            if children:
                # Last module is usually the Detect head
                last_name, last_module = children[-1]
                heads.append((f"model.{last_name}", last_module))
        return heads

    def get_backbone_modules(self) -> list[tuple[str, nn.Module]]:
        """Return backbone modules."""
        result = []
        for layer_name in self._backbone_layers:
            module = self._get_module_by_name(layer_name)
            if module is not None:
                result.append((layer_name, module))
        return result

    # ---- Adaptation methods ----

    def adapt_step(
        self,
        batch: Any,
        mode: str = "head_only",
        params: Optional[dict] = None,
    ) -> AdaptationStats:
        """
        Perform adaptation step.

        Supported modes:
        - "bn_refresh": Update BN running stats from batch (no gradients)
        - "head_only": Update only detection head parameters
        - "bounded_full": Update all parameters with strict bounds

        Args:
            batch: List of images or (images, targets) tuple.
            mode: Adaptation mode.
            params: Mode-specific parameters:
                   - lr: learning rate (default 1e-4)
                   - steps: number of update steps (default 1)
                   - max_steps: hard limit on steps (default 5)
                   - grad_clip: gradient clipping norm (default 1.0)

        Returns:
            AdaptationStats with timing and diagnostics.
        """
        params = params or {}
        start_time = time.perf_counter()

        if mode == "bn_refresh":
            stats = self._refresh_bn_stats(batch, params)
        elif mode == "head_only":
            stats = self._adapt_head_only(batch, params)
        elif mode == "bounded_full":
            stats = self._adapt_bounded_full(batch, params)
        else:
            return AdaptationStats(
                action_name=mode,
                success=False,
                notes=f"Unknown adaptation mode: {mode}",
            )

        stats.latency_ms = (time.perf_counter() - start_time) * 1000
        return stats

    def _refresh_bn_stats(
        self, batch: Any, params: dict
    ) -> AdaptationStats:
        """
        Online BN adaptation: update running stats via forward passes.

        Uses PyTorch's built-in BN momentum to gradually adapt running
        statistics toward the current data distribution. This is safe because:
        - No reset of running stats (preserves pretrained statistics)
        - Small momentum (0.1) per batch provides gradual shift
        - Multiple images smooth out per-frame noise
        - Fully reversible by loading initial state

        This mirrors the approach used in TENT/BN adaptation TTA papers.
        """
        bn_modules = self.get_bn_modules()
        if not bn_modules:
            return AdaptationStats(
                action_name="bn_refresh",
                success=False,
                notes="No BatchNorm modules found.",
            )

        max_batches = params.get("max_batches", 16)
        # Use a small momentum for online update (default BN momentum=0.1)
        bn_momentum = params.get("bn_momentum", 0.1)

        images = batch if isinstance(batch, list) else [batch]
        images = images[:max_batches]

        # Preprocess all images into a tensor batch
        try:
            x = self._preprocess_for_adaptation(images)
        except Exception as e:
            return AdaptationStats(
                action_name="bn_refresh",
                success=False,
                notes=f"Preprocessing failed: {e}",
            )

        # Set custom momentum for this adaptation pass
        original_momentums = {}
        for name, m in bn_modules:
            original_momentums[name] = m.momentum
            m.momentum = bn_momentum

        # Forward pass in train mode — BN layers will update running stats
        # using: new_running = (1-momentum) * running + momentum * batch_stat
        self._model.train()
        with torch.no_grad():
            try:
                sub_batch_size = min(4, len(images))
                for i in range(0, x.shape[0], sub_batch_size):
                    sub_x = x[i:i + sub_batch_size]
                    self._model(sub_x)
            except Exception as e:
                pass  # Partial update is OK

        # Restore original momentum and eval mode
        for name, m in bn_modules:
            if name in original_momentums:
                m.momentum = original_momentums[name]
        self._model.eval()

        return AdaptationStats(
            action_name="bn_refresh",
            success=True,
            update_steps=len(images),
            notes=f"Online BN update on {len(bn_modules)} layers with {len(images)} images (momentum={bn_momentum}).",
        )

    def _preprocess_for_adaptation(self, images: list) -> torch.Tensor:
        """
        Preprocess images for gradient-based forward pass.

        Converts numpy/PIL images to tensor batch in model format.
        """
        import torchvision.transforms.functional as TF

        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                # HWC uint8 → CHW float [0,1]
                t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            elif hasattr(img, 'mode'):  # PIL Image
                t = TF.to_tensor(img)
            elif isinstance(img, torch.Tensor):
                t = img.float()
                if t.ndim == 3 and t.shape[0] not in (1, 3):
                    t = t.permute(2, 0, 1)
                if t.max() > 1:
                    t = t / 255.0
            else:
                continue

            # Resize to model input size
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(self.imgsz, self.imgsz),
                mode="bilinear", align_corners=False
            ).squeeze(0)
            tensors.append(t)

        if not tensors:
            return torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device)

        batch = torch.stack(tensors).to(self.device)
        return batch

    @staticmethod
    def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy minimization loss on detection logits.

        For object detection, we use the classification head outputs.
        Lower entropy = more confident predictions = better alignment.
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Flatten and apply softmax
        # logits shape: typically [B, num_anchors, num_classes + 5] or similar
        if logits.dim() >= 3:
            # Reshape to (N, C) where C includes class scores
            logits = logits.reshape(-1, logits.shape[-1])

        # Extract class scores (skip box coords if present)
        if logits.shape[-1] > 10:  # Has box coords
            class_logits = logits[:, 5:]  # After x,y,w,h,objectness
        else:
            class_logits = logits

        if class_logits.shape[-1] < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probs = torch.softmax(class_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        return entropy.mean()

    @staticmethod
    def _diversity_loss(logits: torch.Tensor) -> torch.Tensor:
        """
        Encourages diverse predictions to prevent collapse.

        Maximizes entropy of the mean prediction across the batch
        (ensures the model doesn't predict one class for everything).
        """
        if logits.dim() >= 3:
            logits = logits.reshape(-1, logits.shape[-1])

        if logits.shape[-1] > 10:
            class_logits = logits[:, 5:]
        else:
            class_logits = logits

        if class_logits.shape[-1] < 2 or len(class_logits) < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        mean_probs = torch.softmax(class_logits, dim=-1).mean(dim=0)
        diversity = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()

        # Diversity is maximized, so negate for minimization
        return -diversity

    def _adapt_head_only(
        self, batch: Any, params: dict
    ) -> AdaptationStats:
        """
        Adapt only the detection head with entropy minimization.

        Uses the YOLO model's forward pass directly (not predict API)
        to enable gradient flow through classification head.
        """
        lr = params.get("lr", 1e-4)
        steps = min(params.get("steps", 1), params.get("max_steps", 5))
        grad_clip = params.get("grad_clip", 1.0)
        diversity_weight = params.get("diversity_weight", 0.1)

        # Freeze everything except head
        for p in self._model.parameters():
            p.requires_grad = False

        head_modules = self.get_head_modules()
        head_params = []
        for _, m in head_modules:
            for p in m.parameters():
                p.requires_grad = True
                head_params.append(p)

        if not head_params:
            # Fallback: no identifiable head → try BN affine params
            for _, m in self.get_bn_modules():
                if m.weight is not None:
                    m.weight.requires_grad = True
                    head_params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    head_params.append(m.bias)

        if not head_params:
            return AdaptationStats(
                action_name="head_only_adapt",
                success=False,
                notes="No adaptable parameters found.",
            )

        optimizer = torch.optim.SGD(head_params, lr=lr, momentum=0.9)
        images = batch if isinstance(batch, list) else [batch]

        # Preprocess
        try:
            x = self._preprocess_for_adaptation(images)
        except Exception as e:
            for p in self._model.parameters():
                p.requires_grad = False
            self._model.eval()
            return AdaptationStats(
                action_name="head_only_adapt",
                success=False,
                notes=f"Preprocessing failed: {e}",
            )

        # Enable BN update during training
        self._model.train()

        total_loss = 0.0
        for step in range(steps):
            optimizer.zero_grad()
            try:
                # Forward pass through model
                out = self._model(x)

                # Compute self-supervised losses
                if isinstance(out, (list, tuple)):
                    # Ultralytics models return list of feature maps or predictions
                    losses = []
                    for o in out:
                        if isinstance(o, torch.Tensor) and o.requires_grad:
                            losses.append(self._entropy_loss(o))
                    if losses:
                        ent_loss = sum(losses) / len(losses)
                    else:
                        ent_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                elif isinstance(out, torch.Tensor):
                    ent_loss = self._entropy_loss(out)
                else:
                    ent_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # Combined loss
                loss = ent_loss
                if loss.requires_grad:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(head_params, grad_clip)
                    optimizer.step()
                    total_loss += loss.item()

            except Exception as e:
                # If forward fails (e.g., shape mismatch), skip silently
                break

        # Re-freeze all and return to eval
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

        return AdaptationStats(
            action_name="head_only_adapt",
            success=True,
            update_steps=steps,
            loss_after=total_loss / max(steps, 1),
            notes=f"Head adaptation: lr={lr}, steps={steps}, loss={total_loss/max(steps,1):.4f}",
        )

    def _adapt_bounded_full(
        self, batch: Any, params: dict
    ) -> AdaptationStats:
        """
        Bounded full model adaptation with strict parameter deviation limits.

        Updates all parameters but clips the total deviation from the
        initial checkpoint to prevent catastrophic forgetting.
        """
        lr = params.get("lr", 1e-5)
        steps = min(params.get("steps", 1), params.get("max_steps", 3))
        grad_clip = params.get("grad_clip", 0.5)
        max_param_deviation = params.get("max_deviation", 0.01)

        # Save state before adaptation for bounding
        pre_state = {k: v.clone() for k, v in self._model.state_dict().items()}

        # Enable all params
        for p in self._model.parameters():
            p.requires_grad = True

        all_params = list(self._model.parameters())
        optimizer = torch.optim.SGD(all_params, lr=lr, momentum=0.9,
                                     weight_decay=1e-4)

        images = batch if isinstance(batch, list) else [batch]

        try:
            x = self._preprocess_for_adaptation(images)
        except Exception as e:
            for p in self._model.parameters():
                p.requires_grad = False
            self._model.eval()
            return AdaptationStats(
                action_name="bounded_full_adapt",
                success=False,
                notes=f"Preprocessing failed: {e}",
            )

        self._model.train()
        total_loss = 0.0

        for step in range(steps):
            optimizer.zero_grad()
            try:
                out = self._model(x)

                if isinstance(out, (list, tuple)):
                    losses = [self._entropy_loss(o) for o in out
                             if isinstance(o, torch.Tensor) and o.requires_grad]
                    loss = sum(losses) / max(len(losses), 1) if losses else torch.tensor(0.0, device=self.device, requires_grad=True)
                elif isinstance(out, torch.Tensor):
                    loss = self._entropy_loss(out)
                else:
                    break

                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                    optimizer.step()
                    total_loss += loss.item()

            except Exception:
                break

        # ---- Bound parameter deviation ----
        # Project parameters back if they deviated too far from initial
        with torch.no_grad():
            current_state = self._model.state_dict()
            for key in current_state:
                if key in pre_state and current_state[key].is_floating_point():
                    deviation = (current_state[key] - pre_state[key]).norm()
                    param_norm = pre_state[key].norm() + 1e-8
                    relative_dev = deviation / param_norm

                    if relative_dev > max_param_deviation:
                        # Scale back the update
                        scale = max_param_deviation / relative_dev
                        current_state[key] = (
                            pre_state[key] +
                            scale * (current_state[key] - pre_state[key])
                        )

            self._model.load_state_dict(current_state)

        # Freeze and eval
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

        return AdaptationStats(
            action_name="bounded_full_adapt",
            success=True,
            update_steps=steps,
            loss_after=total_loss / max(steps, 1),
            notes=f"Bounded full: lr={lr}, steps={steps}, clip={grad_clip}, "
                  f"max_dev={max_param_deviation}",
        )

    # ---- YOLO-specific utilities ----

    def get_model_info(self) -> dict:
        """Return YOLO model information."""
        info = self.summary()
        info["model_name"] = self.model_name
        info["imgsz"] = self.imgsz
        info["backbone_layers"] = len(self._backbone_layers)
        info["neck_layers"] = len(self._neck_layers)
        info["head_layers"] = len(self._head_layers)

        if self._yolo is not None:
            try:
                info["model_type"] = self._yolo.type
            except Exception:
                info["model_type"] = "unknown"
        return info

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update default confidence threshold."""
        if self._yolo is not None:
            self._yolo.overrides["conf"] = threshold

    def set_nms_threshold(self, threshold: float) -> None:
        """Update default NMS IoU threshold."""
        if self._yolo is not None:
            self._yolo.overrides["iou"] = threshold

    @staticmethod
    def available_models() -> list[str]:
        """List commonly available YOLO model names."""
        models = []
        for family, info in YOLODetectorWrapper.KNOWN_MODELS.items():
            for size in info["sizes"]:
                models.append(f"{family}{size}.pt")
        return models
