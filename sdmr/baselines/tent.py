"""
TENT: Fully Test-Time Adaptation by Entropy Minimization.
Adapted for object detection.

Reference: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy
Minimization", ICLR 2021 (Spotlight).

Original: Minimizes prediction entropy by updating BN affine parameters.
Detection adaptation: We minimize entropy of classification logits in the
detection head while updating BN affine parameters (γ, β).

Key insight: TENT was designed for classification. For detection, we
adapt it by computing entropy over the per-anchor class predictions
from the detection head, filtering high-confidence anchors.
"""

import copy
import logging
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TENTConfig:
    """Configuration for TENT baseline."""
    # Which parameters to update
    update_bn_only: bool = True         # True = original TENT (BN affine only)

    # Optimization
    learning_rate: float = 1e-3         # η for entropy minimization
    momentum_bn: float = 0.1            # BN running stats momentum

    # Entropy filtering
    min_confidence: float = 0.1         # min confidence to include anchor
    max_entropy_percentile: float = 0.66  # use bottom 66% entropy predictions

    # Steps per batch
    steps_per_batch: int = 1            # gradient steps per test batch

    # Reset
    episodic: bool = False              # reset to source after each batch


class TENTDetection:
    """
    TENT adapted for streaming object detection.

    Modifications from original TENT (classification):
    1. Entropy computed over detection head class logits (per anchor)
    2. High-confidence anchor filtering (low-conf = background noise)
    3. Compatible with YOLO-family single-stage detectors
    4. Streaming mode: no reset between frames (continual adaptation)

    Usage:
        tent = TENTDetection(model, config=TENTConfig())
        tent.setup()
        for image in stream:
            output = tent.forward_and_adapt(image)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TENTConfig] = None,
    ):
        self.config = config or TENTConfig()
        self.model = model
        self._source_state = None
        self._optimizer = None
        self._step_count = 0

    def setup(self):
        """
        Prepare model for TENT: freeze everything except BN affine params.
        Store source state for potential reset.
        """
        # Save source model state
        self._source_state = copy.deepcopy(self.model.state_dict())

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze BN affine parameters (γ and β)
        # NOTE: We keep track_running_stats=True (use pre-trained running stats)
        # because with batch_size=1 in streaming detection, batch stats from a
        # single image are unreliable and destroy prediction quality.
        # We use eval mode + torch.enable_grad() for adaptation instead.
        bn_params = []
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.requires_grad_(True)
                # Keep running stats for eval-mode forward pass
                module.track_running_stats = True
                # Collect affine params
                if module.weight is not None:
                    module.weight.requires_grad = True
                    bn_params.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad = True
                    bn_params.append(module.bias)

        if not bn_params:
            logger.warning("TENT: No BatchNorm layers found in model")

        # Setup optimizer for BN params only
        self._optimizer = torch.optim.Adam(bn_params, lr=self.config.learning_rate)

        logger.info(f"TENT setup: {len(bn_params)} BN parameters to optimize")

    def forward_and_adapt(self, images: torch.Tensor) -> dict:
        """
        Forward pass with entropy-based adaptation.

        Args:
            images: batch of images, shape (B, C, H, W)

        Returns:
            detection output dict
        """
        if self.config.episodic and self._source_state is not None:
            self.model.load_state_dict(self._source_state)

        # Adaptation steps
        for _ in range(self.config.steps_per_batch):
            self._adaptation_step(images)

        # Final forward pass (no grad)
        self.model.eval()
        with torch.no_grad():
            output = self.model(images)

        self._step_count += 1
        return output

    def _adaptation_step(self, images: torch.Tensor):
        """Single TENT adaptation step: minimize prediction entropy.
        
        Uses eval mode (running stats) with torch.enable_grad() to compute
        gradients through BN affine params. This is necessary because with
        batch_size=1, train mode batch stats destroy detection predictions.
        """
        self.model.eval()  # Use running stats for reliable predictions
        self._optimizer.zero_grad()

        # Forward pass with grad enabled in eval mode
        with torch.enable_grad():
            output = self.model(images)

            # Compute entropy loss over detection predictions
            entropy = self._compute_detection_entropy(output)

            if entropy is not None and entropy.requires_grad:
                entropy.backward()
                self._optimizer.step()

    def _compute_detection_entropy(self, output) -> Optional[torch.Tensor]:
        """
        Compute entropy of detection class predictions for YOLO.

        YOLOv8 output: (B, 4+num_classes, num_anchors) where class predictions
        are already sigmoid probabilities (independent binary classifiers).
        We compute binary entropy for each class and average:
            H = -Σ_c [p_c·log(p_c) + (1-p_c)·log(1-p_c)]
        """
        # Handle YOLO output format
        if isinstance(output, (list, tuple)):
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                preds = output[0]
            else:
                return None
        elif isinstance(output, torch.Tensor):
            preds = output
        else:
            return None

        if preds.dim() != 3 or preds.shape[1] <= 4:
            return None

        # (B, 4+C, N) → (B*N, C): class probs (already sigmoid)
        preds_t = preds.permute(0, 2, 1)  # (B, N, 4+C)
        class_probs = preds_t[:, :, 4:]   # (B, N, C)
        class_probs = class_probs.reshape(-1, class_probs.shape[-1])  # (B*N, C)

        # Filter by max confidence (no double sigmoid — values are already probs)
        max_conf = class_probs.max(dim=-1).values
        mask = max_conf > self.config.min_confidence
        if mask.sum() == 0:
            return None

        reliable_probs = class_probs[mask]  # (M, C)
        p = reliable_probs.clamp(1e-6, 1 - 1e-6)

        # Binary cross-entropy style entropy for independent classifiers
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).sum(dim=-1)

        # Use bottom percentile (most confident = lowest entropy)
        if len(entropy) > 1:
            threshold = torch.quantile(entropy, self.config.max_entropy_percentile)
            reliable = entropy[entropy <= threshold]
            if len(reliable) == 0:
                reliable = entropy
        else:
            reliable = entropy

        return reliable.mean()

    def reset(self):
        """Reset model to source state."""
        if self._source_state is not None:
            self.model.load_state_dict(self._source_state)
            self.setup()
        self._step_count = 0

    @property
    def name(self) -> str:
        return "tent_detection"

    def get_stats(self) -> dict:
        return {
            'method': 'TENT',
            'steps': self._step_count,
            'lr': self.config.learning_rate,
            'bn_only': self.config.update_bn_only,
        }
