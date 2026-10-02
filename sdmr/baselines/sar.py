"""
SAR: Sharpness-Aware and Reliable entropy minimization.
Adapted for object detection.

Reference: Niu et al., "Towards Stable Test-Time Adaptation in
Dynamic Wild World", ICLR 2023 (Notable-Top-5%).

Key ideas:
1. Sharpness-Aware Minimization (SAM): encourages flat loss
   minima during adaptation, improving robustness to domain shift
2. Reliable entropy minimization: filters out unreliable samples
   that would degrade adaptation through noisy gradients
3. Recovery mechanism: detects model collapse and resets to source

Detection adaptation: We apply SAM to BN parameters using detection
anchor entropy; filter unreliable anchors; detect detector collapse
via entropy monitoring.
"""

import copy
import logging
from typing import Optional
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SARConfig:
    """Configuration for SAR baseline."""
    # Sharpness-Aware Minimization
    sam_rho: float = 0.05              # perturbation radius ρ for SAM
    sam_adaptive: bool = True           # use adaptive SAM

    # Reliable entropy filtering
    entropy_margin: float = 0.4 * np.log(80)  # margin for reliable samples
    # Default uses 0.4 * ln(num_classes), 80 = COCO classes

    # Recovery mechanism
    recovery_window: int = 50          # window to monitor model collapse
    collapse_threshold: float = 0.9    # entropy ratio threshold for collapse

    # Optimization
    learning_rate: float = 1e-3
    steps_per_batch: int = 1

    # Detection anchor confidence filtering
    min_confidence: float = 0.1          # min confidence to include anchor


class SARDetection:
    """
    SAR adapted for streaming object detection.

    Combines three mechanisms for stable TTA:
    1. SAM: finds flat minima resistant to domain shifts, computed via
       a two-step gradient (ascent + descent) on BN parameters
    2. Reliable entropy: only uses anchors where H(y) < margin E_0
       to avoid corrupting the model with noisy gradients
    3. Collapse recovery: monitors average entropy and resets if
       the model is detecting collapse (all predictions become uniform)

    This is particularly important for detection where the large
    number of anchors amplifies the effect of noisy samples.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[SARConfig] = None,
    ):
        self.config = config or SARConfig()
        self.model = model
        self._source_state = None
        self._optimizer = None
        self._bn_params = []
        self._step_count = 0
        self._recovery_count = 0
        self._entropy_history = deque(maxlen=self.config.recovery_window)

    def setup(self):
        """Prepare model for SAR adaptation."""
        # Save source state for recovery
        self._source_state = copy.deepcopy(self.model.state_dict())

        # Collect BN parameters
        self._bn_params = []
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.requires_grad_(True)
                # Keep track_running_stats=True (default) — with batch_size=1,
                # train-mode batch stats are unreliable and destroy predictions.
                # We use eval mode + enable_grad() for adaptation instead.
                module.track_running_stats = True
                if module.weight is not None:
                    module.weight.requires_grad = True
                    self._bn_params.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad = True
                    self._bn_params.append(module.bias)

        # Freeze non-BN
        for param in self.model.parameters():
            if not any(param is bp for bp in self._bn_params):
                param.requires_grad = False

        # Base optimizer (SAM wraps this)
        self._optimizer = torch.optim.SGD(
            self._bn_params,
            lr=self.config.learning_rate,
            momentum=0.9,
        )

        logger.info(f"SAR setup: {len(self._bn_params)} BN params, "
                     f"SAM ρ={self.config.sam_rho}")

    def forward_and_adapt(self, images: torch.Tensor) -> dict:
        """Forward with SAR: SAM + reliable entropy + collapse recovery."""
        # Check for collapse first
        if self._detect_collapse():
            self._recover()
            self._recovery_count += 1
            logger.warning(f"SAR: model collapse detected at step {self._step_count}, "
                           f"recovery #{self._recovery_count}")

        # SAM-based adaptation
        for _ in range(self.config.steps_per_batch):
            self._sam_adaptation_step(images)

        # Final prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(images)

        # Track entropy for collapse detection
        mean_entropy = self._compute_mean_entropy(output)
        self._entropy_history.append(mean_entropy)

        self._step_count += 1
        return output

    def _sam_adaptation_step(self, images: torch.Tensor):
        """
        Sharpness-Aware Minimization step:
        1. Compute gradient of reliable entropy loss
        2. Ascend: ε = ρ · grad / ||grad||  (find worst-case perturbation)
        3. Descend: compute loss at perturbed point, update params

        This finds parameters in flat regions of the loss landscape,
        which are more robust to distribution shift.

        Uses eval mode (pre-trained BN running stats) — gradients still flow
        in eval mode, enabling adaptation of BN affine params. This is critical
        for batch_size=1 streaming detection where train-mode batch stats
        from a single image are unreliable.
        """
        self.model.eval()

        # ---- Step 1: First forward-backward for ascent direction ----
        self._optimizer.zero_grad()
        with torch.enable_grad():
            output = self.model(images)
            loss = self._reliable_entropy_loss(output)

            if loss is None or not loss.requires_grad:
                return

            loss.backward()

        # ---- Step 2: SAM perturbation (ascent) ----
        # ε_i = ρ · g_i / ||g||
        grad_norm = self._param_grad_norm()
        if grad_norm < 1e-12:
            return

        # Save original params and apply perturbation
        old_params = {}
        for param in self._bn_params:
            old_params[id(param)] = param.data.clone()
            if param.grad is not None:
                if self.config.sam_adaptive:
                    # Adaptive SAM: normalize per-parameter
                    eps = self.config.sam_rho * param.grad / (
                        param.grad.norm() + 1e-12
                    )
                else:
                    eps = self.config.sam_rho * param.grad / grad_norm
                param.data.add_(eps)

        # ---- Step 3: Second forward-backward at perturbed point ----
        self._optimizer.zero_grad()
        with torch.enable_grad():
            output_perturbed = self.model(images)
            loss_perturbed = self._reliable_entropy_loss(output_perturbed)

            if loss_perturbed is not None and loss_perturbed.requires_grad:
                loss_perturbed.backward()

        # ---- Step 4: Restore original params and step ----
        for param in self._bn_params:
            param.data = old_params[id(param)]

        self._optimizer.step()

    def _reliable_entropy_loss(self, output) -> Optional[torch.Tensor]:
        """
        Compute entropy loss using only reliable predictions.

        Filter: keep anchors where H(y) < E_0 (margin).
        This removes uncertain/wrong detections that would
        push the model in harmful directions.
        """
        entropy = self._extract_entropy(output)
        if entropy is None or len(entropy) == 0:
            return None

        # Reliable sample selection
        reliable_mask = entropy < self.config.entropy_margin
        if reliable_mask.sum() == 0:
            return None

        return entropy[reliable_mask].mean()

    def _extract_entropy(self, output) -> Optional[torch.Tensor]:
        """Extract per-anchor entropy from detection output.

        For YOLO-style detectors, the output is (B, 4+num_classes, num_anchors).
        Class values are already sigmoid probabilities (independent binary
        classifiers). We compute binary entropy per class:
            H = -Σ_c [p_c·log(p_c) + (1-p_c)·log(1-p_c)]
        """
        # Handle YOLO output format: tuple/list where first element is predictions
        if isinstance(output, (list, tuple)):
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                preds = output[0]  # (B, 4+num_classes, num_anchors)
            else:
                return None
        elif isinstance(output, torch.Tensor):
            preds = output
        elif isinstance(output, dict):
            logits = output.get('class_logits', output.get('cls', None))
            if logits is None:
                return None
            if logits.dim() > 2:
                logits = logits.reshape(-1, logits.shape[-1])
            if logits.shape[-1] <= 1:
                return None
            probs = F.softmax(logits, dim=-1)
            return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        else:
            return None

        # YOLO format: (B, 4+C, N) where C=num_classes, N=num_anchors
        if preds.dim() == 3 and preds.shape[1] > 4:
            # Transpose to (B, N, 4+C) then extract class probabilities
            preds_t = preds.permute(0, 2, 1)  # (B, N, 4+C)
            class_probs = preds_t[:, :, 4:]  # (B, N, C) — already sigmoid probabilities
            class_probs = class_probs.reshape(-1, class_probs.shape[-1])  # (B*N, C)

            if class_probs.shape[-1] <= 1:
                return None

            # Filter by confidence (values already in [0,1], no sigmoid needed)
            max_conf = class_probs.max(dim=-1).values
            mask = max_conf > self.config.min_confidence
            if mask.sum() == 0:
                return None

            reliable_probs = class_probs[mask]  # (M, C)

            # Binary entropy: H = -Σ_c [p·log(p) + (1-p)·log(1-p)]
            p = reliable_probs.clamp(1e-7, 1 - 1e-7)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).sum(dim=-1)
            return entropy
        else:
            return None

    def _compute_mean_entropy(self, output) -> float:
        """Compute mean entropy for collapse detection."""
        entropy = self._extract_entropy(output)
        if entropy is not None and len(entropy) > 0:
            return float(entropy.mean().item())
        return 0.0

    def _param_grad_norm(self) -> float:
        """Compute gradient L2 norm across all BN parameters."""
        total = 0.0
        for param in self._bn_params:
            if param.grad is not None:
                total += param.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def _detect_collapse(self) -> bool:
        """
        Detect model collapse: if average entropy approaches
        maximum entropy (uniform predictions), the model has collapsed.
        """
        if len(self._entropy_history) < self.config.recovery_window:
            return False

        recent = list(self._entropy_history)
        mean_entropy = np.mean(recent[-10:])  # recent 10 frames
        max_entropy = np.log(80)  # ln(num_classes) for COCO

        if max_entropy > 0:
            ratio = mean_entropy / max_entropy
            return ratio > self.config.collapse_threshold

        return False

    def _recover(self):
        """Reset model to source state (collapse recovery)."""
        if self._source_state is not None:
            self.model.load_state_dict(self._source_state)
            self.setup()

    def reset(self):
        """Full reset."""
        if self._source_state is not None:
            self.model.load_state_dict(self._source_state)
            self.setup()
        self._step_count = 0
        self._recovery_count = 0
        self._entropy_history.clear()

    @property
    def name(self) -> str:
        return "sar_detection"

    def get_stats(self) -> dict:
        return {
            'method': 'SAR',
            'total_steps': self._step_count,
            'recovery_count': self._recovery_count,
            'sam_rho': self.config.sam_rho,
            'mean_entropy': float(np.mean(list(self._entropy_history)))
                if self._entropy_history else 0.0,
        }
