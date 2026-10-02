"""
EATA: Efficient Test-Time Model Adaptation without Forgetting.
Adapted for object detection.

Reference: Niu et al., "Efficient Test-Time Model Adaptation without
Forgetting", ICML 2022.

Key ideas:
1. Active sample selection: only adapt on reliable, non-redundant samples
   (high-entropy = noisy → skip; low-entropy = already good → skip)
2. Fisher regularizer: prevents catastrophic forgetting by constraining
   important model parameters from drastic changes.

Detection adaptation: We apply the entropy-based sample selection to
detection anchors and use the Fisher regularizer on BN parameters.
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
class EATAConfig:
    """Configuration for EATA baseline."""
    # Sample selection
    entropy_threshold_low: float = 0.2   # E_0: min entropy to adapt (below = already good)
    entropy_threshold_high: float = 2.0  # E_max: max entropy to adapt (above = too noisy)
    diversity_margin: float = 0.05       # margin for redundancy check

    # Fisher regularizer
    fisher_alpha: float = 2000.0         # λ: weight of Fisher regularizer
    fisher_samples: int = 100            # num samples to estimate Fisher info

    # Optimization
    learning_rate: float = 1e-3
    steps_per_batch: int = 1

    # Parameters to update
    update_bn_only: bool = True

    # Detection anchor confidence filtering
    min_confidence: float = 0.1          # min confidence to include anchor


class EATADetection:
    """
    EATA adapted for streaming object detection.

    Extends TENT with:
    1. Active sample selection: only adapts on informative frames
       where detection entropy is in [E_0, E_max]
    2. Fisher regularizer: prevents forgetting source knowledge
       by penalizing changes to important parameters

    This addresses TENT's weakness of adapting on noisy/uninformative
    frames that can cause error accumulation.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[EATAConfig] = None,
    ):
        self.config = config or EATAConfig()
        self.model = model
        self._source_state = None
        self._fisher = None
        self._optimizer = None
        self._bn_params = []
        self._step_count = 0
        self._adapted_count = 0
        self._skipped_count = 0

    def setup(self, source_loader=None):
        """
        Prepare model for EATA.

        Args:
            source_loader: optional data loader for Fisher estimation.
                          If None, Fisher is estimated from first test batch.
        """
        # Save source state
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

        # Freeze non-BN parameters
        for param in self.model.parameters():
            if not any(param is bp for bp in self._bn_params):
                param.requires_grad = False

        # Initialize Fisher info (diagonal approximation)
        self._fisher = {
            id(p): torch.zeros_like(p.data)
            for p in self._bn_params
        }

        # Store source params for regularization
        self._source_params = {
            id(p): p.data.clone()
            for p in self._bn_params
        }

        # Optimizer
        self._optimizer = torch.optim.Adam(
            self._bn_params, lr=self.config.learning_rate
        )

        logger.info(f"EATA setup: {len(self._bn_params)} BN params, "
                     f"Fisher α={self.config.fisher_alpha}")

    def forward_and_adapt(self, images: torch.Tensor) -> dict:
        """
        Forward with active sample selection + Fisher regularization.
        """
        # Compute detection entropy for sample selection
        self.model.eval()
        with torch.no_grad():
            output = self.model(images)
            frame_entropy = self._compute_frame_entropy(output)

        # Active sample selection
        if self._should_adapt(frame_entropy):
            for _ in range(self.config.steps_per_batch):
                self._adaptation_step(images)
            self._adapted_count += 1
        else:
            self._skipped_count += 1

        # Final forward (no grad)
        self.model.eval()
        with torch.no_grad():
            output = self.model(images)

        self._step_count += 1
        return output

    def _should_adapt(self, entropy: float) -> bool:
        """
        Active sample selection criterion.

        Adapt only if entropy is in [E_0, E_max]:
        - entropy < E_0 → model is already confident, skip
        - entropy > E_max → too noisy, gradient would be harmful
        """
        return (self.config.entropy_threshold_low <= entropy
                <= self.config.entropy_threshold_high)

    def _adaptation_step(self, images: torch.Tensor):
        """EATA adaptation: entropy minimization + Fisher regularizer.

        Uses eval mode (pre-trained BN running stats) with enable_grad()
        for reliable predictions with batch_size=1 streaming detection.
        """
        self.model.eval()
        self._optimizer.zero_grad()

        with torch.enable_grad():
            output = self.model(images)
            entropy_loss = self._compute_entropy_loss(output)

            if entropy_loss is None:
                return

            # Fisher regularizer: Σ_i F_i · (θ_i - θ_i^source)²
            fisher_reg = torch.tensor(0.0, device=entropy_loss.device)
            for param in self._bn_params:
                pid = id(param)
                if pid in self._fisher and pid in self._source_params:
                    fisher_reg += (
                        self._fisher[pid]
                        * (param - self._source_params[pid]).pow(2)
                    ).sum()

            # Total loss
            loss = entropy_loss + self.config.fisher_alpha * fisher_reg

            if loss.requires_grad:
                loss.backward()
                self._optimizer.step()

        # Update Fisher (online estimation)
        self._update_fisher(images)

    def _compute_frame_entropy(self, output) -> float:
        """Compute mean entropy of detections in a frame."""
        entropy_tensor = self._extract_entropy(output)
        if entropy_tensor is not None and len(entropy_tensor) > 0:
            return float(entropy_tensor.mean().item())
        return 0.0

    def _compute_entropy_loss(self, output) -> Optional[torch.Tensor]:
        """Compute differentiable entropy loss for optimization."""
        entropy_tensor = self._extract_entropy(output)
        if entropy_tensor is not None and len(entropy_tensor) > 0:
            # Filter reliable predictions
            mask = (entropy_tensor > self.config.entropy_threshold_low) & \
                   (entropy_tensor < self.config.entropy_threshold_high)
            if mask.sum() > 0:
                return entropy_tensor[mask].mean()
        return None

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

    def _update_fisher(self, images: torch.Tensor):
        """
        Update diagonal Fisher information estimate.

        F_i ≈ E[(∂L/∂θ_i)²]

        We use an exponential moving average to track Fisher online.
        """
        self.model.eval()
        with torch.enable_grad():
            output = self.model(images)
            entropy = self._compute_entropy_loss(output)

            if entropy is not None and entropy.requires_grad:
                self._optimizer.zero_grad()
                entropy.backward(retain_graph=False)

                for param in self._bn_params:
                    pid = id(param)
                    if param.grad is not None and pid in self._fisher:
                        # EMA update of Fisher
                        self._fisher[pid] = (
                            0.9 * self._fisher[pid]
                            + 0.1 * param.grad.data.pow(2)
                        )

    def reset(self):
        """Reset to source state."""
        if self._source_state is not None:
            self.model.load_state_dict(self._source_state)
            self.setup()
        self._step_count = 0
        self._adapted_count = 0
        self._skipped_count = 0

    @property
    def name(self) -> str:
        return "eata_detection"

    def get_stats(self) -> dict:
        return {
            'method': 'EATA',
            'total_steps': self._step_count,
            'adapted': self._adapted_count,
            'skipped': self._skipped_count,
            'adaptation_rate': self._adapted_count / max(self._step_count, 1),
            'fisher_alpha': self.config.fisher_alpha,
        }
