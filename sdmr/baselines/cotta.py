"""
CoTTA: Continual Test-Time Domain Adaptation.
Adapted for object detection.

Reference: Wang et al., "Continual Test-Time Domain Adaptation",
CVPR 2022.

Key ideas:
1. Weight-averaged pseudo-labels: use EMA teacher to generate
   more stable pseudo-labels for self-training
2. Augmentation-averaged predictions: average predictions over
   multiple augmentations for more reliable pseudo-labels
3. Stochastic restore: randomly restore a fraction of neurons
   to source weights to prevent catastrophic forgetting

Detection adaptation: Teacher model generates pseudo-labels for
detection, student adapts via pseudo-label matching loss.
Stochastic restore prevents detector drift.
"""

import copy
import logging
import random
from typing import Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CoTTAConfig:
    """Configuration for CoTTA baseline."""
    # EMA teacher
    ema_decay: float = 0.999      # teacher EMA decay rate

    # Augmentation averaging
    num_augmentations: int = 4     # N_aug for augmentation averaging
    aug_strength: float = 0.3      # augmentation strength

    # Stochastic restore
    restore_prob: float = 0.01     # p_rst: probability of restoring each neuron

    # Optimization
    learning_rate: float = 1e-3
    steps_per_batch: int = 1

    # Confidence threshold for pseudo-labels
    pseudo_label_threshold: float = 0.5


class CoTTADetection:
    """
    CoTTA adapted for streaming object detection.

    Uses mean-teacher framework with stochastic restore:
    1. EMA teacher generates stable pseudo-labels for detections
    2. Augmentation averaging improves pseudo-label reliability
    3. Stochastic restore prevents catastrophic forgetting by
       randomly resetting neurons to source weights

    This is particularly effective for continually shifting domains
    where vanilla adaptation would accumulate errors over time.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[CoTTAConfig] = None,
    ):
        self.config = config or CoTTAConfig()
        self.model = model                  # student
        self._teacher = None                # EMA teacher
        self._source_state = None
        self._optimizer = None
        self._step_count = 0
        self._restore_count = 0

    def setup(self):
        """Prepare student + teacher models."""
        # Save source state for stochastic restore
        self._source_state = copy.deepcopy(self.model.state_dict())

        # Create EMA teacher (deep copy)
        self._teacher = copy.deepcopy(self.model)
        self._teacher.eval()
        for param in self._teacher.parameters():
            param.requires_grad = False

        # Configure student: only update BN parameters
        trainable_params = []
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.requires_grad_(True)
                # Keep track_running_stats=True (default) — with batch_size=1,
                # train-mode batch stats are unreliable and destroy predictions.
                # We use eval mode + enable_grad() for adaptation instead.
                module.track_running_stats = True
                if module.weight is not None:
                    module.weight.requires_grad = True
                    trainable_params.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad = True
                    trainable_params.append(module.bias)

        # Freeze non-BN parameters
        for param in self.model.parameters():
            if not any(param is tp for tp in trainable_params):
                param.requires_grad = False

        self._trainable_params = trainable_params
        self._optimizer = torch.optim.Adam(
            trainable_params, lr=self.config.learning_rate
        )

        logger.info(f"CoTTA setup: {len(trainable_params)} trainable params, "
                     f"EMA decay={self.config.ema_decay}, "
                     f"restore_prob={self.config.restore_prob}")

    def forward_and_adapt(self, images: torch.Tensor) -> dict:
        """
        CoTTA forward: teacher pseudo-labels → student adaptation →
        EMA update → stochastic restore.
        """
        # Step 1: Teacher generates pseudo-labels (augmentation-averaged)
        teacher_output = self._augmentation_averaged_prediction(images)

        # Step 2: Student adapts using pseudo-labels
        for _ in range(self.config.steps_per_batch):
            self._adaptation_step(images, teacher_output)

        # Step 3: Update teacher via EMA
        self._update_teacher()

        # Step 4: Stochastic restore
        self._stochastic_restore()

        # Final prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(images)

        self._step_count += 1
        return output

    def _augmentation_averaged_prediction(self, images: torch.Tensor) -> dict:
        """
        Generate pseudo-labels by averaging predictions over
        multiple augmentations of the input.
        """
        self._teacher.eval()

        # Collect predictions from augmented versions
        all_outputs = []
        with torch.no_grad():
            # Original
            base_output = self._teacher(images)
            all_outputs.append(base_output)

            # Augmented versions
            for _ in range(self.config.num_augmentations - 1):
                aug_images = self._apply_augmentation(images)
                aug_output = self._teacher(aug_images)
                all_outputs.append(aug_output)

        # Use base teacher output as pseudo-label
        # (for detection, averaging bboxes is non-trivial, so we use
        # the teacher's direct prediction as pseudo-label)
        return base_output

    def _apply_augmentation(self, images: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations for prediction averaging."""
        aug = images.clone()
        strength = self.config.aug_strength

        # Random brightness
        if random.random() > 0.5:
            factor = 1.0 + (random.random() * 2 - 1) * strength
            aug = aug * factor

        # Random horizontal flip
        if random.random() > 0.5:
            aug = torch.flip(aug, dims=[-1])

        # Random Gaussian noise
        if random.random() > 0.5:
            noise = torch.randn_like(aug) * strength * 0.1
            aug = aug + noise

        # Clamp to valid range
        aug = aug.clamp(0, 1)
        return aug

    def _adaptation_step(self, images: torch.Tensor, teacher_output):
        """
        Adapt student using teacher pseudo-labels.

        Loss is self-training style: minimize KL divergence between
        student and teacher predictions on confident detections.

        Uses eval mode (pre-trained BN running stats) with enable_grad()
        for reliable predictions with batch_size=1 streaming detection.
        """
        self.model.eval()
        self._optimizer.zero_grad()

        with torch.enable_grad():
            student_output = self.model(images)

            # Compute pseudo-label consistency loss
            loss = self._pseudo_label_loss(student_output, teacher_output)

            if loss is not None and loss.requires_grad:
                loss.backward()
                self._optimizer.step()

    def _pseudo_label_loss(self, student_output, teacher_output) -> Optional[torch.Tensor]:
        """
        Compute KL divergence between student and teacher predictions.
        Only uses confident teacher predictions as pseudo-labels.
        """
        # Extract class probabilities from student and teacher
        # (YOLOv8 outputs are already sigmoid probabilities)
        s_probs = self._extract_logits(student_output)
        t_probs = self._extract_logits(teacher_output)

        if s_probs is None or t_probs is None:
            return None

        # Match shapes
        min_n = min(s_probs.shape[0], t_probs.shape[0])
        if min_n == 0:
            return None

        s_probs = s_probs[:min_n]
        t_probs = t_probs[:min_n].detach()

        # Filter confident teacher predictions (already probabilities)
        t_max_prob, _ = t_probs.max(dim=-1)
        confident_mask = t_max_prob >= self.config.pseudo_label_threshold

        if confident_mask.sum() == 0:
            return None

        # Binary cross-entropy for independent sigmoid class outputs
        s_filtered = s_probs[confident_mask].clamp(1e-7, 1 - 1e-7)
        t_filtered = t_probs[confident_mask]

        loss = F.binary_cross_entropy(s_filtered, t_filtered, reduction='mean')
        return loss

    def _extract_logits(self, output) -> Optional[torch.Tensor]:
        """Extract class probabilities from detection output.

        For YOLO-style detectors, the output is (B, 4+num_classes, num_anchors).
        Class values are already sigmoid probabilities (independent binary
        classifiers). Returns (B*N, C) tensor of probabilities in [0, 1].
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
            return logits
        else:
            return None

        # YOLO format: (B, 4+C, N) where C=num_classes, N=num_anchors
        if preds.dim() == 3 and preds.shape[1] > 4:
            # Transpose to (B, N, 4+C) then extract class probabilities
            preds_t = preds.permute(0, 2, 1)  # (B, N, 4+C)
            class_probs = preds_t[:, :, 4:]  # (B, N, C) — already sigmoid probabilities
            class_probs = class_probs.reshape(-1, class_probs.shape[-1])  # (B*N, C)
            return class_probs
        else:
            if preds.dim() > 2:
                preds = preds.reshape(-1, preds.shape[-1])
            return preds

    def _update_teacher(self):
        """EMA update: θ_teacher ← α · θ_teacher + (1 - α) · θ_student."""
        alpha = self.config.ema_decay
        for t_param, s_param in zip(
            self._teacher.parameters(), self.model.parameters()
        ):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)

    def _stochastic_restore(self):
        """
        Randomly restore a fraction of neurons to source weights.

        For each BN parameter, with probability p_rst, reset it
        to the original source value. This prevents indefinite
        drift from the source model.
        """
        n_restored = 0
        source_state = self._source_state

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in source_state:
                # Generate restore mask
                mask = (
                    torch.rand_like(param.data) < self.config.restore_prob
                ).float()
                # Restore selected elements
                param.data = (
                    mask * source_state[name]
                    + (1.0 - mask) * param.data
                )
                n_restored += int(mask.sum().item())

        if n_restored > 0:
            self._restore_count += n_restored

    def reset(self):
        """Reset to source state."""
        if self._source_state is not None:
            self.model.load_state_dict(self._source_state)
            self.setup()
        self._step_count = 0
        self._restore_count = 0

    @property
    def name(self) -> str:
        return "cotta_detection"

    def get_stats(self) -> dict:
        return {
            'method': 'CoTTA',
            'total_steps': self._step_count,
            'ema_decay': self.config.ema_decay,
            'restore_prob': self.config.restore_prob,
            'total_restored_neurons': self._restore_count,
        }
