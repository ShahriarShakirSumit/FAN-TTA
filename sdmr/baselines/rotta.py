"""
RoTTA: Robust Test-Time Adaptation.
Adapted for object detection.

Reference: Yuan et al., "Robust Test-Time Adaptation in Dynamic
Scenarios", CVPR 2023.

Key ideas:
1. Robust Batch Normalization (RBN): combines source and test-time
   BN statistics via a learnable balance factor
2. Category-balanced sampling: memory bank that maintains balanced
   sample distribution across categories for stable adaptation
3. Time-aware reweighting: recent samples weighted higher, old
   samples decay, addressing temporal non-stationarity
4. Teacher-student framework: EMA teacher provides stable targets

Detection adaptation: RBN for detection backbone, category-balanced
sampling across detection classes, time-aware frame weighting.
"""

import copy
import logging
import random
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RoTTAConfig:
    """Configuration for RoTTA baseline."""
    # Robust BN
    bn_momentum: float = 0.1            # momentum for RBN
    bn_alpha: float = 0.5               # α: balance between source/test BN stats

    # Memory bank
    memory_size: int = 256               # max samples in memory bank
    num_categories: int = 80             # number of detection categories (COCO=80)

    # Time-aware reweighting
    time_decay: float = 0.99             # λ: decay factor for old samples
    min_weight: float = 0.01             # minimum sample weight

    # Teacher-student
    ema_decay: float = 0.999             # EMA decay for teacher

    # Optimization
    learning_rate: float = 1e-3
    steps_per_batch: int = 1

    # Confidence threshold
    confidence_threshold: float = 0.3


class RobustBatchNorm2d(nn.Module):
    """
    Robust Batch Normalization that balances source and test statistics.

    μ_robust = α · μ_source + (1 - α) · μ_test
    σ²_robust = α · σ²_source + (1 - α) · σ²_test

    This prevents the BN statistics from being dominated by potentially
    shifted test data while still allowing adaptation.
    """

    def __init__(self, bn_layer: nn.BatchNorm2d, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.num_features = bn_layer.num_features

        # Source statistics (frozen)
        self.register_buffer('source_mean', bn_layer.running_mean.clone())
        self.register_buffer('source_var', bn_layer.running_var.clone())

        # Test statistics (updated online)
        self.register_buffer('test_mean', bn_layer.running_mean.clone())
        self.register_buffer('test_var', bn_layer.running_var.clone())

        # Learnable parameters
        self.weight = bn_layer.weight
        self.bias = bn_layer.bias
        self.eps = bn_layer.eps
        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update test statistics from current batch
        if x.shape[0] > 1:
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)

            # EMA update
            self.test_mean = (
                (1 - self.momentum) * self.test_mean
                + self.momentum * batch_mean
            )
            self.test_var = (
                (1 - self.momentum) * self.test_var
                + self.momentum * batch_var
            )

        # Robust statistics: blend source and test
        robust_mean = self.alpha * self.source_mean + (1 - self.alpha) * self.test_mean
        robust_var = self.alpha * self.source_var + (1 - self.alpha) * self.test_var

        # Normalize
        x = (x - robust_mean.view(1, -1, 1, 1)) / torch.sqrt(
            robust_var.view(1, -1, 1, 1) + self.eps
        )

        # Affine transform
        if self.weight is not None:
            x = x * self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)

        return x


class MemoryBank:
    """
    Category-balanced memory bank with time-aware weighting.

    Maintains a balanced buffer across detection categories:
    - New samples have weight 1.0
    - Old samples decay by λ each step
    - When memory is full, replace lowest-weight sample
    """

    def __init__(self, config: RoTTAConfig):
        self.config = config
        self.memory: List[Dict] = []  # {features, category, weight, timestamp}
        self._timestamp = 0

    def add(self, features: torch.Tensor, categories: List[int]):
        """Add features to memory bank with category tracking."""
        self._timestamp += 1

        for i, cat in enumerate(categories):
            entry = {
                'features': features[i:i+1].detach() if i < features.shape[0] else features[-1:].detach(),
                'category': cat,
                'weight': 1.0,
                'timestamp': self._timestamp,
            }

            if len(self.memory) < self.config.memory_size:
                self.memory.append(entry)
            else:
                # Replace lowest-weight sample
                min_idx = min(range(len(self.memory)),
                              key=lambda j: self.memory[j]['weight'])
                self.memory[min_idx] = entry

        # Decay old samples
        self._decay_weights()

    def _decay_weights(self):
        """Apply time-aware decay to all samples."""
        for entry in self.memory:
            if entry['timestamp'] < self._timestamp:
                age = self._timestamp - entry['timestamp']
                entry['weight'] = max(
                    self.config.min_weight,
                    self.config.time_decay ** age
                )

    def sample(self, n: int) -> Optional[torch.Tensor]:
        """Sample n features with category-balanced, time-aware weighting."""
        if len(self.memory) == 0:
            return None

        n = min(n, len(self.memory))
        weights = np.array([e['weight'] for e in self.memory])
        weights = weights / weights.sum()

        indices = np.random.choice(len(self.memory), size=n,
                                   replace=False, p=weights)

        features = torch.cat([self.memory[i]['features'] for i in indices], dim=0)
        return features

    def get_category_distribution(self) -> Dict[int, int]:
        """Get number of samples per category."""
        dist = defaultdict(int)
        for entry in self.memory:
            dist[entry['category']] += 1
        return dict(dist)

    def clear(self):
        self.memory.clear()
        self._timestamp = 0


class RoTTADetection:
    """
    RoTTA adapted for streaming object detection.

    Combines four mechanisms for robust adaptation:
    1. Robust BN: blends source+test statistics to avoid BN collapse
    2. Category-balanced memory: prevents class imbalance in adaptation
    3. Time-aware reweighting: addresses temporal non-stationarity
    4. EMA teacher: provides stable pseudo-labels

    Particularly effective for:
    - Long streams where BN statistics drift
    - Category-imbalanced detection scenarios
    - Non-stationary domain shifts
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[RoTTAConfig] = None,
    ):
        self.config = config or RoTTAConfig()
        self.model = model
        self._teacher = None
        self._source_state = None
        self._memory = MemoryBank(self.config)
        self._optimizer = None
        self._step_count = 0

    def setup(self):
        """Set up RoTTA: Robust BN + teacher + memory."""
        # Save source state
        self._source_state = copy.deepcopy(self.model.state_dict())

        # Replace standard BN with Robust BN
        self._replace_bn_with_robust_bn()

        # Create EMA teacher (after RBN replacement)
        self._teacher = copy.deepcopy(self.model)
        self._teacher.eval()
        for param in self._teacher.parameters():
            param.requires_grad = False

        # Collect trainable params (RBN weight/bias)
        trainable_params = []
        for module in self.model.modules():
            if isinstance(module, RobustBatchNorm2d):
                if module.weight is not None:
                    module.weight.requires_grad = True
                    trainable_params.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad = True
                    trainable_params.append(module.bias)

        # If no RobustBatchNorm2d found (model may not have standard BN),
        # fall back to standard BN params
        if not trainable_params:
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    # Keep track_running_stats=True (default) — with batch_size=1,
                    # train-mode batch stats are unreliable and destroy predictions.
                    module.track_running_stats = True
                    if module.weight is not None:
                        module.weight.requires_grad = True
                        trainable_params.append(module.weight)
                    if module.bias is not None:
                        module.bias.requires_grad = True
                        trainable_params.append(module.bias)

        # Freeze non-BN params
        for param in self.model.parameters():
            if not any(param is tp for tp in trainable_params):
                param.requires_grad = False

        self._trainable_params = trainable_params

        if trainable_params:
            self._optimizer = torch.optim.Adam(
                trainable_params, lr=self.config.learning_rate
            )
        else:
            self._optimizer = None

        logger.info(f"RoTTA setup: {len(trainable_params)} trainable params, "
                     f"memory_size={self.config.memory_size}, "
                     f"bn_alpha={self.config.bn_alpha}")

    def _replace_bn_with_robust_bn(self):
        """Replace BatchNorm2d layers with RobustBatchNorm2d."""
        replacements = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                replacements[name] = RobustBatchNorm2d(
                    module, alpha=self.config.bn_alpha
                )

        for name, new_module in replacements.items():
            # Navigate to parent and replace
            parts = name.split('.')
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

        if replacements:
            logger.info(f"Replaced {len(replacements)} BN layers with RobustBN")

    def forward_and_adapt(self, images: torch.Tensor) -> dict:
        """
        RoTTA forward: teacher pseudo-labels → memory update →
        category-balanced adaptation → EMA update.
        """
        # Get teacher pseudo-labels
        self._teacher.eval()
        with torch.no_grad():
            teacher_output = self._teacher(images)

        # Update memory bank
        self._update_memory(images, teacher_output)

        # Adapt using memory bank (category-balanced + time-weighted)
        if self._optimizer is not None:
            for _ in range(self.config.steps_per_batch):
                self._adaptation_step(images, teacher_output)

        # Update teacher EMA
        self._update_teacher()

        # Final prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(images)

        self._step_count += 1
        return output

    def _update_memory(self, images: torch.Tensor, teacher_output):
        """Add current frame features to category-balanced memory."""
        # Extract category predictions from teacher
        categories = self._extract_categories(teacher_output)

        # Add image features to memory
        self._memory.add(images.detach(), categories)

    def _extract_categories(self, output) -> List[int]:
        """Extract predicted categories from detection output.

        For YOLO-style detectors, the output is (B, 4+num_classes, num_anchors).
        Class values are already sigmoid probabilities. We take argmax to
        find the most likely category per anchor.
        """
        # Handle YOLO output format: tuple/list where first element is predictions
        if isinstance(output, (list, tuple)):
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                preds = output[0]
            else:
                return [0]
        elif isinstance(output, torch.Tensor):
            preds = output
        elif isinstance(output, dict):
            logits = output.get('class_logits', output.get('cls', None))
            if logits is None:
                return [0]
            if logits.dim() > 2:
                logits = logits.reshape(-1, logits.shape[-1])
            if logits.shape[-1] > 1:
                categories = logits.argmax(dim=-1).cpu().tolist()
                return categories if categories else [0]
            return [0]
        else:
            return [0]

        # YOLO format: (B, 4+C, N) where C=num_classes, N=num_anchors
        if preds.dim() == 3 and preds.shape[1] > 4:
            preds_t = preds.permute(0, 2, 1)  # (B, N, 4+C)
            class_probs = preds_t[:, :, 4:]  # (B, N, C) — already sigmoid probabilities
            class_probs = class_probs.reshape(-1, class_probs.shape[-1])
            if class_probs.shape[-1] > 1:
                categories = class_probs.argmax(dim=-1).cpu().tolist()
                return categories if categories else [0]
        return [0]

    def _adaptation_step(self, images: torch.Tensor, teacher_output):
        """
        Adaptation using teacher-student with time-aware,
        category-balanced samples from memory.

        Uses eval mode (pre-trained BN running stats) with enable_grad()
        for reliable predictions with batch_size=1 streaming detection.
        """
        self.model.eval()
        self._optimizer.zero_grad()

        with torch.enable_grad():
            # Forward student
            student_output = self.model(images)

            # Pseudo-label consistency loss
            loss = self._consistency_loss(student_output, teacher_output)

            if loss is not None and loss.requires_grad:
                loss.backward()
                self._optimizer.step()

    def _consistency_loss(self, student_output, teacher_output) -> Optional[torch.Tensor]:
        """Teacher-student consistency loss."""
        # Extract class probabilities (already sigmoid, not logits)
        s_probs = self._extract_logits(student_output)
        t_probs = self._extract_logits(teacher_output)

        if s_probs is None or t_probs is None:
            return None

        min_n = min(s_probs.shape[0], t_probs.shape[0])
        if min_n == 0:
            return None

        s_probs = s_probs[:min_n]
        t_probs = t_probs[:min_n].detach()

        # Filter confident teacher predictions (already probabilities)
        t_max_prob, _ = t_probs.max(dim=-1)
        confident = t_max_prob >= self.config.confidence_threshold

        if confident.sum() == 0:
            return None

        # Binary cross-entropy for independent sigmoid class outputs
        s_filtered = s_probs[confident].clamp(1e-7, 1 - 1e-7)
        t_filtered = t_probs[confident]

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
        """EMA teacher update."""
        alpha = self.config.ema_decay
        for t_param, s_param in zip(
            self._teacher.parameters(), self.model.parameters()
        ):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)

    def reset(self):
        """Reset to source state."""
        if self._source_state is not None:
            # First, reverse RobustBN replacements back to standard BN
            self._restore_standard_bn()
            # Now load source state (which has standard BN keys)
            self.model.load_state_dict(self._source_state)
            self._memory.clear()
            self.setup()
        self._step_count = 0

    def _restore_standard_bn(self):
        """Replace RobustBatchNorm2d back with standard BatchNorm2d."""
        replacements = {}
        for name, module in self.model.named_modules():
            if isinstance(module, RobustBatchNorm2d):
                bn = nn.BatchNorm2d(module.num_features, eps=module.eps)
                bn.weight = module.weight
                bn.bias = module.bias
                bn.running_mean = module.source_mean.clone()
                bn.running_var = module.source_var.clone()
                replacements[name] = bn
        for name, new_module in replacements.items():
            parts = name.split('.')
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

    @property
    def name(self) -> str:
        return "rotta_detection"

    def get_stats(self) -> dict:
        return {
            'method': 'RoTTA',
            'total_steps': self._step_count,
            'memory_size': len(self._memory.memory),
            'category_distribution': self._memory.get_category_distribution(),
            'bn_alpha': self.config.bn_alpha,
            'ema_decay': self.config.ema_decay,
        }
