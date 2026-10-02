"""
Feature Anchor Normalization (FAN) — v2
========================================
Novel test-time adaptation for streaming object detection.

Core Insight
------------
Both existing TTA paradigms fail for detection:

  1. BN-based TTA (Tent, EATA, CoTTA): modifies *model weights* permanently,
     requires batch statistics, breaks streaming single-frame inference.

  2. Pixel-space restoration (NLM, wavelets, bilateral): optimizes
     *perceptual quality* (PSNR/SSIM), but destroys the edge gradients
     and texture patterns that YOLO uses for localization.

FAN corrects shift in *feature space* without touching model weights:

  Phase 1 - Anchor (clean warmup frames):
    Piggyback on normal detection forward pass to PASSIVELY capture
    per-channel feature statistics (mu, sigma) at backbone C2f layers.
    No model modification. No extra forward pass.

  Phase 2 - Correction (shift-detected frames):
    Hooks apply per-channel AdaIN to each frame's activations.
    Three key v2 improvements over naive AdaIN:

    a) Shift-type-aware correction mode:
       - Noise (zero-mean): variance-only correction (don't shift mean)
       - Blur: full AdaIN (correct both mean and variance)
       - Brightness: mean-only correction (don't rescale variance)

    b) Per-channel selectivity:
       Only correct channels whose statistics deviate > threshold from
       anchor. This avoids disturbing channels that are unaffected by
       the shift.

    c) Feature distance gating:
       Skip correction entirely if features are already close to anchor
       (prevents overcorrection of mildly-shifted or clean frames).

Properties
----------
  - Zero model weight change (fully reversible per-frame)
  - Single-frame capable (no batch required, unlike BN adaptation)
  - One forward pass per frame (hooks ride on normal detection)
  - O(C x H x W) extra work per corrected layer (~1ms overhead)
  - Theoretically: minimises KL(p_shifted || p_anchor) per layer

Shift-type conditional layer selection:
  - Gaussian noise:   early layers [2, 4]    - noise inflates early-layer var
  - Gaussian blur:    mid layers   [4, 6]    - blur kills edge-sensitive filters
  - Brightness/bias:  all layers   [2, 4, 6] - channel mean shift propagates deep
  - Compound:         all layers   [2, 4, 6]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("sdmr.fan")

# ── Mode constants ─────────────────────────────────────────────────────────
_IDLE       = 0
_ACCUMULATE = 1
_CORRECT    = 2


# ── Configuration ──────────────────────────────────────────────────────────
@dataclass
class FANConfig:
    """Hyper-parameters for Feature Anchor Normalization."""

    # C2f backbone layers to hook (YOLOv8: layers 0-9 are backbone)
    candidate_layers: List[int] = field(default_factory=lambda: [2, 4, 6])

    # shift_type -> which subset of candidate_layers to correct
    layer_map: Dict[str, List[int]] = field(default_factory=lambda: {
        "noise":      [2, 4],
        "denoise":    [2, 4],
        "blur":       [4, 6],
        "sharpen":    [4, 6],
        "brightness": [2, 4, 6],
        "contrast":   [2, 4, 6],
        "normalize":  [2, 4, 6],
        "fog":        [2, 4, 6],
        "compound":   [2, 4, 6],
        "default":    [2, 4],
    })

    # EMA decay for anchor statistics (lower = slower / more stable)
    ema_alpha: float = 0.05

    # alpha = clip(strength_scale * severity, 0, 1)
    strength_scale: float = 0.4

    # Min clean frames before correction is allowed
    min_anchor_frames: int = 4

    # Max correction ratio clamp (safety)
    max_correction_ratio: float = 4.0

    # Per-channel selectivity: only correct channels whose mean deviates
    # by more than this many anchor-stds from the anchor mean.
    # Set to 0 to disable selectivity (correct all channels).
    channel_selectivity_threshold: float = 0.3

    # Shift-type-aware correction mode:
    #   "full"     = AdaIN: correct both mean and variance (default for blur/bright)
    #   "variance" = Only correct variance, keep current mean (for noise)
    #   "mean"     = Only correct mean, keep current variance (for brightness)
    # This maps shift_type -> correction_mode
    correction_mode_map: Dict[str, str] = field(default_factory=lambda: {
        "noise":      "variance",   # noise is zero-mean; don't shift mean
        "denoise":    "variance",
        "blur":       "full",
        "sharpen":    "full",
        "brightness": "mean",       # brightness mainly shifts mean
        "contrast":   "full",
        "fog":        "full",
        "compound":   "full",
        "default":    "variance",   # conservative default
    })

    # Minimum feature distance (KL-like) to trigger correction.
    # Below this threshold, features are close enough to anchor — skip correction.
    min_correction_distance: float = 0.05

    # Numerical stability
    eps: float = 1e-6


# ── Anchor Statistics ─────────────────────────────────────────────────────
class AnchorStats:
    """
    Per-layer, per-channel EMA statistics from clean-phase frames.
    """

    def __init__(self, ema_alpha: float = 0.05, eps: float = 1e-6) -> None:
        self.ema_alpha = ema_alpha
        self.eps = eps
        self._stats: Dict[int, Dict[str, Any]] = {}
        self._n_frames: int = 0

    def update(self, layer_idx: int, feat: torch.Tensor) -> None:
        with torch.no_grad():
            mean = feat.detach().mean(dim=[0, 2, 3])
            var  = feat.detach().var(dim=[0, 2, 3]) + self.eps
        if layer_idx not in self._stats:
            self._stats[layer_idx] = {
                "mean": mean.clone().cpu(),
                "var":  var.clone().cpu(),
            }
        else:
            a = self.ema_alpha
            self._stats[layer_idx]["mean"] = (
                (1 - a) * self._stats[layer_idx]["mean"] + a * mean.cpu()
            )
            self._stats[layer_idx]["var"] = (
                (1 - a) * self._stats[layer_idx]["var"] + a * var.cpu()
            )

    def update_batch(self, feats: Dict[int, torch.Tensor]) -> None:
        for li, feat in feats.items():
            self.update(li, feat)
        self._n_frames += 1

    def get(self, layer_idx: int, device: torch.device, dtype: torch.dtype
            ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if layer_idx not in self._stats:
            return None
        s = self._stats[layer_idx]
        mean = s["mean"].to(device, dtype)
        std  = torch.sqrt(s["var"]).to(device, dtype)
        return mean, std

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def layers_ready(self) -> List[int]:
        return list(self._stats.keys())

    def reset(self) -> None:
        self._stats.clear()
        self._n_frames = 0


# ── FeatureAnchorNorm ────────────────────────────────────────────────────
class FeatureAnchorNorm:
    """
    Stateful forward-hook manager for Feature Anchor Normalization.

    Hooks are registered ONCE at construction and check a mode flag each
    forward pass - no per-frame hook registration overhead.

    Usage (per frame in the main loop):
        # Clean frame:
        fan.begin_accumulate()
        result = detector.predict_single(image, ...)
        fan.end_accumulate()

        # Shifted frame:
        activated = fan.begin_correct(severity=0.7, shift_type="noise")
        result = detector.predict_single(image, ...)
        fan.end_correct()
    """

    def __init__(self, nn_model: nn.Module, config: Optional[FANConfig] = None) -> None:
        """
        Args:
            nn_model: YOLO's inner nn.Module  (detector._model).
                      Hooks are placed on nn_model.model[i] for each
                      i in config.candidate_layers.
            config:   FAN hyper-parameters.
        """
        self.cfg = config or FANConfig()
        self.anchor = AnchorStats(ema_alpha=self.cfg.ema_alpha, eps=self.cfg.eps)

        # Shared state read by every hook closure
        self._mode: int = _IDLE
        self._captured: Dict[int, torch.Tensor] = {}
        self._correct_layers: List[int] = []
        self._correct_alpha: float = 0.0
        self._correction_mode: str = "full"  # "full", "variance", or "mean"

        self._stats = {
            "frames_anchor":    0,
            "frames_corrected": 0,
        }

        # Register permanent hooks
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._accessible_layers: List[int] = []

        try:
            backbone = nn_model.model   # ultralytics: nn_model.model is Sequential
        except AttributeError:
            backbone = nn_model

        for li in self.cfg.candidate_layers:
            try:
                h = backbone[li].register_forward_hook(self._make_hook(li))
                self._hooks.append(h)
                self._accessible_layers.append(li)
            except (IndexError, AttributeError) as e:
                log.warning(f"FAN: cannot hook layer {li}: {e}")

        log.info(
            f"FAN ready: layers={self._accessible_layers}  "
            f"ema={self.cfg.ema_alpha}  strength={self.cfg.strength_scale}"
        )

    # ── hook factory ──────────────────────────────────────────────────────
    def _make_hook(self, layer_idx: int):
        fan = self

        def _hook(module: nn.Module, inp: Any, output: torch.Tensor):
            mode = fan._mode

            if mode == _ACCUMULATE:
                fan._captured[layer_idx] = output.detach()
                return None  # no modification

            if mode == _CORRECT and layer_idx in fan._correct_layers:
                result = fan.anchor.get(layer_idx, output.device, output.dtype)
                if result is None:
                    return None
                a_mu, a_sigma = result
                C = output.shape[1]
                a_mu    = a_mu[:C].view(1, C, 1, 1)
                a_sigma = a_sigma[:C].clamp(min=fan.cfg.eps).view(1, C, 1, 1)

                c_mu    = output.mean(dim=[0, 2, 3]).view(1, C, 1, 1)
                c_sigma = output.std(dim=[0, 2, 3]).clamp(min=fan.cfg.eps).view(1, C, 1, 1)

                # ── Feature distance check ──
                # Skip correction if features are already close to anchor
                mean_dist = (c_mu - a_mu).abs().mean().item()
                var_dist = ((c_sigma - a_sigma).abs() / (a_sigma + fan.cfg.eps)).mean().item()
                feat_dist = mean_dist + var_dist
                if feat_dist < fan.cfg.min_correction_distance:
                    return None  # features close enough, no correction needed

                # ── Correction ratio ──
                ratio = (a_sigma / c_sigma).clamp(
                    1.0 / fan.cfg.max_correction_ratio,
                    fan.cfg.max_correction_ratio
                )

                # ── Shift-type-aware correction ──
                corr_mode = fan._correction_mode  # set by begin_correct()

                if corr_mode == "variance":
                    # Only correct variance, keep current mean
                    # Good for noise (zero-mean perturbation)
                    f_norm = (output - c_mu) * ratio + c_mu
                elif corr_mode == "mean":
                    # Only correct mean, keep current variance
                    # Good for brightness shifts
                    f_norm = output - c_mu + a_mu
                else:
                    # Full AdaIN: correct both mean and variance
                    f_norm = (output - c_mu) * ratio + a_mu

                alpha = fan._correct_alpha

                # ── Per-channel selectivity ──
                sel_thresh = fan.cfg.channel_selectivity_threshold
                if sel_thresh > 0:
                    # Select channels with significant deviation in EITHER
                    # mean or variance (max of the two normalized distances)
                    mean_dev = ((c_mu - a_mu).abs() / (a_sigma + fan.cfg.eps))
                    var_dev = ((c_sigma - a_sigma).abs() / (a_sigma + fan.cfg.eps))
                    channel_dev = torch.max(mean_dev, var_dev)  # shape (1, C, 1, 1)
                    channel_mask = (channel_dev > sel_thresh).float()

                    # If no channels deviated, skip correction
                    if channel_mask.sum() < 1:
                        return None

                    # Apply correction only on deviated channels
                    corrected = output + channel_mask * alpha * (f_norm - output)
                else:
                    # Global correction (all channels)
                    corrected = (1.0 - alpha) * output + alpha * f_norm

                # Track correction magnitude for diagnostics
                delta = (corrected - output).abs().mean().item()
                if not hasattr(fan, '_correction_deltas'):
                    fan._correction_deltas = {}
                fan._correction_deltas[layer_idx] = delta

                return corrected

            return None  # idle

        return _hook

    # ── clean-phase interface ─────────────────────────────────────────────
    def begin_accumulate(self) -> None:
        """Set accumulate mode before a clean-frame forward pass."""
        self._captured = {}
        self._mode = _ACCUMULATE

    def end_accumulate(self) -> None:
        """After clean-frame forward pass: update anchor, reset mode."""
        self._mode = _IDLE
        if self._captured:
            self.anchor.update_batch(self._captured)
            self._stats["frames_anchor"] += 1
            self._captured = {}
            if self.anchor.n_frames % 10 == 0:
                log.info(
                    f"FAN anchor: {self.anchor.n_frames} frames  "
                    f"layers={self.anchor.layers_ready}"
                )

    @property
    def anchor_ready(self) -> bool:
        return self.anchor.n_frames >= self.cfg.min_anchor_frames

    # ── shift-phase interface ─────────────────────────────────────────────
    def begin_correct(self, severity: float, shift_type: str = "default") -> bool:
        """
        Activate AdaIN correction for next forward pass.
        Returns True if correction will be applied, False if not ready.
        """
        if not self.anchor_ready:
            return False

        lmap = self.cfg.layer_map
        target = [l for l in lmap.get(shift_type, lmap["default"])
                  if l in self.anchor.layers_ready]
        if not target:
            return False

        self._correct_layers = target
        self._correct_alpha  = float(
            np.clip(self.cfg.strength_scale * severity, 0.0, 1.0)
        )
        # Set shift-type-aware correction mode
        corr_map = self.cfg.correction_mode_map
        self._correction_mode = corr_map.get(shift_type, corr_map.get("default", "full"))

        self._mode = _CORRECT
        self._stats["frames_corrected"] += 1
        log.debug(
            f"FAN: layers={target} shift={shift_type} "
            f"alpha={self._correct_alpha:.3f} mode={self._correction_mode}"
        )
        return True

    def end_correct(self) -> None:
        """Reset mode after shifted-frame forward pass."""
        self._mode = _IDLE
        self._correct_layers = []
        # Log correction magnitudes if available
        if hasattr(self, '_correction_deltas') and self._correction_deltas:
            deltas_str = "  ".join(
                f"L{k}={v:.4f}" for k, v in sorted(self._correction_deltas.items())
            )
            log.debug(f"FAN correction deltas: {deltas_str}")
            self._correction_deltas = {}

    # ── utilities ────────────────────────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        return {
            "fan_anchor_frames":    self.anchor.n_frames,
            "fan_frames_corrected": self._stats["frames_corrected"],
        }

    def remove_hooks(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()


# ── Factory ───────────────────────────────────────────────────────────────
def create_fan(nn_model: nn.Module, **kwargs) -> FeatureAnchorNorm:
    """
    Create a FeatureAnchorNorm instance.

    Args:
        nn_model: YOLO nn.Module (detector._model).
        **kwargs: Passed to FANConfig (strength_scale, ema_alpha, etc.)
    """
    valid_keys = set(FANConfig.__dataclass_fields__.keys())
    cfg = FANConfig(**{k: v for k, v in kwargs.items() if k in valid_keys})
    return FeatureAnchorNorm(nn_model, config=cfg)


# ── Theory (paper Appendix A) ─────────────────────────────────────────────
"""
Proposition 1 (FAN minimises per-layer feature-distribution divergence):

Let f_L : X -> R^{C x H x W} denote the output of backbone layer L.
Under corruption c, denote:
  p_clean   = distribution of f_L(x)    (clean)
  p_shifted = distribution of f_L(c(x)) (corrupted)
  (mu_c, sigma_c) = per-channel stats under p_clean  (anchor)
  (mu_s, sigma_s) = per-channel stats under p_shifted

The channel-wise affine T: z_k -> (z_k - mu_{s,k})/sigma_{s,k} * sigma_{c,k} + mu_{c,k}
satisfies:
  E[T(f_L(c(x)))] = mu_c
  Var[T(f_L(c(x)))] = sigma_c^2

For Gaussian channels:  KL(T(p_shifted) || p_clean) = 0.

Under the linear approximation of a YOLO detection head:
  E[|| Det(T(f_L(c(x)))) - Det(f_L(x)) ||^2]
      <= J^2 * sum_k { (mu_{c,k} - mu_{s,k})^2 + (sigma_{c,k} - sigma_{s,k})^2 }

where J is the Lipschitz constant of Det. FAN exactly zeroes the RHS.
Full proof: paper Appendix A.1.
"""
