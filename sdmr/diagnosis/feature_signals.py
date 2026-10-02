"""
Feature-space diagnosis signals.

Extracts signals from intermediate neural-network feature maps
(backbone, FPN, neck) to detect distribution shifts in learned
representations. These signals complement input-space and
output-space signals by capturing shifts invisible to pixel-level
or box-level analysis.

Key signals:
- Channel-wise activation statistics (mean, var, sparsity)
- Feature norm trajectory
- Gram matrix style statistics
- Channel correlation shifts
- Batch-norm running-stat divergence
- Feature entropy proxy
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class FeatureSignals:
    """Container for feature-space signals from a single frame."""
    # Per-layer stats
    layer_mean: dict[str, float]
    layer_std: dict[str, float]
    layer_sparsity: dict[str, float]
    layer_l2_norm: dict[str, float]
    layer_entropy_proxy: dict[str, float]

    # Aggregated across layers
    agg_mean: float = 0.0
    agg_std: float = 0.0
    agg_sparsity: float = 0.0
    agg_l2_norm: float = 0.0
    agg_entropy_proxy: float = 0.0

    def to_flat_dict(self, prefix: str = "feat") -> dict[str, float]:
        """Flatten to a single dict for fusion with other signal types."""
        d: dict[str, float] = {}
        for attr in ["layer_mean", "layer_std", "layer_sparsity",
                      "layer_l2_norm", "layer_entropy_proxy"]:
            for layer_name, val in getattr(self, attr).items():
                d[f"{prefix}_{attr}_{layer_name}"] = val
        d[f"{prefix}_agg_mean"] = self.agg_mean
        d[f"{prefix}_agg_std"] = self.agg_std
        d[f"{prefix}_agg_sparsity"] = self.agg_sparsity
        d[f"{prefix}_agg_l2_norm"] = self.agg_l2_norm
        d[f"{prefix}_agg_entropy_proxy"] = self.agg_entropy_proxy
        return d


def _channel_mean(feat: np.ndarray) -> float:
    """Mean activation across spatial + channel dimensions."""
    return float(np.mean(feat))


def _channel_std(feat: np.ndarray) -> float:
    """Std of activations across spatial + channel dimensions."""
    return float(np.std(feat))


def _channel_sparsity(feat: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of near-zero activations (ReLU-dead proxy)."""
    return float(np.mean(np.abs(feat) < threshold))


def _l2_norm(feat: np.ndarray) -> float:
    """L2 norm of the flattened feature vector."""
    return float(np.sqrt(np.sum(feat ** 2)))


def _entropy_proxy(feat: np.ndarray) -> float:
    """
    Entropy-like proxy from spatial activation distribution.

    For a feature map of shape (C, H, W), we compute the spatial
    softmax for each channel and average the entropy.
    """
    if feat.ndim < 3:
        return 0.0

    # feat shape: (C, H, W) or (1, C, H, W)
    if feat.ndim == 4:
        feat = feat[0]  # drop batch dim

    c, h, w = feat.shape
    entropies = []
    for ci in range(min(c, 64)):  # sample up to 64 channels
        spatial = feat[ci].flatten()
        # shift for numerical stability
        spatial = spatial - spatial.max()
        exp_s = np.exp(spatial)
        softmax = exp_s / (exp_s.sum() + 1e-8)
        entropy = -np.sum(softmax * np.log(softmax + 1e-8))
        entropies.append(entropy)
    return float(np.mean(entropies))


def _gram_matrix_trace(feat: np.ndarray) -> float:
    """
    Trace of the Gram matrix of channel activations.

    For (C, H, W) features, the Gram matrix is C x C.
    Trace encodes total energy; shifts in trace indicate
    distribution changes in feature response patterns.
    """
    if feat.ndim == 4:
        feat = feat[0]
    if feat.ndim < 3:
        return 0.0

    c, h, w = feat.shape
    feat_flat = feat.reshape(c, -1)  # (C, H*W)
    gram = feat_flat @ feat_flat.T   # (C, C)
    return float(np.trace(gram)) / (c * h * w + 1e-8)


def _channel_correlation_std(feat: np.ndarray, sample_channels: int = 32) -> float:
    """
    Std of pairwise channel correlations (sample for speed).

    Changes in inter-channel correlation structure indicate
    feature-level distribution shift.
    """
    if feat.ndim == 4:
        feat = feat[0]
    if feat.ndim < 3:
        return 0.0

    c, h, w = feat.shape
    feat_flat = feat.reshape(c, -1)

    # Sample channels
    if c > sample_channels:
        idx = np.linspace(0, c - 1, sample_channels, dtype=int)
        feat_flat = feat_flat[idx]

    # Correlation matrix
    norms = np.sqrt(np.sum(feat_flat ** 2, axis=1, keepdims=True)) + 1e-8
    feat_normed = feat_flat / norms
    corr = feat_normed @ feat_normed.T

    # Upper triangle correlations
    triu_idx = np.triu_indices(corr.shape[0], k=1)
    correlations = corr[triu_idx]
    return float(np.std(correlations))


def compute_feature_signals(
    feature_maps: dict[str, np.ndarray],
    compute_gram: bool = True,
    compute_correlation: bool = True,
) -> FeatureSignals:
    """
    Compute feature-space signals from extracted feature maps.

    Args:
        feature_maps: Dict of layer_name -> feature array with shape
                     (C, H, W) or (1, C, H, W).
        compute_gram: Whether to compute Gram-matrix trace signal.
        compute_correlation: Whether to compute channel correlation signal.

    Returns:
        FeatureSignals with per-layer and aggregated statistics.
    """
    layer_mean: dict[str, float] = {}
    layer_std: dict[str, float] = {}
    layer_sparsity: dict[str, float] = {}
    layer_l2_norm: dict[str, float] = {}
    layer_entropy_proxy: dict[str, float] = {}

    for name, feat in feature_maps.items():
        layer_mean[name] = _channel_mean(feat)
        layer_std[name] = _channel_std(feat)
        layer_sparsity[name] = _channel_sparsity(feat)
        layer_l2_norm[name] = _l2_norm(feat)
        layer_entropy_proxy[name] = _entropy_proxy(feat)

        if compute_gram:
            layer_mean[f"{name}_gram_trace"] = _gram_matrix_trace(feat)
        if compute_correlation:
            layer_std[f"{name}_corr_std"] = _channel_correlation_std(feat)

    # Aggregate across layers
    def _safe_mean(d: dict[str, float]) -> float:
        vals = list(d.values())
        return float(np.mean(vals)) if vals else 0.0

    signals = FeatureSignals(
        layer_mean=layer_mean,
        layer_std=layer_std,
        layer_sparsity=layer_sparsity,
        layer_l2_norm=layer_l2_norm,
        layer_entropy_proxy=layer_entropy_proxy,
        agg_mean=_safe_mean(layer_mean),
        agg_std=_safe_mean(layer_std),
        agg_sparsity=_safe_mean(layer_sparsity),
        agg_l2_norm=_safe_mean(layer_l2_norm),
        agg_entropy_proxy=_safe_mean(layer_entropy_proxy),
    )
    return signals


def compute_bn_divergence(
    current_bn_stats: dict[str, tuple[np.ndarray, np.ndarray]],
    reference_bn_stats: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """
    Compute divergence between current and reference batch-norm statistics.

    Each entry is (running_mean, running_var) per BN layer.
    Returns per-layer KL-like divergence and aggregated score.
    """
    divergences: dict[str, float] = {}

    for name in current_bn_stats:
        if name not in reference_bn_stats:
            continue

        c_mean, c_var = current_bn_stats[name]
        r_mean, r_var = reference_bn_stats[name]

        # Gaussian KL divergence per channel, then average
        c_var = np.maximum(c_var, 1e-8)
        r_var = np.maximum(r_var, 1e-8)

        kl = 0.5 * (
            np.log(r_var / c_var)
            + c_var / r_var
            + (c_mean - r_mean) ** 2 / r_var
            - 1.0
        )
        divergences[f"bn_kl_{name}"] = float(np.mean(kl))

    if divergences:
        divergences["bn_kl_agg"] = float(np.mean(list(divergences.values())))

    return divergences
