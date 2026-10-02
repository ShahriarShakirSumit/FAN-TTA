"""
Input-space shift diagnosis signals.

Computes cheap per-frame statistics from raw images:
- Brightness (mean/std)
- Contrast estimate
- Color histogram drift
- Blur score (variance of Laplacian)
- Noise estimate proxy
- Compression artifact proxy
- Edge density
- Frame difference (temporal)
- Illumination score

All functions operate on single images and return scalar features.
Window aggregation is handled by TemporalBuffer.

Usage:
    signals = compute_input_signals(image)
    # signals = {"brightness_mean": 0.45, "blur_score": 120.5, ...}
"""

import numpy as np
from typing import Optional
from PIL import Image


def _to_numpy(image) -> np.ndarray:
    """Convert image to numpy uint8 HWC array."""
    if isinstance(image, np.ndarray):
        if image.dtype in (np.float32, np.float64):
            return np.clip(image * 255, 0, 255).astype(np.uint8)
        return image.astype(np.uint8)
    if isinstance(image, Image.Image):
        return np.array(image, dtype=np.uint8)
    try:
        import torch
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in (1, 3):
                image = image.permute(1, 2, 0)
            arr = image.cpu().numpy()
            if arr.dtype in (np.float32, np.float64):
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            return arr
    except ImportError:
        pass
    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 image to grayscale."""
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 1:
        return arr[:, :, 0]
    # Standard luminance weights
    return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)


def brightness_stats(image) -> dict[str, float]:
    """
    Compute brightness statistics.

    Returns:
        brightness_mean: Mean pixel intensity [0, 1]
        brightness_std: Std of pixel intensity
    """
    arr = _to_numpy(image)
    gray = _to_gray(arr).astype(np.float32) / 255.0
    return {
        "brightness_mean": float(np.mean(gray)),
        "brightness_std": float(np.std(gray)),
    }


def contrast_estimate(image) -> dict[str, float]:
    """
    Estimate contrast using the Michelson contrast formula
    and RMS contrast.

    Returns:
        contrast_rms: RMS contrast
        contrast_michelson: Michelson contrast (Imax-Imin)/(Imax+Imin)
    """
    arr = _to_numpy(image)
    gray = _to_gray(arr).astype(np.float64) / 255.0

    rms = float(np.std(gray))

    i_max = float(np.max(gray))
    i_min = float(np.min(gray))
    michelson = (i_max - i_min) / (i_max + i_min + 1e-8)

    return {
        "contrast_rms": rms,
        "contrast_michelson": michelson,
    }


def blur_score(image) -> dict[str, float]:
    """
    Estimate blur using variance of the Laplacian.
    Lower values = more blurry.

    Returns:
        blur_laplacian_var: Variance of Laplacian (higher = sharper)
    """
    arr = _to_numpy(image)
    gray = _to_gray(arr).astype(np.float64)

    # Laplacian kernel
    # [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    from scipy.ndimage import laplace
    lap = laplace(gray)
    var = float(np.var(lap))

    return {"blur_laplacian_var": var}


def noise_estimate(image) -> dict[str, float]:
    """
    Estimate noise level using median absolute deviation of
    high-frequency Laplacian response.

    Returns:
        noise_estimate_mad: Estimated noise level
    """
    arr = _to_numpy(image)
    gray = _to_gray(arr).astype(np.float64)

    from scipy.ndimage import laplace
    lap = laplace(gray)
    mad = float(np.median(np.abs(lap - np.median(lap))))
    # Robust noise estimate (Donoho's MAD estimator)
    sigma_est = mad * 1.4826

    return {"noise_estimate_mad": sigma_est}


def color_histogram_features(image, bins: int = 16) -> dict[str, float]:
    """
    Compute color histogram statistics per channel.

    Returns:
        color_r_mean, color_g_mean, color_b_mean: Channel means
        color_r_std, color_g_std, color_b_std: Channel stds
        color_hist_entropy: Entropy of luminance histogram
    """
    arr = _to_numpy(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    features = {}
    for c, name in enumerate(["r", "g", "b"]):
        channel = arr[:, :, c].astype(np.float64) / 255.0
        features[f"color_{name}_mean"] = float(np.mean(channel))
        features[f"color_{name}_std"] = float(np.std(channel))

    # Luminance histogram entropy
    gray = _to_gray(arr)
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256), density=True)
    hist = hist + 1e-10
    hist = hist / hist.sum()
    entropy = -float(np.sum(hist * np.log2(hist)))
    features["color_hist_entropy"] = entropy

    return features


def edge_density(image) -> dict[str, float]:
    """
    Compute edge density using Sobel filters.

    Returns:
        edge_density: Fraction of edge pixels
        edge_mean_magnitude: Mean edge magnitude
    """
    arr = _to_numpy(image)
    gray = _to_gray(arr).astype(np.float64)

    from scipy.ndimage import sobel
    sx = sobel(gray, axis=0)
    sy = sobel(gray, axis=1)
    magnitude = np.sqrt(sx**2 + sy**2)

    # Threshold for edge detection
    threshold = np.mean(magnitude) + np.std(magnitude)
    edge_mask = magnitude > threshold
    density = float(np.mean(edge_mask))

    return {
        "edge_density": density,
        "edge_mean_magnitude": float(np.mean(magnitude)),
    }


def compression_proxy(image) -> dict[str, float]:
    """
    Estimate JPEG compression artifacts by analyzing 8x8 block boundaries.

    Returns:
        compression_blockiness: Block boundary discontinuity score
    """
    arr = _to_numpy(image)
    gray = _to_gray(arr).astype(np.float64)
    h, w = gray.shape

    # Measure discontinuity at 8-pixel boundaries
    h_diffs = []
    for y in range(8, h - 8, 8):
        diff = float(np.mean(np.abs(gray[y, :] - gray[y-1, :])))
        h_diffs.append(diff)

    v_diffs = []
    for x in range(8, w - 8, 8):
        diff = float(np.mean(np.abs(gray[:, x] - gray[:, x-1])))
        v_diffs.append(diff)

    # Normalize by overall gradient magnitude
    all_h_diff = float(np.mean(np.abs(np.diff(gray, axis=0))))
    all_v_diff = float(np.mean(np.abs(np.diff(gray, axis=1))))
    avg_overall = (all_h_diff + all_v_diff) / 2 + 1e-8

    blockiness = 0.0
    if h_diffs and v_diffs:
        avg_boundary = (np.mean(h_diffs) + np.mean(v_diffs)) / 2
        blockiness = avg_boundary / avg_overall

    return {"compression_blockiness": float(blockiness)}


def frame_difference(
    current_image,
    previous_image,
) -> dict[str, float]:
    """
    Compute frame-to-frame difference (for temporal streams).

    Returns:
        frame_diff_mean: Mean absolute difference
        frame_diff_max: Max absolute difference
    """
    curr = _to_numpy(current_image)
    prev = _to_numpy(previous_image)

    curr_gray = _to_gray(curr).astype(np.float64) / 255.0
    prev_gray = _to_gray(prev).astype(np.float64) / 255.0

    # Resize if shapes don't match
    if curr_gray.shape != prev_gray.shape:
        from PIL import Image
        prev_pil = Image.fromarray((prev_gray * 255).astype(np.uint8))
        prev_pil = prev_pil.resize((curr_gray.shape[1], curr_gray.shape[0]))
        prev_gray = np.array(prev_pil).astype(np.float64) / 255.0

    diff = np.abs(curr_gray - prev_gray)
    return {
        "frame_diff_mean": float(np.mean(diff)),
        "frame_diff_max": float(np.max(diff)),
    }


def illumination_score(image) -> dict[str, float]:
    """
    Simple illumination analysis: ratio of bright/dark regions.

    Returns:
        illumination_bright_ratio: Fraction of bright pixels (> 200/255)
        illumination_dark_ratio: Fraction of dark pixels (< 50/255)
        illumination_balance: bright_ratio - dark_ratio (positive = bright scene)
    """
    arr = _to_numpy(image)
    gray = _to_gray(arr)

    bright = float(np.mean(gray > 200))
    dark = float(np.mean(gray < 50))

    return {
        "illumination_bright_ratio": bright,
        "illumination_dark_ratio": dark,
        "illumination_balance": bright - dark,
    }


def compute_input_signals(
    image,
    previous_image=None,
    include_expensive: bool = True,
) -> dict[str, float]:
    """
    Compute all input-space signals for a single image.

    Args:
        image: Input image (PIL, numpy, or tensor).
        previous_image: Previous frame for temporal difference (optional).
        include_expensive: Include more expensive computations.

    Returns:
        Dict of signal_name -> value.
    """
    signals = {}

    # Always compute these (cheap)
    signals.update(brightness_stats(image))
    signals.update(contrast_estimate(image))
    signals.update(color_histogram_features(image))
    signals.update(illumination_score(image))

    # Moderately expensive
    signals.update(blur_score(image))
    signals.update(edge_density(image))

    if include_expensive:
        signals.update(noise_estimate(image))
        signals.update(compression_proxy(image))

    # Temporal
    if previous_image is not None:
        signals.update(frame_difference(image, previous_image))

    return signals


# List of all signal names for reference
INPUT_SIGNAL_NAMES = [
    "brightness_mean", "brightness_std",
    "contrast_rms", "contrast_michelson",
    "blur_laplacian_var",
    "noise_estimate_mad",
    "color_r_mean", "color_g_mean", "color_b_mean",
    "color_r_std", "color_g_std", "color_b_std",
    "color_hist_entropy",
    "edge_density", "edge_mean_magnitude",
    "compression_blockiness",
    "frame_diff_mean", "frame_diff_max",
    "illumination_bright_ratio", "illumination_dark_ratio", "illumination_balance",
]
