"""
Synthetic shift injector for controlled distribution shift experiments.

Implements configurable shift families:
- Illumination: brightness, contrast, gamma, color temperature
- Blur: gaussian, motion
- Noise: gaussian, shot, speckle
- Weather-like: fog/haze simulation
- Compression: JPEG artifacts
- Occlusion: random box masking
- Resolution: degradation and rescaling

Each injector:
- Takes severity in [0, 1]
- Is deterministic with seed
- Logs exact parameters used
- Exposes shift family label for diagnosis supervision

Usage:
    injector = SyntheticShiftInjector()
    corrupted = injector.apply(image, family="gaussian_blur", severity=0.5, seed=42)
"""

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass
from PIL import Image, ImageFilter, ImageEnhance
import io


@dataclass
class ShiftParams:
    """Records exact parameters used for a shift application."""
    family: str
    severity: float
    seed: int
    params: dict

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "severity": self.severity,
            "seed": self.seed,
            "params": self.params,
        }


# Registry of shift families
SHIFT_FAMILIES = [
    "brightness",
    "contrast",
    "gamma",
    "color_temp",
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "shot_noise",
    "speckle_noise",
    "fog",
    "jpeg_compression",
    "occlusion",
    "resolution_degradation",
]


def _ensure_pil(image) -> Image.Image:
    """Convert image to PIL if needed."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image)
    # Try torch tensor
    try:
        import torch
        if isinstance(image, torch.Tensor):
            arr = image.permute(1, 2, 0).cpu().numpy()
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
    except ImportError:
        pass
    raise TypeError(f"Unsupported image type: {type(image)}")


def _pil_to_same_type(result: Image.Image, original):
    """Convert result back to the same type as the original."""
    if isinstance(original, Image.Image):
        return result
    if isinstance(original, np.ndarray):
        arr = np.array(result).astype(original.dtype)
        if original.dtype in (np.float32, np.float64):
            arr = arr / 255.0
        return arr
    try:
        import torch
        if isinstance(original, torch.Tensor):
            arr = np.array(result).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
    except ImportError:
        pass
    return result


class SyntheticShiftInjector:
    """
    Applies synthetic distribution shifts to images.

    All methods are deterministic given a seed. Severity is in [0, 1].
    """

    def __init__(self):
        self._last_params: Optional[ShiftParams] = None

    @property
    def last_params(self) -> Optional[ShiftParams]:
        return self._last_params

    def apply(
        self,
        image,
        family: str,
        severity: float,
        seed: int = 42,
    ):
        """
        Apply a shift to an image.

        Args:
            image: Input image (PIL, numpy, or torch Tensor).
            family: Shift family name (see SHIFT_FAMILIES).
            severity: Shift severity in [0, 1].
            seed: Random seed for reproducibility.

        Returns:
            Corrupted image in same format as input.
        """
        severity = float(np.clip(severity, 0.0, 1.0))
        if severity == 0.0:
            self._last_params = ShiftParams(family, 0.0, seed, {})
            return image

        pil_img = _ensure_pil(image)
        rng = np.random.RandomState(seed)

        method_map = {
            "brightness": self._brightness,
            "contrast": self._contrast,
            "gamma": self._gamma,
            "color_temp": self._color_temp,
            "gaussian_blur": self._gaussian_blur,
            "motion_blur": self._motion_blur,
            "gaussian_noise": self._gaussian_noise,
            "shot_noise": self._shot_noise,
            "speckle_noise": self._speckle_noise,
            "fog": self._fog,
            "jpeg_compression": self._jpeg_compression,
            "occlusion": self._occlusion,
            "resolution_degradation": self._resolution_degradation,
        }

        if family not in method_map:
            raise ValueError(
                f"Unknown shift family '{family}'. "
                f"Available: {list(method_map.keys())}"
            )

        result, params = method_map[family](pil_img, severity, rng)
        self._last_params = ShiftParams(family, severity, seed, params)
        return _pil_to_same_type(result, image)

    def __call__(self, image, family: str, severity: float, seed: int = 42):
        """Callable interface for StreamWrapper compatibility."""
        return self.apply(image, family, severity, seed)

    # ---- Shift implementations ----

    def _brightness(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Adjust brightness. severity→ factor from 0.2 (dark) to 1.8 (bright)."""
        # Alternate between darker and brighter based on seed
        if rng.random() > 0.5:
            factor = 1.0 + severity * 0.8  # brighter: 1.0 → 1.8
        else:
            factor = 1.0 - severity * 0.8  # darker: 1.0 → 0.2
        result = ImageEnhance.Brightness(img).enhance(factor)
        return result, {"factor": factor}

    def _contrast(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Adjust contrast."""
        if rng.random() > 0.5:
            factor = 1.0 + severity * 1.0
        else:
            factor = 1.0 - severity * 0.8
        result = ImageEnhance.Contrast(img).enhance(factor)
        return result, {"factor": factor}

    def _gamma(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Apply gamma correction."""
        if rng.random() > 0.5:
            gamma = 1.0 + severity * 1.5  # 1.0 → 2.5
        else:
            gamma = 1.0 / (1.0 + severity * 1.5)  # 1.0 → 0.4
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr), {"gamma": gamma}

    def _color_temp(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Simulate color temperature shift (warm/cool)."""
        arr = np.array(img).astype(np.float32)
        shift = severity * 40  # max 40 unit shift
        if rng.random() > 0.5:
            # Warm (increase red, decrease blue)
            arr[:, :, 0] = np.clip(arr[:, :, 0] + shift, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] - shift, 0, 255)
            direction = "warm"
        else:
            # Cool (decrease red, increase blue)
            arr[:, :, 0] = np.clip(arr[:, :, 0] - shift, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + shift, 0, 255)
            direction = "cool"
        return Image.fromarray(arr.astype(np.uint8)), {"shift": shift, "direction": direction}

    def _gaussian_blur(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Apply Gaussian blur."""
        radius = severity * 5.0  # 0 → 5 pixel radius
        radius = max(0.5, radius)
        result = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return result, {"radius": radius}

    def _motion_blur(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Simulate motion blur using directional kernel."""
        kernel_size = int(3 + severity * 12)  # 3 → 15
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create horizontal motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size

        # Rotate based on random angle
        angle = rng.uniform(0, 180)
        from scipy.ndimage import rotate as scipy_rotate
        kernel = scipy_rotate(kernel, angle, reshape=False, order=1)
        kernel = kernel / kernel.sum()

        from scipy.ndimage import convolve
        arr = np.array(img).astype(np.float32)
        for c in range(arr.shape[2]):
            arr[:, :, c] = convolve(arr[:, :, c], kernel)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr), {"kernel_size": kernel_size, "angle": angle}

    def _gaussian_noise(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Add Gaussian noise."""
        sigma = severity * 50  # 0 → 50
        arr = np.array(img).astype(np.float32)
        noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr), {"sigma": sigma}

    def _shot_noise(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Add shot (Poisson) noise."""
        # Lower lambda = more noise
        lam = max(1, int(60 - severity * 55))  # 60 → 5
        arr = np.array(img).astype(np.float32) / 255.0
        noisy = rng.poisson(arr * lam) / lam
        noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy), {"lambda": lam}

    def _speckle_noise(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Add speckle (multiplicative) noise."""
        sigma = severity * 0.5  # 0 → 0.5
        arr = np.array(img).astype(np.float32) / 255.0
        noise = rng.normal(0, sigma, arr.shape)
        arr = arr + arr * noise
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr), {"sigma": sigma}

    def _fog(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Simulate fog/haze by blending with a bright fog layer."""
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]

        # Create foggy gradient (brighter at center/bottom)
        y_coords = np.linspace(0, 1, h)[:, None]
        fog_map = 0.5 + 0.5 * y_coords  # stronger fog at bottom
        fog_map = np.broadcast_to(fog_map, (h, w))

        # Add some spatial randomness
        noise = rng.uniform(0.8, 1.2, (h // 8 + 1, w // 8 + 1))
        from PIL import Image as PilImage
        noise_img = PilImage.fromarray(noise.astype(np.float32))
        noise_img = noise_img.resize((w, h), PilImage.BILINEAR)
        noise_large = np.array(noise_img)
        fog_map = fog_map * noise_large

        alpha = severity * 0.7  # max 70% fog blend
        fog_color = 220.0  # light gray fog

        for c in range(3):
            arr[:, :, c] = arr[:, :, c] * (1 - alpha * fog_map) + fog_color * alpha * fog_map

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr), {"alpha": alpha, "fog_color": fog_color}

    def _jpeg_compression(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Apply JPEG compression artifacts."""
        # quality: 95 (mild) → 5 (extreme)
        quality = int(95 - severity * 90)
        quality = max(1, quality)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        result = Image.open(buffer).copy()
        return result, {"quality": quality}

    def _occlusion(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Add random rectangular occlusion patches."""
        arr = np.array(img).copy()
        h, w = arr.shape[:2]
        num_patches = int(1 + severity * 5)  # 1 → 6 patches
        patch_size_frac = 0.05 + severity * 0.1  # %5 → 15% of image

        patches = []
        for _ in range(num_patches):
            ph = int(h * patch_size_frac * rng.uniform(0.5, 1.5))
            pw = int(w * patch_size_frac * rng.uniform(0.5, 1.5))
            py = rng.randint(0, max(1, h - ph))
            px = rng.randint(0, max(1, w - pw))
            # Fill with mean color + noise
            fill_val = rng.randint(0, 128, size=3).astype(np.uint8)
            arr[py:py+ph, px:px+pw] = fill_val
            patches.append({"x": px, "y": py, "w": pw, "h": ph})

        return Image.fromarray(arr), {"num_patches": num_patches, "patches": patches}

    def _resolution_degradation(
        self, img: Image.Image, severity: float, rng: np.random.RandomState
    ) -> tuple[Image.Image, dict]:
        """Degrade resolution by downscaling then upscaling."""
        scale = 1.0 / (1.0 + severity * 3.0)  # 1.0 → 0.25
        w, h = img.size
        new_w = max(8, int(w * scale))
        new_h = max(8, int(h * scale))
        small = img.resize((new_w, new_h), Image.BILINEAR)
        result = small.resize((w, h), Image.BILINEAR)
        return result, {"scale_factor": scale, "intermediate_size": (new_w, new_h)}


def get_shift_families() -> list[str]:
    """Return list of available shift families."""
    return SHIFT_FAMILIES.copy()


def create_injector() -> SyntheticShiftInjector:
    """Factory function for creating a shift injector."""
    return SyntheticShiftInjector()
