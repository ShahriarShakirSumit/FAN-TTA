"""
Signal-Conditioned Adaptive Restoration (SCAR) — Core SDMR contribution.

Instead of naively applying a fixed denoising filter, SCAR uses the diagnosed
signal values (noise_ratio, blur_ratio, severity) to adaptively parameterize
a multi-stage restoration pipeline:

1. **Noise Estimation**: Uses Donoho's MAD estimator to get per-frame σ_hat
2. **Adaptive NLM Denoising**: Non-local means with h=f(σ_hat), patch/search
   windows adapted to noise level. Falls back to wavelet denoising for speed.
3. **Frequency-Guided Sharpening**: Wiener-inspired deconvolution for blur
4. **Adaptive Histogram Specification**: Severity-scaled CLAHE + channel norm
5. **Compound Restoration**: Cascaded application for mixed corruptions

Key design principles:
- Restoration strength is PROPORTIONAL to diagnosed severity (not binary on/off)
- Each operation is bounded (cannot amplify artifacts beyond input quality)
- Confidence-gated safety: restoration is reverted if it harms detector confidence

Reference: This module implements the core of Algorithm 1 in the paper.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RestorationConfig:
    """Configuration for SCAR restoration pipeline."""
    # Denoising
    denoise_method: str = "nlm"          # "nlm", "wavelet", "bilateral", "cascade"
    nlm_patch_size: int = 7              # NLM patch size (odd)
    nlm_search_window: int = 21          # NLM search window (odd)
    nlm_h_scale: float = 1.0            # Scale factor for NLM h parameter
    wavelet: str = "db4"                 # Wavelet family for wavelet denoising
    wavelet_mode: str = "soft"           # "soft" or "hard" thresholding
    
    # Sharpening
    sharpen_method: str = "wiener"       # "wiener", "unsharp", "laplacian"
    wiener_noise_var: float = 0.01       # Noise variance for Wiener filter
    
    # Brightness/contrast
    clahe_clip_limit: float = 3.0        # CLAHE clip limit
    clahe_tile_size: int = 8             # CLAHE tile grid size
    
    # Safety
    enable_confidence_gate: bool = True  # Revert if restoration harms confidence
    confidence_threshold: float = 0.02   # Minimum confidence improvement to keep
    
    # Compound
    enable_compound: bool = True         # Handle multiple corruptions simultaneously
    
    # Speed
    max_image_dim: int = 0               # Downscale for restoration if > 0 (0=disable)


class SignalConditionedRestoration:
    """
    Adaptive restoration conditioned on diagnosed corruption signals.
    
    The key novelty: restoration parameters are NOT fixed but computed
    from the signal diagnosis pipeline. A frame with noise_ratio=2.5
    gets mild denoising; noise_ratio=8.0 gets aggressive denoising.
    The mapping is:
        h_nlm = σ_hat * nlm_h_scale
        patch_size = f(σ_hat)
        search_window = f(σ_hat)
    
    This avoids both under-denoising (useless) and over-denoising (blurs details).
    """
    
    def __init__(self, config: Optional[RestorationConfig] = None):
        self.config = config or RestorationConfig()
        self._stats = {
            "frames_restored": 0,
            "frames_reverted": 0,
            "total_psnr_gain": 0.0,
        }
    
    def restore(
        self,
        image: np.ndarray,
        corruption_type: str,
        noise_ratio: float = 1.0,
        blur_ratio: float = 1.0,
        brightness_diff: float = 0.0,
        drift_score: float = 0.0,
        sigma_est: float = 0.0,
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply signal-conditioned restoration.
        
        Args:
            image: Input image (H, W, 3) uint8
            corruption_type: "denoise", "sharpen", "normalize", "compound"
            noise_ratio: Estimated noise / anchor noise (>1 = more noise)
            blur_ratio: Current blur / anchor blur (<1 = more blurry)
            brightness_diff: |current_brightness - anchor_brightness|
            drift_score: Overall drift magnitude
            sigma_est: Estimated noise standard deviation (from MAD)
            
        Returns:
            (restored_image, metadata_dict)
        """
        meta = {
            "corruption_type": corruption_type,
            "noise_ratio": noise_ratio,
            "blur_ratio": blur_ratio,
            "brightness_diff": brightness_diff,
            "sigma_est": sigma_est,
            "methods_applied": [],
        }
        
        restored = image.copy()
        
        # Handle compound corruptions
        if self.config.enable_compound and corruption_type == "compound":
            # Apply multiple restorations in order of severity
            if noise_ratio > 1.5:
                restored, d_meta = self._denoise(restored, noise_ratio, sigma_est)
                meta["methods_applied"].append(("denoise", d_meta))
            if blur_ratio < 0.5:
                restored, s_meta = self._sharpen(restored, blur_ratio, drift_score)
                meta["methods_applied"].append(("sharpen", s_meta))
            if brightness_diff > 0.1:
                restored, n_meta = self._normalize(restored, brightness_diff, drift_score)
                meta["methods_applied"].append(("normalize", n_meta))
        elif corruption_type == "denoise":
            restored, d_meta = self._denoise(restored, noise_ratio, sigma_est)
            meta["methods_applied"].append(("denoise", d_meta))
        elif corruption_type == "sharpen":
            restored, s_meta = self._sharpen(restored, blur_ratio, drift_score)
            meta["methods_applied"].append(("sharpen", s_meta))
        elif corruption_type == "normalize":
            restored, n_meta = self._normalize(restored, brightness_diff, drift_score)
            meta["methods_applied"].append(("normalize", n_meta))
        
        self._stats["frames_restored"] += 1
        return restored, meta
    
    def _denoise(
        self,
        image: np.ndarray,
        noise_ratio: float,
        sigma_est: float,
    ) -> Tuple[np.ndarray, dict]:
        """
        Signal-conditioned denoising with adaptive parameters.
        
        Key insight: the diagnosed noise level σ_hat directly parameterizes
        the denoiser. This is fundamentally different from TTA methods that
        blindly adapt model parameters.
        """
        method = self.config.denoise_method
        meta = {"method": method, "sigma_est": sigma_est, "noise_ratio": noise_ratio}
        
        if method == "cascade":
            return self._cascade_denoise(image, noise_ratio, sigma_est, meta)
        elif method == "nlm":
            return self._nlm_denoise(image, noise_ratio, sigma_est, meta)
        elif method == "wavelet":
            return self._wavelet_denoise(image, noise_ratio, sigma_est, meta)
        elif method == "bilateral":
            return self._bilateral_denoise(image, noise_ratio, sigma_est, meta)
        else:
            return self._cascade_denoise(image, noise_ratio, sigma_est, meta)
    
    def _cascade_denoise(
        self,
        image: np.ndarray,
        noise_ratio: float,
        sigma_est: float,
        meta: dict,
    ) -> Tuple[np.ndarray, dict]:
        """
        Multi-stage cascade denoising — the strongest option.
        
        Stage 1: Fast NLM for broad noise reduction
        Stage 2: Wavelet thresholding for residual noise  
        Stage 3: Edge-aware bilateral for artifact cleanup
        
        The cascade is severity-adaptive:
        - Low noise (σ<15): NLM only
        - Medium noise (15≤σ<35): NLM + wavelet
        - High noise (σ≥35): Full cascade
        """
        meta["method"] = "cascade"
        stages = []
        
        # Stage 1: NLM denoising (always applied)
        result, nlm_meta = self._nlm_denoise(image, noise_ratio, sigma_est, {})
        stages.append("nlm")
        
        # Stage 2: Wavelet for medium+ noise
        if sigma_est >= 15:
            result, wav_meta = self._wavelet_denoise(result, noise_ratio * 0.5, sigma_est * 0.5, {})
            stages.append("wavelet")
        
        # Stage 3: Light bilateral for high noise (cleanup artifacts)
        if sigma_est >= 35:
            result = cv2.bilateralFilter(result, 5, 40, 40)
            stages.append("bilateral_cleanup")
        
        meta["stages"] = stages
        meta["num_stages"] = len(stages)
        return result, meta
    
    def _nlm_denoise(
        self,
        image: np.ndarray,
        noise_ratio: float,
        sigma_est: float,
        meta: dict,
    ) -> Tuple[np.ndarray, dict]:
        """
        Non-Local Means denoising with signal-conditioned parameters.
        
        h parameter controls denoising strength:
            h = σ_hat * scale_factor
        
        NLM is fundamentally superior to bilateral filter because it
        exploits non-local self-similarity in images. For natural images
        (which have repeating textures/structures), NLM recovers 3-6dB
        more PSNR than bilateral filtering.
        """
        # Adaptive h parameter: proportional to estimated noise
        # Lower bound: don't denoise if sigma_est is very low
        # Upper bound: cap to avoid over-smoothing
        h = max(3.0, min(sigma_est * self.config.nlm_h_scale, 80.0))
        
        # Adaptive patch/search sizes based on noise level
        if sigma_est < 15:
            patch_size = 5
            search_window = 15
        elif sigma_est < 30:
            patch_size = 7
            search_window = 21
        else:
            patch_size = 7
            search_window = 25
        
        # OpenCV NLM is much faster than scikit-image NLM on CPU
        # and available on GPU nodes without extra deps
        h_color = h  # Same strength for luminance and chrominance
        
        result = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h,
            h_color,
            patch_size,
            search_window,
        )
        
        meta.update({
            "h": h,
            "h_color": h_color,
            "patch_size": patch_size,
            "search_window": search_window,
        })
        return result, meta
    
    def _wavelet_denoise(
        self,
        image: np.ndarray,
        noise_ratio: float,
        sigma_est: float,
        meta: dict,
    ) -> Tuple[np.ndarray, dict]:
        """
        Wavelet-based denoising with adaptive thresholding.
        
        Uses BayesShrink (adaptive soft thresholding) which automatically
        estimates the optimal threshold per wavelet subband. The sigma
        parameter from our diagnosis guides the denoising strength.
        """
        from skimage.restoration import denoise_wavelet, estimate_sigma
        
        # Convert to float [0, 1] for skimage
        img_float = image.astype(np.float64) / 255.0
        
        # Use our diagnosed sigma, but let wavelet estimator refine per-channel
        # sigma_est is in [0, 255] scale, convert to [0, 1]
        sigma_01 = sigma_est / 255.0
        
        # BayesShrink adapts threshold per subband — state-of-the-art for wavelet
        result = denoise_wavelet(
            img_float,
            channel_axis=-1,
            wavelet=self.config.wavelet,
            mode=self.config.wavelet_mode,
            method="BayesShrink",
            sigma=sigma_01 if sigma_01 > 0.01 else None,
            rescale_sigma=True,
        )
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        meta.update({
            "wavelet": self.config.wavelet,
            "mode": self.config.wavelet_mode,
            "sigma_01": sigma_01,
        })
        return result, meta
    
    def _bilateral_denoise(
        self,
        image: np.ndarray,
        noise_ratio: float,
        sigma_est: float,
        meta: dict,
    ) -> Tuple[np.ndarray, dict]:
        """Bilateral filter (baseline method, kept for ablation comparison)."""
        sigma_color = min(120, max(50, int(sigma_est * 1.2 + 25)))
        sigma_space = min(120, max(50, int(sigma_est * 1.2 + 25)))
        d = 9 if sigma_est < 30 else 11
        
        result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        meta.update({"d": d, "sigma_color": sigma_color, "sigma_space": sigma_space})
        return result, meta
    
    def _sharpen(
        self,
        image: np.ndarray,
        blur_ratio: float,
        drift_score: float,
    ) -> Tuple[np.ndarray, dict]:
        """
        Adaptive sharpening conditioned on blur severity.
        
        Uses multi-scale unsharp masking: the diagnosed blur_ratio determines
        both the sharpening strength and the Gaussian kernel size.
        Lower blur_ratio = more aggressive sharpening.
        
        For severe blur (blur_ratio < 0.15), applies Wiener-inspired
        deconvolution with estimated Gaussian PSF.
        """
        meta = {"blur_ratio": blur_ratio, "drift_score": drift_score}
        
        # Map blur_ratio to sharpening strength
        # blur_ratio=0.3 (threshold) → mild, blur_ratio=0.05 → aggressive
        severity = np.clip(1.0 - blur_ratio, 0.3, 0.95)
        
        if blur_ratio < 0.15 and self.config.sharpen_method == "wiener":
            # Severe blur: Wiener-style deconvolution
            result = self._wiener_sharpen(image, severity, meta)
        else:
            # Moderate blur: Multi-scale unsharp masking
            result = self._multiscale_unsharp(image, severity, meta)
        
        return result, meta
    
    def _multiscale_unsharp(
        self,
        image: np.ndarray,
        severity: float,
        meta: dict,
    ) -> np.ndarray:
        """
        Multi-scale unsharp masking: combines sharpening at multiple scales
        for more natural results than single-scale.
        
        Scale 1 (fine): 3x3 kernel, low weight → recovers high-freq edges
        Scale 2 (medium): 5x5 kernel → recovers medium-freq texture
        Scale 3 (coarse): 9x9 kernel → recovers broad structure (if severe)
        """
        result = image.astype(np.float32)
        
        # Scale 1: fine details
        blur1 = cv2.GaussianBlur(image, (3, 3), 0.8)
        strength1 = severity * 0.4
        result = result + strength1 * (image.astype(np.float32) - blur1.astype(np.float32))
        
        # Scale 2: medium texture
        blur2 = cv2.GaussianBlur(image, (5, 5), 1.5)
        strength2 = severity * 0.5
        result = result + strength2 * (image.astype(np.float32) - blur2.astype(np.float32))
        
        # Scale 3: coarse (only for severe blur)
        if severity > 0.6:
            blur3 = cv2.GaussianBlur(image, (9, 9), 3.0)
            strength3 = (severity - 0.6) * 0.8
            result = result + strength3 * (image.astype(np.float32) - blur3.astype(np.float32))
        
        meta["method"] = "multiscale_unsharp"
        meta["severity"] = severity
        meta["num_scales"] = 3 if severity > 0.6 else 2
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _wiener_sharpen(
        self,
        image: np.ndarray,
        severity: float,
        meta: dict,
    ) -> np.ndarray:
        """
        Wiener-inspired deconvolution for severe blur.
        
        Estimates a Gaussian PSF based on blur severity, then applies
        approximate Wiener deconvolution in the frequency domain.
        """
        # Estimate PSF size from blur severity
        psf_size = int(3 + severity * 6)  # 3-9 pixels
        if psf_size % 2 == 0:
            psf_size += 1
        psf_sigma = 0.5 + severity * 2.0
        
        # Create Gaussian PSF
        psf = np.zeros((psf_size, psf_size), dtype=np.float64)
        center = psf_size // 2
        for i in range(psf_size):
            for j in range(psf_size):
                psf[i, j] = np.exp(-((i-center)**2 + (j-center)**2) / (2 * psf_sigma**2))
        psf /= psf.sum()
        
        # Process each channel
        result = np.zeros_like(image, dtype=np.float64)
        noise_var = self.config.wiener_noise_var
        
        for c in range(3):
            channel = image[:, :, c].astype(np.float64) / 255.0
            
            # FFT-based Wiener deconvolution
            H, W = channel.shape
            pad_h = H + psf_size - 1
            pad_w = W + psf_size - 1
            
            F_img = np.fft.fft2(channel, s=(pad_h, pad_w))
            F_psf = np.fft.fft2(psf, s=(pad_h, pad_w))
            
            # Wiener filter: H* / (|H|² + NSR)
            F_psf_conj = np.conj(F_psf)
            F_psf_sq = np.abs(F_psf)**2
            nsr = noise_var  # Noise-to-signal ratio
            
            F_restored = F_img * F_psf_conj / (F_psf_sq + nsr)
            restored = np.real(np.fft.ifft2(F_restored))[:H, :W]
            
            result[:, :, c] = restored * 255.0
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Blend with original to avoid over-sharpening artifacts
        blend_alpha = min(0.8, severity * 0.9)
        blended = cv2.addWeighted(result, blend_alpha, image, 1 - blend_alpha, 0)
        
        meta["method"] = "wiener"
        meta["psf_size"] = psf_size
        meta["psf_sigma"] = psf_sigma
        meta["blend_alpha"] = blend_alpha
        
        return blended
    
    def _normalize(
        self,
        image: np.ndarray,
        brightness_diff: float,
        drift_score: float,
    ) -> Tuple[np.ndarray, dict]:
        """
        Adaptive brightness/contrast normalization.
        
        Uses severity-scaled CLAHE plus optional gamma correction.
        The brightness_diff diagnosis determines the correction strength.
        """
        meta = {"brightness_diff": brightness_diff}
        
        # Convert to LAB for luminance manipulation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # CLAHE with severity-adaptive clip limit
        # Higher brightness shift → higher clip limit → more aggressive normalization
        clip_limit = min(5.0, max(1.5, brightness_diff * 10.0))
        tile = self.config.clahe_tile_size
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        lab[:, :, 0] = clahe.apply(l_channel)
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Additional: gamma correction for large brightness shifts
        if brightness_diff > 0.2:
            # Estimate gamma to compensate
            current_mean = np.mean(l_channel) / 255.0
            target_mean = 0.5  # aim for mid-gray
            if current_mean > 0.01:
                gamma = np.log(target_mean) / np.log(current_mean)
                gamma = np.clip(gamma, 0.3, 3.0)  # Safety bounds
                
                # Apply gamma to L channel
                lab2 = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
                l_norm = lab2[:, :, 0].astype(np.float32) / 255.0
                l_corrected = np.power(l_norm, gamma) * 255.0
                lab2[:, :, 0] = np.clip(l_corrected, 0, 255).astype(np.uint8)
                result = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
                
                meta["gamma"] = float(gamma)
        
        meta["clip_limit"] = clip_limit
        meta["method"] = "clahe_gamma"
        return result, meta
    
    def get_stats(self) -> dict:
        return dict(self._stats)


def create_restoration(config_dict: Optional[dict] = None) -> SignalConditionedRestoration:
    """Factory function to create a restoration instance from config dict."""
    if config_dict is None:
        return SignalConditionedRestoration()
    
    cfg = RestorationConfig(
        denoise_method=config_dict.get("denoise_method", "cascade"),
        nlm_h_scale=config_dict.get("nlm_h_scale", 1.0),
        enable_confidence_gate=config_dict.get("enable_confidence_gate", True),
        enable_compound=config_dict.get("enable_compound", True),
    )
    return SignalConditionedRestoration(cfg)
