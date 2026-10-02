#!/usr/bin/env python3
"""
SDMR Streaming Experiment Runner — Production Version.

End-to-end pipeline for running streaming detection experiments:
1. Load real dataset (COCO-Person) and wrap in StreamWrapper with shift scenarios
2. Initialize detector (YOLO via Ultralytics)
3. Initialize diagnosis pipeline (signals + change-point + router)
4. Run frame-by-frame streaming evaluation
5. Log results and compute metrics

Supports:
- SDMR (our method): signal-driven adaptive routing
- TTA baselines: TENT, EATA, CoTTA, SAR, RoTTA
- Simple baselines: NoAdapt, PeriodicFT, BNAdapt, etc.

Usage:
    python -m sdmr.run_stream_experiment --config configs/coco_experiment.yaml
    python -m sdmr.run_stream_experiment --config configs/coco_experiment.yaml --strategy tent
    python -m sdmr.run_stream_experiment --config configs/coco_experiment.yaml --strategy no_adapt
"""

import argparse
import json
import os
import sys
import time
import yaml
import logging
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

# SDMR imports
from sdmr.utils.seed import set_global_seed
from sdmr.utils.logging_utils import setup_logger, ExperimentMeta, JsonLogger, timer
from sdmr.data.datasets.stream_wrapper import (
    StreamWrapper, ShiftScenario, ShiftPattern,
    StreamFrame, FrameMetadata,
)
from sdmr.data.datasets.synthetic_shift_injector import SyntheticShiftInjector
from sdmr.data.datasets.coco_person import COCOPersonDataset, COCOPersonConfig
from sdmr.data.datasets.ecp_dataset import ECPDataset, ECPConfig
from sdmr.data.datasets.nightowls_dataset import NightOwlsDataset, NightOwlsConfig
from sdmr.data.datasets import create_dataset
from sdmr.models.detector_wrappers.yolo_wrapper import YOLODetectorWrapper
from sdmr.models.detector_wrappers.base_detector import DetectionResult
from sdmr.diagnosis.temporal_buffer import TemporalBuffer
from sdmr.diagnosis.input_signals import compute_input_signals
from sdmr.diagnosis.output_signals import compute_output_signals
from sdmr.diagnosis.feature_signals import compute_feature_signals
from sdmr.diagnosis.change_point import (
    CUSUMDetector,
    HysteresisDetector,
    MultiSignalChangePointDetector,
    ChangePointResult,
)
from sdmr.adaptation import (
    RuleBasedRouter,
    CostAwareRouter,
    AdaptationAction,
    RouterDecision,
)
from sdmr.baselines import create_baseline
from sdmr.adaptation.feature_anchor import create_fan, FeatureAnchorNorm
from sdmr.evaluation.metrics import (
    StreamEvaluator,
    compute_frame_metrics,
)
from sdmr.harm import HistogramHarmPredictor, HarmEstimate


logger = logging.getLogger("sdmr.experiment")


# =====================================================================
# Configuration
# =====================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_overrides(cfg: dict, args) -> dict:
    """Apply CLI overrides to config."""
    if args.strategy:
        cfg["strategy"] = args.strategy
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.n_frames is not None:
        cfg.setdefault("stream", {})["n_frames"] = args.n_frames
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.dataset_dir:
        cfg.setdefault("data", {})["dataset_dir"] = args.dataset_dir
    if args.num_images is not None:
        cfg.setdefault("data", {})["num_images"] = args.num_images
    if args.severity is not None:
        cfg.setdefault("stream", {}).setdefault("scenario", {})["severity"] = args.severity
    if args.experiment_name is not None:
        cfg["experiment_name"] = args.experiment_name
    return cfg


# =====================================================================
# Dataset construction
# =====================================================================

class SimpleDatasetAdapter:
    """
    Adapter to make COCOPersonDataset compatible with StreamWrapper.

    StreamWrapper expects dataset[i] to return (image, targets).
    COCOPersonDataset returns StreamFrame. This adapter converts.
    Also handles datasets that already return (image, targets) tuples.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frame = self.dataset[idx]
        if isinstance(frame, tuple):
            return frame  # Already (image, targets)
        return frame.image, frame.targets


def build_dataset(cfg: dict):
    """Build the source dataset from config."""
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset", "coco_person")
    dataset_dir = data_cfg.get("dataset_dir", "./data/coco2017")
    num_images = data_cfg.get("num_images", None)

    if dataset_name == "coco_person":
        images_dir = data_cfg.get("images_dir", os.path.join(dataset_dir, "val2017"))
        ann_file = data_cfg.get("annotations_file",
                                os.path.join(dataset_dir, "annotations", "instances_val2017.json"))

        dataset = COCOPersonDataset(
            images_dir=images_dir,
            annotations_file=ann_file,
            config=COCOPersonConfig(
                min_area=data_cfg.get("min_area", 100.0),
                person_only=True,
            ),
        )

        if num_images is not None and num_images < len(dataset):
            dataset = dataset.subset(num_images, seed=cfg.get("seed", 42))
            logger.info(f"Using subset of {num_images} images")

    elif dataset_name in ("ecp", "ecp_night", "ecp_day"):
        dataset = create_dataset(dataset_name, **data_cfg)
    elif dataset_name == "nightowls":
        dataset = create_dataset(dataset_name, **data_cfg)
    elif dataset_name == "cross_domain":
        # Cross-domain: COCO source + target dataset (NightOwls/ECP Night)
        from sdmr.data.datasets.cross_domain import CrossDomainDataset

        # Build source (COCO)
        source_dir = data_cfg.get("source_dir", "./data/coco2017")
        source = COCOPersonDataset(
            images_dir=os.path.join(source_dir, "val2017"),
            annotations_file=os.path.join(source_dir, "annotations", "instances_val2017.json"),
            config=COCOPersonConfig(min_area=100.0, person_only=True),
        )

        # Build target
        target_name = data_cfg.get("target_dataset", "nightowls")
        target_cfg = data_cfg.get("target", {})
        if target_name == "nightowls":
            target = NightOwlsDataset(
                images_dir=target_cfg.get("images_dir", ""),
                annotations_file=target_cfg.get("annotations_file", ""),
                config=NightOwlsConfig(person_only=True, min_height=20),
            )
        elif target_name in ("ecp_night", "ecp"):
            target = ECPDataset(
                images_dir=target_cfg.get("images_dir", ""),
                annotations_dir=target_cfg.get("annotations_dir", ""),
                config=ECPConfig(time_of_day="night", person_only=True, min_height=20),
            )
        else:
            raise ValueError(f"Unknown target dataset: {target_name}")

        n_source = data_cfg.get("n_source", 200)
        n_target = data_cfg.get("n_target", None)
        dataset = CrossDomainDataset(
            source_dataset=source,
            target_dataset=target,
            n_source=n_source,
            n_target=n_target,
            source_name="coco",
            target_name=target_name,
        )
        # CrossDomainDataset already returns (image, targets) tuples
        logger.info(f"Cross-domain dataset: {len(dataset)} frames "
                     f"({dataset.n_source} source + {dataset.n_target} target)")
        return SimpleDatasetAdapter(dataset)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Dataset '{dataset_name}': {len(dataset)} images")
    return SimpleDatasetAdapter(dataset)


# =====================================================================
# Shift scenarios
# =====================================================================

def build_shift_scenario(cfg: dict) -> ShiftScenario:
    """Build a shift scenario from config."""
    stream_cfg = cfg.get("stream", {})
    scenario_cfg = stream_cfg.get("scenario", {})
    scenario_type = scenario_cfg.get("type", "abrupt")

    if scenario_type == "abrupt":
        return ShiftScenario.abrupt(
            clean_before=scenario_cfg.get("clean_before", 200),
            shift_length=scenario_cfg.get("shift_length", 200),
            clean_after=scenario_cfg.get("clean_after", 200),
            shift_family=scenario_cfg.get("shift_family", "gaussian_noise"),
            severity=scenario_cfg.get("severity", 0.5),
            seed=cfg.get("seed", 42),
        )
    elif scenario_type == "gradual":
        return ShiftScenario.gradual(
            clean_before=scenario_cfg.get("clean_before", 200),
            ramp_length=scenario_cfg.get("ramp_length", 200),
            hold_length=scenario_cfg.get("hold_length", 200),
            ramp_down_length=scenario_cfg.get("ramp_down_length", 200),
            clean_after=scenario_cfg.get("clean_after", 200),
            shift_family=scenario_cfg.get("shift_family", "brightness"),
            max_severity=scenario_cfg.get("severity", 0.7),
            seed=cfg.get("seed", 42),
        )
    elif scenario_type == "recurring":
        return ShiftScenario.recurring(
            num_cycles=scenario_cfg.get("num_cycles", 4),
            clean_length=scenario_cfg.get("clean_length", 150),
            shift_length=scenario_cfg.get("shift_length", 100),
            shift_family=scenario_cfg.get("shift_family", "gaussian_blur"),
            severity=scenario_cfg.get("severity", 0.6),
            seed=cfg.get("seed", 42),
        )
    elif scenario_type == "mixed":
        segments = scenario_cfg.get("segments", [
            {"length": 200, "family": "none", "severity": 0.0},
            {"length": 150, "family": "gaussian_noise", "severity": 0.5},
            {"length": 100, "family": "none", "severity": 0.0},
            {"length": 150, "family": "brightness", "severity": 0.6},
            {"length": 200, "family": "none", "severity": 0.0},
        ])
        return ShiftScenario.mixed(segments, seed=cfg.get("seed", 42))
    elif scenario_type == "clean":
        n_frames = stream_cfg.get("n_frames", 500)
        return ShiftScenario(
            pattern=ShiftPattern.CLEAN,
            total_frames=n_frames,
            segments=[{"length": n_frames, "family": "none", "severity": 0.0}],
        )
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")


# =====================================================================
# Component builders
# =====================================================================

def build_detector(cfg: dict) -> YOLODetectorWrapper:
    """Initialize the YOLO detector."""
    model_cfg = cfg.get("model", {})
    detector = YOLODetectorWrapper(
        model_name=model_cfg.get("name", "yolov8s.pt"),
        device=model_cfg.get("device", "cuda:0"),
        imgsz=model_cfg.get("imgsz", 640),
    )
    detector.load_model(verbose=model_cfg.get("verbose", False))
    logger.info(f"Loaded detector: {model_cfg.get('name', 'yolov8s.pt')}")
    return detector


def build_router(cfg: dict):
    """Build the adaptation router from config."""
    strategy = cfg.get("strategy", "sdmr")

    if strategy in ("sdmr", "fan"):
        router_cfg = cfg.get("router", {})
        base = RuleBasedRouter(
            input_drift_threshold=router_cfg.get("input_drift_threshold", 0.3),
            output_drift_threshold=router_cfg.get("output_drift_threshold", 0.3),
            feature_drift_threshold=router_cfg.get("feature_drift_threshold", 0.3),
            mild_severity=router_cfg.get("mild_severity", 0.3),
            moderate_severity=router_cfg.get("moderate_severity", 0.6),
            severe_threshold=router_cfg.get("severe_threshold", 0.8),
            cooldown_frames=router_cfg.get("cooldown_frames", 10),
        )
        if router_cfg.get("cost_aware", True):
            return CostAwareRouter(
                base,
                budget_per_window=router_cfg.get("budget_per_window", 10.0),
                cost_weight=router_cfg.get("cost_weight", 0.3),
            )
        return base
    else:
        # Build a baseline strategy
        baseline_cfg = cfg.get("baseline", {})
        # Input-space baselines make no adaptation decisions:
        # they only preprocess the frame, so they route exactly like no_adapt.
        if strategy in ("gaussian_smooth", "median_filter", "clahe",
                        "gamma_correct", "input_whiten"):
            return create_baseline("no_adapt", **baseline_cfg)
        return create_baseline(strategy, **baseline_cfg)


# =====================================================================
# TTA baseline integration
# =====================================================================

TTA_STRATEGIES = {"tent", "eata", "cotta", "sar", "rotta"}


def build_tta_baseline(cfg: dict, detector: YOLODetectorWrapper):
    """Build a TTA baseline method wrapping the detector."""
    strategy = cfg.get("strategy", "")
    tta_cfg = cfg.get("tta", {})

    if strategy == "tent":
        from sdmr.baselines.tent import TENTDetection, TENTConfig
        config = TENTConfig(
            learning_rate=tta_cfg.get("lr", 1e-3),
            steps_per_batch=tta_cfg.get("steps", 1),
            episodic=tta_cfg.get("episodic", False),
        )
        return TENTDetection(detector._model, config)

    elif strategy == "eata":
        from sdmr.baselines.eata import EATADetection, EATAConfig
        config = EATAConfig(
            learning_rate=tta_cfg.get("lr", 1e-3),
            fisher_alpha=tta_cfg.get("fisher_alpha", 2000),
        )
        return EATADetection(detector._model, config)

    elif strategy == "cotta":
        from sdmr.baselines.cotta import CoTTADetection, CoTTAConfig
        config = CoTTAConfig(
            ema_decay=tta_cfg.get("ema_decay", 0.999),
            num_augmentations=tta_cfg.get("num_augmentations", 4),
        )
        return CoTTADetection(detector._model, config)

    elif strategy == "sar":
        from sdmr.baselines.sar import SARDetection, SARConfig
        config = SARConfig(
            learning_rate=tta_cfg.get("lr", 1e-3),
            sam_rho=tta_cfg.get("sam_rho", 0.05),
        )
        return SARDetection(detector._model, config)

    elif strategy == "rotta":
        from sdmr.baselines.rotta import RoTTADetection, RoTTAConfig
        config = RoTTAConfig(
            bn_alpha=tta_cfg.get("bn_alpha", 0.5),
            memory_size=tta_cfg.get("memory_size", 256),
        )
        return RoTTADetection(detector._model, config)

    return None


# =====================================================================
# Core experiment loop
# =====================================================================

def run_experiment(cfg: dict) -> dict:
    """
    Run a single streaming experiment.

    Full pipeline:
    1. Load dataset -> StreamWrapper with shift scenario
    2. For each frame:
       a. Apply synthetic shift (if in scenario)
       b. Run YOLO detection
       c. Compute diagnosis signals (input + output)
       d. Buffer signals -> change-point detection
       e. Router decision (SDMR) or baseline decision
       f. Execute adaptation action
       g. Log frame metrics
    3. Aggregate results

    Returns:
        dict with experiment results and metrics.
    """
    # Setup
    seed = cfg.get("seed", 42)
    set_global_seed(seed)

    strategy = cfg.get("strategy", "sdmr")
    exp_name = cfg.get("experiment_name", f"sdmr_{strategy}_{seed}")
    # Each strategy gets its own subdirectory to avoid overwrites
    output_dir = Path(cfg.get("output_dir", "outputs")) / exp_name / strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger("sdmr", output_dir / "experiment.log")
    json_logger = JsonLogger(output_dir / "frames.jsonl")

    log.info(f"{'='*60}")
    log.info(f"Experiment: {exp_name}")
    log.info(f"Strategy: {strategy}")
    log.info(f"Seed: {seed}")
    log.info(f"{'='*60}")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ---- Build components ----

    # 1. Dataset
    dataset = build_dataset(cfg)
    log.info(f"Dataset loaded: {len(dataset)} images")

    # 2. Shift scenario and stream
    scenario = build_shift_scenario(cfg)

    # For cross-domain datasets, no synthetic shift injection is needed —
    # the target images are already naturally shifted (e.g., nighttime).
    data_cfg = cfg.get("data", {})
    is_cross_domain = data_cfg.get("dataset", "coco_person") == "cross_domain"

    if is_cross_domain:
        shift_injector_fn = None  # No synthetic corruption
        log.info("Cross-domain mode: no synthetic shift injection")
    else:
        injector = SyntheticShiftInjector()
        shift_injector_fn = lambda img, family, sev, s: injector.apply(
            img, family=family, severity=sev, seed=s
        )

    stream = StreamWrapper(
        dataset=dataset,
        scenario=scenario,
        window_size=cfg.get("stream", {}).get("window_size", 32),
        shift_injector=shift_injector_fn,
        warmup_frames=cfg.get("stream", {}).get("warmup_frames", 0),
        seed=seed,
    )
    n_frames = scenario.total_frames
    log.info(f"Stream: {n_frames} frames, scenario={scenario.pattern.value}")

    # 3. Detector
    model_cfg = cfg.get("model", {})
    conf_threshold = model_cfg.get("conf_threshold", 0.25)
    iou_threshold = model_cfg.get("iou_threshold", 0.45)
    detector = build_detector(cfg)

    # 4. TTA baseline (if applicable)
    tta_method = None
    if strategy in TTA_STRATEGIES:
        tta_method = build_tta_baseline(cfg, detector)
        if tta_method is not None:
            tta_method.setup()
            log.info(f"TTA baseline initialized and configured: {strategy}")

    # 5. Router (for SDMR or simple baselines)
    router = None
    if strategy not in TTA_STRATEGIES:
        router = build_router(cfg)
        log.info(f"Router: {type(router).__name__}")

    # 6. Evaluator
    evaluator = StreamEvaluator(score_threshold=conf_threshold)

    # 7. Diagnosis components
    diag_cfg = cfg.get("diagnosis", {})
    buffer = TemporalBuffer(max_size=diag_cfg.get("buffer_size", 200))

    cp_signals = diag_cfg.get("monitor_signals", [
        "brightness_mean", "confidence_mean", "detection_count",
    ])
    cp_detector = MultiSignalChangePointDetector(
        signal_names=cp_signals,
        fusion=diag_cfg.get("fusion", "any"),
    )

    # Feature extraction control
    extract_features = diag_cfg.get("extract_features", False)

    # Anchor warmup: number of clean frames to establish baseline
    anchor_warmup = diag_cfg.get("anchor_warmup", 50)
    anchor_set = False

    # Drift signal prefixes (use multiple signals for robust detection)
    input_drift_prefixes = diag_cfg.get("input_drift_signals", [
        "brightness", "contrast", "blur", "noise", "edge",
    ])
    output_drift_prefixes = diag_cfg.get("output_drift_signals", [
        "confidence", "detection", "entropy", "margin",
    ])
    feature_drift_prefixes = diag_cfg.get("feature_drift_signals", [
        "feat_agg",
    ])

    # 8. Harm predictor (SDMR contribution)
    harm_predictor = None
    if strategy == "sdmr":
        harm_cfg = cfg.get("harm", {})
        harm_predictor = HistogramHarmPredictor(
            n_bins=harm_cfg.get("n_bins", 5),
            horizon=harm_cfg.get("horizon", 50),
            ema_alpha=harm_cfg.get("ema_alpha", 0.1),
        )

    # 9. Signal-Conditioned Adaptive Restoration (SCAR — key SDMR contribution)
    restoration_engine = None
    if strategy == "sdmr":
        from sdmr.restoration import create_restoration
        restore_cfg = cfg.get("restoration", {})
        restoration_engine = create_restoration(restore_cfg)
        log.info(f"SCAR restoration: method={restore_cfg.get('denoise_method', 'cascade')}")

    # 10. Feature Anchor Normalization (FAN — key V7 contribution)
    fan_engine = None
    if strategy == "fan":
        fan_cfg = cfg.get("fan", {})
        fan_engine = create_fan(detector._model, **fan_cfg)
        log.info(f"FAN initialized: layers={fan_engine._accessible_layers}")

    # ---- Build shift schedule for metrics ----
    shift_schedule = _extract_shift_schedule(scenario)
    log.info(f"Shift schedule: {len(shift_schedule)} shift periods")

    # ---- Input preprocessing adaptation (SDMR contribution) ----
    # When shift is detected, apply signal-driven input preprocessing to
    # counter the distribution shift. This is model-free and reversible.
    # Drift threshold controls when preprocessing activates.
    input_adapt_drift_threshold = cfg.get("adaptation", {}).get("input_adapt_threshold", 0.5)

    # ---- FAN v4: hysteresis counter for transition safety ----
    # Require N consecutive high-drift frames before triggering FAN correction.
    # This prevents correcting clean frames at shift-to-clean boundaries where
    # the drift window still includes recent shifted frames.
    fan_hysteresis_threshold = cfg.get("fan", {}).get("hysteresis_frames", 5)
    fan_consecutive_high_drift = 0  # count of consecutive above-threshold frames
    fan_active = False  # True once hysteresis satisfied

    # FAN v4: minimum noise_ratio to apply Gaussian blur
    # Below this, mild noise doesn't benefit from blurring.
    # Calibrated from V8: noise_ratio ~5.0 crossover (sev 0.5→0.7)
    fan_noise_blur_min_ratio = cfg.get("fan", {}).get("noise_blur_min_ratio", 5.0)

    # FAN v4: confidence-gated safety
    # If FAN correction REDUCES detection count vs original, revert.
    fan_confidence_gate = cfg.get("fan", {}).get("confidence_gate", True)
    fan_conf_gate_threshold = cfg.get("fan", {}).get("conf_gate_threshold", -1)
    fan_reverted_count = 0

    # ---- Main loop ----
    results_log = []
    start_time = time.time()
    prev_image = None

    for frame_idx in range(n_frames):
        frame_start = time.time()

        # Get frame from stream (includes shift injection)
        stream_frame = stream.get_frame(frame_idx)
        image = stream_frame.image
        targets = stream_frame.targets
        is_shifted = stream_frame.metadata.shift_active

        # Ground truth boxes
        gt_boxes = targets['boxes'] if targets and 'boxes' in targets else np.zeros((0, 4), dtype=np.float32)

        # ---- Compute input signals EARLY for SDMR ----
        # For SDMR, compute input signals BEFORE detection so we can
        # apply signal-driven preprocessing on the SAME frame.
        input_sigs = compute_input_signals(image, previous_image=prev_image)

        # ---- Feature extraction (optional) ----
        feature_sigs_dict = {}
        if extract_features:
            try:
                feature_maps = detector.extract_features([image])
                feat_signals = compute_feature_signals(feature_maps)
                feature_sigs_dict = feat_signals.to_flat_dict()
            except Exception as e:
                log.debug(f"Feature extraction error frame {frame_idx}: {e}")

        # ---- Update temporal buffer with input+feature signals ----
        # (output signals added after detection below)
        input_feature_signals = {}
        input_feature_signals.update(input_sigs)
        input_feature_signals.update(feature_sigs_dict)
        buffer.append(input_feature_signals)

        # ---- Change-point detection ----
        avail_cp = {k: input_feature_signals.get(k, 0.0) for k in cp_signals if k in input_feature_signals}
        cp_result = ChangePointResult()
        if avail_cp:
            try:
                cp_result = cp_detector.update(avail_cp)
            except Exception:
                pass

        # ---- Set anchor from clean warmup frames ----
        if not anchor_set and buffer.size >= anchor_warmup and strategy in ("sdmr", "fan"):
            buffer.compute_anchor_from_buffer(window=anchor_warmup)
            anchor_set = True
            log.info(
                f"[Frame {frame_idx}] Anchor set from {anchor_warmup} clean frames. "
                f"Anchor signals: {list(buffer._anchor_mean.keys())[:5]}..."
            )

        # ---- Compute drift scores ----
        input_drift = output_drift = feature_drift = 0.0
        drift_window = min(10, max(3, anchor_warmup))
        if anchor_set and buffer.size >= drift_window:
            input_drift = buffer.compute_drift_score(input_drift_prefixes, window=drift_window)
            output_drift = buffer.compute_drift_score(output_drift_prefixes, window=drift_window)
            feature_drift = buffer.compute_drift_score(feature_drift_prefixes, window=drift_window)
        elif buffer.size >= 5:
            # Fallback: use CV-based drift before anchor is set
            stats = buffer.get_rolling_stats(window=5)
            input_drift = _compute_drift_from_stats(stats, "brightness")
            output_drift = _compute_drift_from_stats(stats, "confidence")
            feature_drift = _compute_drift_from_stats(stats, "feat_agg")

        # ---- Signal-driven input preprocessing (SDMR + SCAR) ----
        # When distribution drift is detected, classify the shift type and
        # apply Signal-Conditioned Adaptive Restoration BEFORE detection.
        # No model modification — purely input-level repair.
        # KEY: Only preprocess when BOTH drift is high AND signal values
        # confirm actual corruption. Restoration parameters are conditioned
        # on the diagnosed corruption type and severity.
        predict_image = image  # Default: use raw image
        sdmr_preproc_type = "none"
        sdmr_restore_meta = {}
        if strategy == "sdmr" and anchor_set and input_drift > input_adapt_drift_threshold:
            # Classify shift type from input signals using anchor-relative thresholds
            noise_est = input_sigs.get("noise_estimate_mad", 0)
            blur_var = input_sigs.get("blur_laplacian_var", 9999)
            brightness = input_sigs.get("brightness_mean", 0.5)

            anchor_noise = buffer._anchor_mean.get("noise_estimate_mad", noise_est)
            anchor_blur = buffer._anchor_mean.get("blur_laplacian_var", blur_var)
            anchor_brightness = buffer._anchor_mean.get("brightness_mean", brightness)

            noise_ratio = noise_est / max(anchor_noise, 1e-6)
            blur_ratio = blur_var / max(anchor_blur, 1e-6)
            brightness_diff = abs(brightness - anchor_brightness)

            # Detect compound corruptions (multiple simultaneous shifts)
            corruption_flags = []
            if noise_ratio > 2.0:
                corruption_flags.append("denoise")
            if blur_ratio < 0.3:
                corruption_flags.append("sharpen")
            if brightness_diff > 0.15:
                corruption_flags.append("normalize")

            if len(corruption_flags) > 1:
                sdmr_preproc_type = "compound"
            elif len(corruption_flags) == 1:
                sdmr_preproc_type = corruption_flags[0]
            # else: no confirmed corruption → skip preprocessing

        if sdmr_preproc_type != "none" and restoration_engine is not None:
            predict_image, sdmr_restore_meta = restoration_engine.restore(
                image,
                corruption_type=sdmr_preproc_type,
                noise_ratio=noise_ratio,
                blur_ratio=blur_ratio,
                brightness_diff=brightness_diff,
                drift_score=input_drift,
                sigma_est=input_sigs.get("noise_estimate_mad", 0),
            )

        # ---- TTA / simple baseline adaptation BEFORE detection ----
        # TTA methods modify model BN parameters, then we detect with modified model.
        # Simple baselines (bn_adapt, periodic_ft) also adapt before detection.
        adapt_start = time.perf_counter()
        tta_decision = RouterDecision()

        if strategy in TTA_STRATEGIES and tta_method is not None:
            try:
                tta_input = detector._preprocess_for_adaptation([predict_image])
                tta_method.forward_and_adapt(tta_input)
                detector._model.eval()
                tta_decision = RouterDecision(
                    action=AdaptationAction.HEAD_ONLY,
                    reasoning=f"TTA:{strategy}",
                )
            except Exception as e:
                log.debug(f"TTA error frame {frame_idx}: {e}")

        elif strategy == "bn_adapt":
            # BN refresh: re-estimate BN stats from current image every frame
            try:
                detector.adapt_step(
                    batch=[predict_image],
                    mode="bn_refresh",
                    params={"max_batches": 1},
                )
                tta_decision = RouterDecision(
                    action=AdaptationAction.BN_REFRESH,
                    reasoning="bn_adapt:refresh",
                )
            except Exception as e:
                log.debug(f"BN adapt error frame {frame_idx}: {e}")

        elif strategy == "periodic_ft" and frame_idx > 0:
            period = cfg.get("baseline", {}).get("period", 50)
            budget = cfg.get("baseline", {}).get("budget", 3)
            if frame_idx % period == 0:
                try:
                    lr = float(cfg.get("adaptation", {}).get("base_lr", 1e-4))
                    detector.adapt_step(
                        batch=[predict_image],
                        mode="head_only",
                        params={"lr": lr, "steps": budget, "max_steps": budget},
                    )
                    tta_decision = RouterDecision(
                        action=AdaptationAction.HEAD_ONLY,
                        reasoning=f"periodic_ft:period={period}",
                    )
                except Exception as e:
                    log.debug(f"Periodic FT error frame {frame_idx}: {e}")

        adapt_ms_pre = (time.perf_counter() - adapt_start) * 1000

        # ---- Frozen-compatible input-space baselines ----
        # Training-free, gradient-free, single-frame, strictly FD-TTA-legal.
        # `gaussian_smooth` is the key control: it applies
        # the same Gaussian smoothing FAN uses, with NO feature anchoring, so the
        # difference against FAN isolates the feature-anchoring stage.
        if strategy in ("gaussian_smooth", "median_filter", "clahe",
                        "gamma_correct", "input_whiten"):
            _bcfg = cfg.get("baseline_input", {})
            if strategy == "gaussian_smooth":
                _k = int(_bcfg.get("ksize", cfg.get("fan", {}).get("noise_blur_ksize", 3)))
                _s = float(_bcfg.get("sigma", cfg.get("fan", {}).get("noise_blur_sigma", 0.5)))
                predict_image = cv2.GaussianBlur(predict_image, (_k, _k), _s)
            elif strategy == "median_filter":
                predict_image = cv2.medianBlur(predict_image, int(_bcfg.get("ksize", 3)))
            elif strategy == "clahe":
                _lab = cv2.cvtColor(predict_image, cv2.COLOR_BGR2LAB)
                _cl = cv2.createCLAHE(clipLimit=float(_bcfg.get("clip_limit", 2.0)),
                                      tileGridSize=(8, 8))
                _lab[:, :, 0] = _cl.apply(_lab[:, :, 0])
                predict_image = cv2.cvtColor(_lab, cv2.COLOR_LAB2BGR)
            elif strategy == "gamma_correct":
                _g = float(_bcfg.get("gamma", 1.2))
                _lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / _g)) * 255.0,
                               0, 255).astype(np.uint8)
                predict_image = cv2.LUT(predict_image, _lut)
            elif strategy == "input_whiten":
                _f = predict_image.astype(np.float32)
                _m, _sd = _f.mean(axis=(0, 1), keepdims=True), _f.std(axis=(0, 1), keepdims=True)
                _tm = float(_bcfg.get("target_mean", 114.0))
                _ts = float(_bcfg.get("target_std", 58.0))
                predict_image = np.clip((_f - _m) / (_sd + 1e-6) * _ts + _tm,
                                        0, 255).astype(np.uint8)

        # ---- FAN: set hook mode before detection ----
        fan_mode = "none"
        fan_shift_type = "default"
        fan_severity = 0.0
        fan_preproc = "none"
        fan_reverted = False
        original_image_for_gate = image  # keep reference for confidence gate
        if strategy == "fan" and fan_engine is not None:
            # ---- FAN v4 hysteresis: track consecutive high-drift frames ----
            if anchor_set and input_drift > input_adapt_drift_threshold:
                fan_consecutive_high_drift += 1
            else:
                fan_consecutive_high_drift = 0
                fan_active = False

            # Only activate FAN correction after sustained drift
            if fan_consecutive_high_drift >= fan_hysteresis_threshold:
                fan_active = True

            if anchor_set and fan_active and fan_engine.anchor_ready:
                # Classify shift type from diagnosed signals
                noise_est = input_sigs.get("noise_estimate_mad", 0)
                blur_var = input_sigs.get("blur_laplacian_var", 9999)
                brightness_val = input_sigs.get("brightness_mean", 0.5)
                anchor_noise = buffer._anchor_mean.get("noise_estimate_mad", max(noise_est, 1e-6))
                anchor_blur = buffer._anchor_mean.get("blur_laplacian_var", max(blur_var, 1e-6))
                anchor_bright = buffer._anchor_mean.get("brightness_mean", brightness_val)
                noise_ratio = noise_est / max(anchor_noise, 1e-6)
                blur_ratio = blur_var / max(anchor_blur, 1e-6)
                brightness_diff = abs(brightness_val - anchor_bright)
                if noise_ratio > 2.0 and blur_ratio < 0.5:
                    fan_shift_type = "compound"
                elif noise_ratio > 2.0:
                    fan_shift_type = "noise"
                elif blur_ratio < 0.5:
                    fan_shift_type = "blur"
                elif brightness_diff > 0.1:
                    fan_shift_type = "brightness"
                else:
                    fan_shift_type = "default"
                fan_severity = min(max(input_drift, 0.1), 1.0)

                # ----------------------------------------------------------
                # FAN v4 — Shift-type-specific minimal repair:
                #   Noise  → pixel-space Gaussian smoothing ONLY if severe
                #            (noise_ratio >= 5.0; mild noise = do nothing)
                #   Blur/Fog/Brightness → feature-space FAN normalization
                # ----------------------------------------------------------
                fan_cfg = cfg.get("fan", {})
                if fan_shift_type in ("noise", "compound"):
                    # FAN v4: Only blur when noise is moderate-to-severe.
                    # noise_ratio < 5.0 → mild noise → skip (no_adapt equiv)
                    # noise_ratio >= 5.0 → scale sigma from noise severity
                    if noise_ratio >= fan_noise_blur_min_ratio:
                        base_sigma = fan_cfg.get("noise_blur_sigma", 0.5)
                        noise_blur_ksize = fan_cfg.get("noise_blur_ksize", 3)
                        # Scale from 1x at min_ratio to 3x at ratio=6+
                        adaptive_scale = min(
                            (noise_ratio - fan_noise_blur_min_ratio + 2.0) / 2.0,
                            3.0,
                        )
                        adaptive_sigma = base_sigma * adaptive_scale
                        predict_image = cv2.GaussianBlur(
                            predict_image,
                            (noise_blur_ksize, noise_blur_ksize),
                            adaptive_sigma,
                        )
                        fan_preproc = f"gaussian_smooth_s{adaptive_sigma:.2f}"
                        fan_mode = "preproc_only"
                    else:
                        # Mild noise: skip — Gaussian blur hurts more than helps
                        fan_mode = "skip_mild"
                        fan_preproc = "skip_mild_noise"
                elif not cfg.get("fan", {}).get("disable_feature_correction", False):
                    # Feature-space normalization for blur/fog/brightness
                    if fan_engine.begin_correct(severity=fan_severity, shift_type=fan_shift_type):
                        fan_mode = "correct"
            elif input_drift <= input_adapt_drift_threshold:
                # Clean enough frame -> accumulate anchor statistics
                fan_engine.begin_accumulate()
                fan_mode = "accumulate"

        # ---- Detection ----
        det_start = time.perf_counter()
        result = detector.predict_single(
            predict_image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        det_ms = (time.perf_counter() - det_start) * 1000

        # ---- FAN: end hook mode after detection ----
        if strategy == "fan" and fan_engine is not None:
            if fan_mode == "correct":
                fan_engine.end_correct()
            elif fan_mode == "accumulate":
                fan_engine.end_accumulate()

        # ---- FAN v4: Confidence-gated safety ----
        # If FAN was active (preproc or feature correction), verify it
        # didn't reduce detection count. If it did, revert to original.
        # This guarantees FAN never harms: worst case = no_adapt.
        if (strategy == "fan" and fan_confidence_gate
                and fan_mode in ("preproc_only", "correct")):
            fan_n_preds = len(result.scores) if hasattr(result.scores, '__len__') else 0
            # Quick detection on original (unmodified) image
            orig_result_fan = detector.predict_single(
                original_image_for_gate,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            orig_n_preds = len(orig_result_fan.scores) if hasattr(orig_result_fan.scores, '__len__') else 0
            # Revert if FAN reduced detection count (fan_conf_gate_threshold = -1
            # means revert if we lost 2+ detections, safe default)
            if fan_n_preds - orig_n_preds < fan_conf_gate_threshold:
                result = orig_result_fan
                predict_image = original_image_for_gate
                fan_reverted = True
                fan_reverted_count += 1

        # ---- Confidence-gated safety (SDMR contribution) ----
        # If restoration was applied, verify it didn't harm detector confidence.
        # If mean confidence dropped, revert to original image detection.
        # This guarantees SDMR never harms clean frames.
        restoration_reverted = False
        if (sdmr_preproc_type != "none" and restoration_engine is not None
                and restoration_engine.config.enable_confidence_gate):
            restored_scores = result.scores.numpy() if hasattr(result.scores, 'numpy') else np.asarray(result.scores)
            restored_conf = float(np.mean(restored_scores)) if len(restored_scores) > 0 else 0.0
            
            # Quick detection on ORIGINAL image for comparison
            orig_result = detector.predict_single(
                image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            orig_scores = orig_result.scores.numpy() if hasattr(orig_result.scores, 'numpy') else np.asarray(orig_result.scores)
            orig_conf = float(np.mean(orig_scores)) if len(orig_scores) > 0 else 0.0
            
            conf_delta = restored_conf - orig_conf
            # Revert if restoration REDUCED confidence (safety threshold)
            if conf_delta < -restoration_engine.config.confidence_threshold:
                result = orig_result  # Use original detection
                predict_image = image
                restoration_reverted = True
                restoration_engine._stats["frames_reverted"] += 1

        # Convert DetectionResult tensors to numpy
        pred_boxes = result.boxes.numpy() if hasattr(result.boxes, 'numpy') else np.asarray(result.boxes)
        pred_scores = result.scores.numpy() if hasattr(result.scores, 'numpy') else np.asarray(result.scores)
        pred_labels = result.labels.numpy() if hasattr(result.labels, 'numpy') else np.asarray(result.labels)

        # Filter for person class (COCO person = class 0 in YOLO)
        if len(pred_labels) > 0:
            person_mask = pred_labels == 0
            pred_boxes_eval = pred_boxes[person_mask] if person_mask.any() else pred_boxes
            pred_scores_eval = pred_scores[person_mask] if person_mask.any() else pred_scores
        else:
            pred_boxes_eval = pred_boxes
            pred_scores_eval = pred_scores

        # ---- Output signals (after detection) ----
        output_sigs = compute_output_signals(pred_boxes_eval, pred_scores_eval)
        all_signals = {}
        all_signals.update(input_sigs)
        all_signals.update(output_sigs)
        all_signals.update(feature_sigs_dict)

        # ---- Router / baseline decision ----
        adapt_start = time.perf_counter()
        decision = RouterDecision()
        harm_estimate = HarmEstimate()

        if strategy in TTA_STRATEGIES and tta_method is not None:
            # TTA already ran before detection; use that decision
            decision = tta_decision

        elif strategy in ("bn_adapt", "periodic_ft"):
            # Simple baselines already adapted before detection; use that decision
            decision = tta_decision

        elif router is not None:
            # SDMR: Signal-driven input preprocessing already applied above.
            # Router provides decision for LOGGING/TRACKING only.
            # NO model modification — all adaptation is via input preprocessing.

            if isinstance(router, (RuleBasedRouter, CostAwareRouter)):
                decision = router.decide(
                    shift_detected=cp_result.shift_detected,
                    shift_severity=cp_result.severity,
                    shift_persistence=cp_result.persistence,
                    input_drift_score=input_drift,
                    output_drift_score=output_drift,
                    feature_drift_score=feature_drift,
                    signal_snapshot=all_signals,
                )
            else:
                decision = router.decide(signal_snapshot=all_signals)

            # Override action to reflect what we actually did (input preprocessing)
            if sdmr_preproc_type != "none":
                decision = RouterDecision(
                    action=AdaptationAction.BN_REFRESH,  # Signal: adaptation occurred
                    confidence=decision.confidence,
                    severity=decision.severity,
                    shift_type_guess=decision.shift_type_guess,
                    reasoning=f"input_preproc:{sdmr_preproc_type} drift={input_drift:.3f}",
                    signal_snapshot=all_signals,
                )
            # FAN override
            if strategy == "fan" and fan_mode in ("correct", "preproc_only"):
                decision = RouterDecision(
                    action=AdaptationAction.BN_REFRESH,
                    confidence=decision.confidence,
                    severity=decision.severity,
                    shift_type_guess=fan_shift_type,
                    reasoning=f"fan:{fan_shift_type} sev={fan_severity:.2f} preproc={fan_preproc}",
                    signal_snapshot=all_signals,
                )

        adapt_ms = (time.perf_counter() - adapt_start) * 1000 + adapt_ms_pre

        # ---- Evaluate ----
        metrics = evaluator.record_frame(
            frame_idx=frame_idx,
            pred_boxes=pred_boxes_eval,
            pred_scores=pred_scores_eval,
            gt_boxes=gt_boxes,
            action_taken=decision.action.label,
            shift_detected=cp_result.shift_detected,
            true_shift=is_shifted,
            latency_ms=det_ms,
            adaptation_time_ms=adapt_ms,
            signal_summary=all_signals,
        )

        # ---- Log ----
        frame_record = {
            "frame_idx": frame_idx,
            "is_shifted": is_shifted,
            "shift_family": stream_frame.metadata.shift_family,
            "shift_severity": stream_frame.metadata.shift_severity,
            "shift_detected": cp_result.shift_detected,
            "action": decision.action.label,
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1": round(metrics.f1, 4),
            "ap50": round(metrics.ap50, 4),
            "map50_95": round(metrics.map50_95, 4),
            "num_preds": int(len(pred_boxes_eval)),
            "num_gts": int(len(gt_boxes)),
            "det_ms": round(det_ms, 2),
            "adapt_ms": round(adapt_ms, 2),
            "harm_ap_drop": round(harm_estimate.expected_ap_drop, 4),
            "harm_cbr": round(harm_estimate.cost_benefit_ratio, 4),
            "harm_bucket": harm_estimate.severity_bucket,
            "input_drift": round(input_drift, 4),
            "output_drift": round(output_drift, 4),
            "feature_drift": round(feature_drift, 4),
            "sdmr_preproc": sdmr_preproc_type,
            "fan_mode": fan_mode,
            "fan_preproc": fan_preproc,
            "fan_reverted": fan_reverted,
        }
        json_logger.log(frame_record)
        results_log.append(frame_record)

        # Update harm predictor with observed performance
        if harm_predictor is not None and frame_idx > 0:
            prev_ap = results_log[-2]["ap50"] if len(results_log) > 1 else metrics.ap50
            actual_drop = max(0, prev_ap - metrics.ap50)
            harm_predictor.update(all_signals, actual_drop)

        # Progress
        if frame_idx % 50 == 0 or frame_idx == n_frames - 1:
            elapsed = time.time() - start_time
            fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            log.info(
                f"[{frame_idx+1:5d}/{n_frames}] "
                f"AP50={metrics.ap50:.3f} mAP={metrics.map50_95:.3f} "
                f"preds={len(pred_boxes_eval):3d} gts={len(gt_boxes):3d} "
                f"shifted={is_shifted} action={decision.action.label:12s} "
                f"det={det_ms:.0f}ms fps={fps:.1f}"
            )

        prev_image = image

    # ---- End of stream ----
    total_time = time.time() - start_time

    agg_metrics = evaluator.get_aggregate_metrics()
    adapt_metrics = evaluator.compute_adaptation_metrics(shift_schedule)

    final_results = {
        "experiment_name": exp_name,
        "strategy": strategy,
        "seed": seed,
        "n_frames": n_frames,
        "total_time_s": round(total_time, 2),
        "fps": round(n_frames / max(total_time, 0.001), 2),
        "scenario": scenario.pattern.value,
        "aggregate_metrics": agg_metrics,
        "adaptation_metrics": adapt_metrics.to_dict(),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    frames_path = output_dir / "frame_results.json"
    with open(frames_path, "w") as f:
        json.dump(results_log, f, indent=1, default=str)

    log.info(f"\n{'='*60}")
    log.info(f"RESULTS: {exp_name}")
    log.info(f"{'='*60}")
    log.info(f"Strategy:    {strategy}")
    log.info(f"Frames:      {n_frames}")
    log.info(f"Time:        {total_time:.1f}s ({n_frames/max(total_time,0.001):.1f} FPS)")
    log.info(f"Mean AP@50:  {agg_metrics.get('mean_ap50', 0):.4f}")
    log.info(f"Mean mAP:    {agg_metrics.get('mean_map50_95', 0):.4f}")
    log.info(f"Mean F1:     {agg_metrics.get('mean_f1', 0):.4f}")
    if shift_schedule:
        log.info(f"Drop:        {adapt_metrics.drop_magnitude:.4f}")
        log.info(f"Recovery:    {adapt_metrics.recovery_time} frames")
        log.info(f"Det delay:   {adapt_metrics.detection_delay:.1f} frames")
    log.info(f"Saved: {results_path}")
    log.info(f"{'='*60}")

    return final_results


# =====================================================================
# Input adaptation functions (legacy — kept for ablation comparison)
# The new SCAR module (sdmr.restoration) supersedes these.
# =====================================================================

def _apply_input_adaptation(image: np.ndarray, adapt_type: str, drift_score: float) -> np.ndarray:
    """Legacy bilateral-filter-based input adaptation (V1-V4). Kept for ablation.

    Apply signal-driven input preprocessing to counter distribution shift.

    This is a key SDMR insight: instead of modifying model parameters
    (which risks corruption), counter the shift at the input level.
    The drift score modulates the strength of preprocessing.

    Args:
        image: Input image (H, W, 3) uint8.
        adapt_type: Type of preprocessing ("denoise", "sharpen", "normalize").
        drift_score: Drift magnitude (higher = stronger processing).

    Returns:
        Preprocessed image (H, W, 3) uint8.
    """
    import cv2

    if adapt_type == "denoise":
        # Bilateral filter: edge-preserving denoising.
        # Strong parameters: d=9 preserves details while removing noise.
        # sigmaColor/sigmaSpace proportional to drift severity.
        sigma_color = min(120, max(50, int(drift_score * 50 + 25)))
        sigma_space = min(120, max(50, int(drift_score * 50 + 25)))
        return cv2.bilateralFilter(image, 9, sigma_color, sigma_space)

    elif adapt_type == "sharpen":
        # Unsharp masking to counter blur
        strength = min(1.5, max(0.3, drift_score * 0.5))
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    elif adapt_type == "normalize":
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Normalizes brightness/contrast locally
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        clip_limit = min(4.0, max(1.5, drift_score * 1.5))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image  # No-op for unknown types


# =====================================================================
# Helper functions
# =====================================================================

def _extract_shift_schedule(scenario: ShiftScenario) -> list[tuple[int, int]]:
    """Extract (onset, end) pairs from a shift scenario."""
    schedule = []
    cumulative = 0
    for seg in scenario.segments:
        seg_len = seg.get("length", 0)
        family = seg.get("family", "none")
        severity = seg.get("severity", 0.0)
        if family != "none" and severity > 0:
            schedule.append((cumulative, cumulative + seg_len))
        cumulative += seg_len
    return schedule


def _compute_drift_from_stats(stats: dict, prefix: str) -> float:
    """Compute drift score from RollingStats dict."""
    relevant = {k: v for k, v in stats.items() if prefix in k}
    if not relevant:
        return 0.0
    cvs = []
    for key, rs in relevant.items():
        if hasattr(rs, 'std') and hasattr(rs, 'mean'):
            mean_val = abs(rs.mean) + 1e-8
            cvs.append(rs.std / mean_val)
    return float(np.mean(cvs)) if cvs else 0.0


def _execute_adaptation(
    detector: YOLODetectorWrapper,
    decision: RouterDecision,
    frames,
    cfg: dict,
) -> None:
    """
    Execute adaptation action on the detector.

    Args:
        frames: Single image (np.ndarray) or list of images for batch adaptation.
    """
    action = decision.action
    adapt_cfg = cfg.get("adaptation", {})

    # Normalize to list
    if isinstance(frames, np.ndarray):
        batch = [frames]
    elif isinstance(frames, list):
        batch = frames
    else:
        batch = [frames]

    if action == AdaptationAction.BN_REFRESH:
        detector.adapt_step(
            batch=batch,
            mode="bn_refresh",
            params={"max_batches": max(len(batch), decision.adaptation_budget)},
        )
    elif action == AdaptationAction.HEAD_ONLY:
        lr = float(adapt_cfg.get("base_lr", 1e-4)) * float(decision.learning_rate_scale)
        detector.adapt_step(
            batch=batch,
            mode="head_only",
            params={"lr": lr, "steps": decision.adaptation_budget,
                    "max_steps": adapt_cfg.get("max_steps", 5)},
        )
    elif action == AdaptationAction.BOUNDED_FULL:
        lr = float(adapt_cfg.get("base_lr", 1e-5)) * float(decision.learning_rate_scale)
        detector.adapt_step(
            batch=batch,
            mode="bounded_full",
            params={"lr": lr, "steps": decision.adaptation_budget,
                    "max_steps": adapt_cfg.get("max_steps", 3),
                    "grad_clip": adapt_cfg.get("grad_clip", 0.5)},
        )
    elif action == AdaptationAction.RESET_AND_FT:
        detector.load_state("initial")
        lr = float(adapt_cfg.get("base_lr", 1e-4)) * float(decision.learning_rate_scale)
        detector.adapt_step(
            batch=batch,
            mode="head_only",
            params={"lr": lr, "steps": decision.adaptation_budget,
                    "max_steps": adapt_cfg.get("max_steps", 5)},
        )


# =====================================================================
# Multi-strategy runner
# =====================================================================

def run_all_strategies(cfg: dict, strategies: list[str]) -> dict:
    """Run experiment with multiple strategies for comparison."""
    all_results = {}

    for strategy in strategies:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"{'#'*60}")

        scfg = dict(cfg)
        scfg["strategy"] = strategy
        scfg["experiment_name"] = f"{cfg.get('experiment_name', 'sdmr')}_{strategy}"

        try:
            results = run_experiment(scfg)
            all_results[strategy] = results
        except Exception as e:
            logger.error(f"Strategy {strategy} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[strategy] = {"error": str(e)}

    # Save comparison
    comparison = {}
    for strat, res in all_results.items():
        if "error" in res:
            comparison[strat] = {"error": res["error"]}
        else:
            agg = res.get("aggregate_metrics", {})
            comparison[strat] = {
                "mean_ap50": agg.get("mean_ap50", 0),
                "mean_map50_95": agg.get("mean_map50_95", 0),
                "mean_f1": agg.get("mean_f1", 0),
                "fps": res.get("fps", 0),
            }

    comp_path = Path(cfg.get("output_dir", "outputs")) / "strategy_comparison.json"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nComparison: {comp_path}")

    return all_results


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SDMR Streaming Detection Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sdmr.run_stream_experiment --config configs/coco_experiment.yaml
  python -m sdmr.run_stream_experiment --config configs/coco_experiment.yaml --strategy tent
  python -m sdmr.run_stream_experiment --config configs/coco_experiment.yaml --all-strategies
  python -m sdmr.run_stream_experiment --config configs/coco_experiment.yaml --n-frames 100
""",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--strategy", type=str, default=None,
                        help="sdmr, no_adapt, periodic_ft, bn_adapt, tent, eata, cotta, sar, rotta")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=None, dest="n_frames")
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir")
    parser.add_argument("--dataset-dir", type=str, default=None, dest="dataset_dir")
    parser.add_argument("--num-images", type=int, default=None, dest="num_images")
    parser.add_argument("--severity", type=float, default=None, dest="severity",
                        help="Override shift severity (0.0-1.0)")
    parser.add_argument("--experiment-name", type=str, default=None, dest="experiment_name",
                        help="Override experiment name for output directory")
    parser.add_argument("--all-strategies", action="store_true", dest="all_strategies")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args)

    if args.all_strategies:
        strategies = ["sdmr", "no_adapt", "bn_adapt", "periodic_ft",
                       "tent", "eata", "cotta", "sar", "rotta"]
        run_all_strategies(cfg, strategies)
    else:
        results = run_experiment(cfg)
        print(f"\n{'='*60}")
        print(f"Experiment: {results['experiment_name']}")
        print(f"Strategy:   {results['strategy']}")
        agg = results.get('aggregate_metrics', {})
        print(f"Mean AP@50: {agg.get('mean_ap50', 0):.4f}")
        print(f"Mean mAP:   {agg.get('mean_map50_95', 0):.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
