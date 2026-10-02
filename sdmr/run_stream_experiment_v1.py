#!/usr/bin/env python3
"""
SDMR Streaming Experiment Runner.

End-to-end pipeline for running streaming detection experiments:
1. Load dataset and wrap in StreamWrapper with shift scenarios
2. Initialize detector (YOLO via Ultralytics)
3. Initialize diagnosis pipeline (signals + change-point + router)
4. Run frame-by-frame streaming evaluation
5. Log results and compute metrics

Usage:
    python -m sdmr.run_stream_experiment --config configs/base.yaml
    python -m sdmr.run_stream_experiment --config configs/base.yaml --strategy sdmr
    python -m sdmr.run_stream_experiment --config configs/base.yaml --strategy no_adapt
"""

import argparse
import json
import os
import sys
import time
import yaml
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

# SDMR imports
from sdmr.utils.seed import set_global_seed
from sdmr.utils.logging_utils import setup_logger, ExperimentMeta, JsonLogger, timer
from sdmr.data.datasets.stream_wrapper import StreamWrapper, ShiftScenario
from sdmr.data.datasets.synthetic_shift_injector import SyntheticShiftInjector
from sdmr.models.detector_wrappers.yolo_wrapper import YOLODetectorWrapper
from sdmr.diagnosis.temporal_buffer import TemporalBuffer
from sdmr.diagnosis.input_signals import compute_input_signals
from sdmr.diagnosis.output_signals import compute_output_signals
from sdmr.diagnosis.feature_signals import compute_feature_signals
from sdmr.diagnosis.change_point import (
    CUSUMDetector,
    HysteresisDetector,
    MultiSignalChangePointDetector,
)
from sdmr.adaptation import (
    RuleBasedRouter,
    CostAwareRouter,
    AdaptationAction,
    RouterDecision,
)
from sdmr.baselines import create_baseline
from sdmr.evaluation.metrics import (
    StreamEvaluator,
    compute_frame_metrics,
)


logger = logging.getLogger("sdmr.experiment")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_shift_scenarios(cfg: dict) -> list[ShiftScenario]:
    """Build shift scenarios from config."""
    scenarios = []
    for sc in cfg.get("shift_scenarios", []):
        method = sc.get("type", "abrupt")
        if method == "abrupt":
            scenarios.append(ShiftScenario.abrupt(
                onset=sc["onset"],
                shift_type=sc["shift_type"],
                severity=sc.get("severity", 0.5),
            ))
        elif method == "gradual":
            scenarios.append(ShiftScenario.gradual(
                onset=sc["onset"],
                duration=sc.get("duration", 50),
                shift_type=sc["shift_type"],
                severity=sc.get("severity", 0.5),
            ))
        elif method == "recurring":
            scenarios.append(ShiftScenario.recurring(
                onset=sc["onset"],
                period=sc.get("period", 100),
                duty_cycle=sc.get("duty_cycle", 0.5),
                shift_type=sc["shift_type"],
                severity=sc.get("severity", 0.5),
            ))
    return scenarios


def build_detector(cfg: dict) -> YOLODetectorWrapper:
    """Initialize the YOLO detector."""
    model_cfg = cfg.get("model", {})
    detector = YOLODetectorWrapper(
        model_name=model_cfg.get("name", "yolov8s.pt"),
        confidence_threshold=model_cfg.get("conf_threshold", 0.25),
        iou_threshold=model_cfg.get("iou_threshold", 0.45),
        device=model_cfg.get("device", "cuda:0"),
        half=model_cfg.get("half", False),
    )
    return detector


def build_router(cfg: dict):
    """Build the adaptation router from config."""
    strategy = cfg.get("strategy", "sdmr")

    if strategy == "sdmr":
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
        if router_cfg.get("cost_aware", False):
            return CostAwareRouter(
                base,
                budget_per_window=router_cfg.get("budget_per_window", 10.0),
                cost_weight=router_cfg.get("cost_weight", 0.3),
            )
        return base
    else:
        # Build a baseline strategy
        baseline_cfg = cfg.get("baseline", {})
        return create_baseline(strategy, **baseline_cfg)


def run_experiment(cfg: dict) -> dict:
    """
    Run a single streaming experiment.

    Returns:
        dict with experiment results and metrics.
    """
    # Setup
    seed = cfg.get("seed", 42)
    set_global_seed(seed)

    exp_name = cfg.get("experiment_name", "sdmr_stream")
    output_dir = Path(cfg.get("output_dir", "outputs")) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger("sdmr", output_dir / "experiment.log")
    json_logger = JsonLogger(output_dir / "frames.jsonl")

    log.info(f"Starting experiment: {exp_name}")
    log.info(f"Strategy: {cfg.get('strategy', 'sdmr')}")
    log.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    # Build components
    detector = build_detector(cfg)
    router = build_router(cfg)
    evaluator = StreamEvaluator(
        score_threshold=cfg.get("model", {}).get("conf_threshold", 0.25)
    )

    # Diagnosis components
    diag_cfg = cfg.get("diagnosis", {})
    buffer = TemporalBuffer(
        max_size=diag_cfg.get("buffer_size", 100),
        signal_names=None,  # auto-detect from first frame
    )

    # Change-point detector
    cp_cfg = diag_cfg.get("change_point", {})
    cp_signals = cp_cfg.get("monitor_signals", [
        "brightness_mean", "confidence_mean", "detection_count",
        "feat_agg_mean", "feat_agg_sparsity",
    ])
    cp_detector = MultiSignalChangePointDetector(
        signal_names=cp_signals,
        fusion=cp_cfg.get("fusion", "majority"),
    )

    # Shift injector
    injector = SyntheticShiftInjector(seed=seed)

    # Stream setup
    stream_cfg = cfg.get("stream", {})
    n_frames = stream_cfg.get("n_frames", 1000)
    scenarios = build_shift_scenarios(stream_cfg)

    log.info(f"Running {n_frames} frames with {len(scenarios)} shift scenarios")

    # Build shift schedule for evaluation
    shift_schedule = []
    for sc in scenarios:
        shift_schedule.append((
            sc.onset,
            sc.onset + sc.duration if sc.duration else sc.onset + 50,
            sc.severity,
        ))

    # ---- Main loop ----
    results_log = []
    start_time = time.time()
    prev_frame = None

    for frame_idx in range(n_frames):
        frame_start = time.time()

        # Generate or load frame
        # In real experiments, this reads from dataset
        # For synthetic experiments, we create dummy frames + apply shifts
        frame = _generate_or_load_frame(cfg, frame_idx)
        gt_boxes = _get_ground_truth(cfg, frame_idx)

        # Apply shift if applicable
        shifted_frame = frame.copy()
        is_shifted = False
        for sc in scenarios:
            severity = _get_current_severity(sc, frame_idx)
            if severity > 0:
                shifted_frame = injector.apply(
                    shifted_frame,
                    shift_type=sc.shift_type,
                    severity=severity,
                )
                is_shifted = True

        # Detect
        with timer("detection") as det_timer:
            result = detector.predict(shifted_frame)

        # Extract features (for diagnosis)
        feature_maps = detector.extract_features(shifted_frame)

        # Compute diagnosis signals
        input_signals = compute_input_signals(
            shifted_frame, prev_frame=prev_frame
        )
        output_signals = compute_output_signals(
            result.boxes, result.scores, result.classes
        )
        feature_sigs = compute_feature_signals(feature_maps)

        # Fuse all signals
        all_signals = {}
        all_signals.update(input_signals)
        all_signals.update(output_signals)
        all_signals.update(feature_sigs.to_flat_dict())

        # Update temporal buffer
        buffer.add(all_signals)

        # Change-point detection
        cp_result = cp_detector.update(
            {k: all_signals.get(k, 0.0) for k in cp_signals}
        )

        # Compute drift scores
        if buffer.size >= 10:
            stats = buffer.get_rolling_stats(window=10)
            input_drift = _compute_drift_score(stats, prefix="brightness")
            output_drift = _compute_drift_score(stats, prefix="confidence")
            feature_drift = _compute_drift_score(stats, prefix="feat_agg")
        else:
            input_drift = output_drift = feature_drift = 0.0

        # Router decision
        if hasattr(router, "decide"):
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
                # Baseline strategy
                decision = router.decide(signal_snapshot=all_signals)
        else:
            decision = RouterDecision()

        # Execute adaptation action
        adapt_start = time.time()
        if decision.action != AdaptationAction.NONE:
            _execute_adaptation(
                detector, decision, shifted_frame, cfg
            )
        adapt_time_ms = (time.time() - adapt_start) * 1000

        # Evaluate
        metrics = evaluator.record_frame(
            frame_idx=frame_idx,
            pred_boxes=result.boxes,
            pred_scores=result.scores,
            gt_boxes=gt_boxes,
            action_taken=decision.action.label,
            shift_detected=cp_result.shift_detected,
            true_shift=is_shifted,
            latency_ms=(time.time() - frame_start) * 1000,
            adaptation_time_ms=adapt_time_ms,
            signal_summary=all_signals,
        )

        # Log frame
        frame_record = {
            "frame_idx": frame_idx,
            "is_shifted": is_shifted,
            "shift_detected": cp_result.shift_detected,
            "shift_severity": cp_result.severity,
            "action": decision.action.label,
            "action_budget": decision.adaptation_budget,
            "map50": metrics.ap50,
            "map50_95": metrics.map50_95,
            "latency_ms": (time.time() - frame_start) * 1000,
            "adapt_ms": adapt_time_ms,
        }
        json_logger.log(frame_record)
        results_log.append(frame_record)

        # Progress logging
        if frame_idx % 100 == 0:
            log.info(
                f"Frame {frame_idx}/{n_frames}: "
                f"mAP50={metrics.ap50:.3f}, "
                f"action={decision.action.label}, "
                f"shifted={is_shifted}"
            )

        prev_frame = shifted_frame

    # ---- End of stream ----
    total_time = time.time() - start_time

    # Compute final metrics
    agg_metrics = evaluator.get_aggregate_metrics()
    adapt_metrics = evaluator.compute_adaptation_metrics(
        [(s, e) for s, e, _ in shift_schedule]
    )

    # Save results
    final_results = {
        "experiment_name": exp_name,
        "strategy": cfg.get("strategy", "sdmr"),
        "seed": seed,
        "n_frames": n_frames,
        "total_time_s": total_time,
        "fps": n_frames / total_time,
        "aggregate_metrics": agg_metrics,
        "adaptation_metrics": adapt_metrics.to_dict(),
        "config": cfg,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    log.info(f"Experiment complete. Results saved to {results_path}")
    log.info(f"Mean mAP@50: {agg_metrics.get('mean_ap50', 0):.4f}")
    log.info(f"Mean mAP@50:95: {agg_metrics.get('mean_map50_95', 0):.4f}")
    log.info(f"Total time: {total_time:.1f}s ({n_frames/total_time:.1f} FPS)")

    return final_results


def _generate_or_load_frame(cfg: dict, frame_idx: int) -> np.ndarray:
    """
    Generate or load a frame for the experiment.

    In synthetic mode, creates random frames.
    In dataset mode, loads from the configured dataset.
    """
    data_cfg = cfg.get("data", {})
    mode = data_cfg.get("mode", "synthetic")

    if mode == "synthetic":
        h = data_cfg.get("height", 640)
        w = data_cfg.get("width", 640)
        # Deterministic pseudo-random frame
        rng = np.random.RandomState(frame_idx)
        frame = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return frame
    else:
        # Note: dataset loading lives in the main run_stream_experiment.py entry point;
        # this legacy stub only supports synthetic frames.
        raise NotImplementedError(
            f"Dataset mode '{mode}' not yet implemented. "
            "Use mode='synthetic' for initial experiments."
        )


def _get_ground_truth(cfg: dict, frame_idx: int) -> np.ndarray:
    """Get ground-truth boxes for a frame."""
    data_cfg = cfg.get("data", {})
    mode = data_cfg.get("mode", "synthetic")

    if mode == "synthetic":
        # Synthetic GT: random boxes
        rng = np.random.RandomState(frame_idx + 10000)
        n_boxes = rng.randint(1, 6)
        boxes = []
        for _ in range(n_boxes):
            x1 = rng.randint(0, 500)
            y1 = rng.randint(0, 500)
            w = rng.randint(30, 140)
            h = rng.randint(30, 140)
            boxes.append([x1, y1, x1 + w, y1 + h])
        return np.array(boxes, dtype=np.float32)
    else:
        raise NotImplementedError("Dataset GT loading not yet implemented")


def _get_current_severity(scenario: ShiftScenario, frame_idx: int) -> float:
    """Compute current severity for a scenario at given frame index."""
    if frame_idx < scenario.onset:
        return 0.0

    if scenario.pattern.value == "abrupt":
        duration = scenario.duration if scenario.duration else 9999
        if frame_idx < scenario.onset + duration:
            return scenario.severity
        return 0.0

    elif scenario.pattern.value == "gradual":
        if scenario.duration and scenario.duration > 0:
            progress = (frame_idx - scenario.onset) / scenario.duration
            progress = min(1.0, max(0.0, progress))
            return scenario.severity * progress
        return scenario.severity

    elif scenario.pattern.value == "recurring":
        if scenario.period and scenario.period > 0:
            phase = (frame_idx - scenario.onset) % scenario.period
            duty = scenario.duty_cycle if scenario.duty_cycle else 0.5
            if phase < scenario.period * duty:
                return scenario.severity
        return 0.0

    return 0.0


def _compute_drift_score(stats: dict, prefix: str) -> float:
    """Compute a drift score from rolling stats for a signal group."""
    relevant = {k: v for k, v in stats.items() if prefix in k and "mean" in k}
    if not relevant:
        return 0.0

    # Use coefficient of variation as drift proxy
    cvs = []
    for key, val in relevant.items():
        std_key = key.replace("mean", "std")
        if std_key in stats:
            mean_val = abs(val) + 1e-8
            cvs.append(stats[std_key] / mean_val)
    return float(np.mean(cvs)) if cvs else 0.0


def _execute_adaptation(
    detector: YOLODetectorWrapper,
    decision: RouterDecision,
    frame: np.ndarray,
    cfg: dict,
) -> None:
    """Execute the adaptation action on the detector."""
    action = decision.action

    if action == AdaptationAction.BN_REFRESH:
        detector.adapt_step(
            frames=[frame],
            mode="bn_refresh",
            n_steps=decision.adaptation_budget,
        )
    elif action == AdaptationAction.HEAD_ONLY:
        adapt_cfg = cfg.get("adaptation", {})
        lr = adapt_cfg.get("base_lr", 1e-4) * decision.learning_rate_scale
        detector.adapt_step(
            frames=[frame],
            mode="head_only",
            n_steps=decision.adaptation_budget,
            lr=lr,
        )
    elif action == AdaptationAction.BOUNDED_FULL:
        adapt_cfg = cfg.get("adaptation", {})
        lr = adapt_cfg.get("base_lr", 1e-4) * decision.learning_rate_scale
        detector.adapt_step(
            frames=[frame],
            mode="bounded_full",
            n_steps=decision.adaptation_budget,
            lr=lr,
        )
    elif action == AdaptationAction.RESET_AND_FT:
        # Reset to source checkpoint
        source_ckpt = cfg.get("model", {}).get("source_checkpoint", None)
        if source_ckpt and os.path.exists(source_ckpt):
            detector.load_state(source_ckpt)
        # Then fine-tune
        adapt_cfg = cfg.get("adaptation", {})
        lr = adapt_cfg.get("base_lr", 1e-4) * decision.learning_rate_scale
        detector.adapt_step(
            frames=[frame],
            mode="bounded_full",
            n_steps=decision.adaptation_budget,
            lr=lr,
        )


def main():
    parser = argparse.ArgumentParser(description="SDMR Streaming Experiment")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Override strategy: sdmr, no_adapt, periodic_ft, bn_adapt, etc."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed"
    )
    parser.add_argument(
        "--n_frames", type=int, default=None,
        help="Override number of frames"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply overrides
    if args.strategy:
        cfg["strategy"] = args.strategy
    if args.seed:
        cfg["seed"] = args.seed
    if args.n_frames:
        cfg["stream"]["n_frames"] = args.n_frames
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    # Run
    results = run_experiment(cfg)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Experiment: {results['experiment_name']}")
    print(f"Strategy: {results['strategy']}")
    print(f"Frames: {results['n_frames']}")
    print(f"FPS: {results['fps']:.1f}")
    print(f"Mean mAP@50: {results['aggregate_metrics'].get('mean_ap50', 0):.4f}")
    print(f"Mean mAP@50:95: {results['aggregate_metrics'].get('mean_map50_95', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
