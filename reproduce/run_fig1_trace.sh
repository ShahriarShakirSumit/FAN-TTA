#!/usr/bin/env bash
# Figure 1: measured per-frame trace under Noise 0.9, seed 42, for No-Adapt, FAN, BN-Adapt and CoTTA.
# COCO val2017 at ./data/coco2017. Results go to outputs/runs.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-outputs/runs}
for s in no_adapt fan bn_adapt cotta; do
  python -m sdmr.run_stream_experiment --config configs/repro/fig1_noise09.yaml --strategy $s \
      --experiment-name repro_fig1_$s --output-dir $OUT
done
