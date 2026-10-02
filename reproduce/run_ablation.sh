#!/usr/bin/env bash
# FAN hyper-parameter ablation (the ablation table in the paper): Noise 0.5 and Blur 0.9, seed 42, default + six variants,
# plus the No-Adapt reference for each shift. COCO val2017 at ./data/coco2017. Results go to outputs/runs.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-outputs/runs}
python -m sdmr.run_stream_experiment --config configs/repro/abl_default.yaml --strategy no_adapt \
    --experiment-name repro_no_adapt --output-dir $OUT
python -m sdmr.run_stream_experiment --config configs/repro/ablblur_default.yaml --strategy no_adapt \
    --experiment-name repro_ablblur_no_adapt --output-dir $OUT
for pre in abl ablblur; do
  for v in default layers24 layers46 layers2468 strength02 strength06 ema02; do
    python -m sdmr.run_stream_experiment --config configs/repro/${pre}_${v}.yaml --strategy fan --output-dir $OUT
  done
done
