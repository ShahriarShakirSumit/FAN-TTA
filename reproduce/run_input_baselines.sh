#!/usr/bin/env bash
# Input-space baselines in the paper (tables of frozen-compatible methods): Gaussian smoothing at seeds
# 42/123/456, median filter / CLAHE / gamma correction / input whitening at seed 42, on all 15 shift conditions.
# COCO val2017 must be at ./data/coco2017 (val2017/ + annotations/instances_val2017.json). Results go to outputs/runs.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-outputs/runs}
declare -A CFG=(
 [noise_0_3]=v7_noise_0_3 [noise_0_5]=v7_base       [noise_0_7]=v7_noise_0_7
 [noise_0_9]=v7_noise_0_9 [noise_1_0]=v7_noise_1_0  [blur_0_5]=v7_blur_0_5
 [blur_0_7]=v7_blur_0_7   [blur_0_9]=v7_blur_0_9    [brightness_0_7]=v7_brightness_0_7
 [brightness_0_9]=v7_brightness_0_9 [contrast_0_7]=v7_contrast_0_7 [fog_0_7]=v7_fog_0_7
 [gradual_brightness]=coco_gradual_brightness [recurring_blur]=coco_recurring_blur
 [mixed_shifts]=coco_mixed_shifts )
run () {  # condition strategy seed
    local name="$2__$1__s$3"
    if [ -f "$OUT/$name/$2/results.json" ]; then echo "skip $name"; return; fi
    python -m sdmr.run_stream_experiment --config configs/${CFG[$1]}.yaml --strategy $2 --seed $3 \
        --experiment-name $name --output-dir $OUT || echo "FAILED $name"
}
for seed in 42 123 456; do for c in "${!CFG[@]}"; do run $c gaussian_smooth $seed; done; done
for s in median_filter clahe gamma_correct input_whiten; do for c in "${!CFG[@]}"; do run $c $s 42; done; done
