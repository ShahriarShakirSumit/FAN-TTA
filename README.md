# FAN-TTA

**When TTA Meets Frozen Detectors: A Benchmark and Feature-Anchoring Solution for Streaming Object Detection**

Shahriar Shakir Sumit, Murat Tahtali and Seyedali Mirjalili. ACML 2026 (PMLR).

This repository has the FAN code, the streaming benchmark we built to test test-time adaptation on frozen detectors,
and the results of all our runs.

## The problem

Object detectors in cars, surveillance cameras or medical devices are often frozen once they ship. They have been
certified, the hardware can't run training, or there simply isn't time. So their weights never change. The world they
look at does change, though: the light fades at dusk, fog settles on the lens, sensors get noisy. And the frames come
in one at a time.

Test-time adaptation (TTA) is meant for exactly this kind of drift, so we asked whether it actually helps a frozen
detector watching a video stream. It doesn't. All seven methods we tried either made detection worse or did nothing
at all, and there are clear reasons why:

1. Methods that re-estimate batch-normalisation statistics need a batch, and a single frame gives them mostly noise.
2. Entropy is a poor signal of accuracy for a detector, whose output is hundreds of boxes rather than one label.
3. Methods that learn by gradient descent have no gradients to work with in a frozen model.

## Our method: Feature Anchor Networks (FAN)

With the weights, batches and gradients all off the table, one thing is still there: the feature maps inside the
detector. FAN watches them. It keeps a running record (an "anchor") of what each channel looks like on clean frames,
and when a shift arrives it gently pulls the channels that have drifted back toward their anchors. Strong noise is
handled by light smoothing of the input instead. FAN needs no training, no gradients and no learnable parameters, and
when nothing has drifted it leaves the detector exactly as it was.

## What we found

On a frozen YOLOv8s watching streams of people from COCO val2017:

* Over three seeds, FAN beats the frozen detector in 13 of 15 kinds of synthetic shift (on average +0.47 AP50, and up
  to +2.08 under the heaviest noise).
* It is significantly better than all four TTA methods we ran with several seeds (one-sided Wilcoxon, p of 0.0027 or
  smaller).
* We also compared it with five simple image fixes that work on a frozen detector. FAN beats four of them. Always-on
  Gaussian smoothing ends up level with FAN overall (p = 0.17): smoothing wins on light noise, where FAN deliberately
  holds back, and FAN wins on blur and fog, where smoothing can't help.
* On two real day-to-night shifts (COCO to NightOwls and COCO to ECP Night), FAN adds 2.28 and 1.04 AP50. These are
  single-seed results.

## The benchmark

The benchmark asks one question: when a frozen detector watches a stream whose images change, does a test-time
method help or hurt? It is defined entirely by code and config files, so there is nothing extra to download apart
from the images.

* **Detector:** a frozen YOLOv8s. Nothing in it is ever updated.
* **Streams:** 900 frames of people from COCO val2017. The first 300 are clean, the next 300 are shifted and the last
  300 are clean again, so you can see both the drop and the recovery.
* **Shifts (17):** Gaussian noise at five strengths, blur at three, brightness at two, contrast, fog, a gradual
  brightness change, recurring blur and a mixed sequence, plus two real day-to-night shifts (COCO to NightOwls and
  COCO to EuroCity Persons at night, 200 warm-up frames and 700 night frames each).
* **What is compared:** FAN, seven existing methods (BN Adapt, TENT, EATA, CoTTA, SAR, RoTTA and periodic
  fine-tuning), five simple image fixes, and the frozen detector on its own (No Adapt).
* **Score:** AP50 for people, averaged over the stream. The five main methods are run with three seeds, and we use a
  one-sided Wilcoxon test over the 15 synthetic shifts.

Each shift is one config file in `configs/` (the table under "Run the experiments" lists them), and running a method on
a shift is a single command. The results of all our runs are in `experiment_results/`.

## What's in the repository

```
sdmr/                  the code: FAN (adaptation/feature_anchor.py), the TTA baselines (baselines/), stream and
                       shift generation (data/), detector wrappers (models/), metrics (evaluation/), and the main
                       script run_stream_experiment.py
configs/               one file per experiment: v7_*.yaml and coco_*.yaml for the 15 synthetic shifts,
                       cross_domain_*.yaml for the night shifts, and configs/repro/ for the ablation and Figure 1
experiment_results/artifacts/  summary results of the paper's main experiments, with SHA-256 checksums
experiment_results/runs/       125 re-runs from our pinned environment (a summary and the exact config for each)
reproduce/             scripts that compute the paper's numbers from experiment_results/, and scripts to re-run our checks
scripts/legacy/        our earlier plotting and aggregation scripts, kept for reference
tests/                 264 unit tests
```

## Getting started

We ran everything with Python 3.11.5, PyTorch 2.0.1 (CUDA 11.8) and Ultralytics 8.0.196 on a single NVIDIA V100 with
32 GB of memory.

```bash
git clone https://github.com/ShahriarShakirSumit/FAN-TTA.git
cd FAN-TTA
conda create -n fan-tta python=3.11 -y && conda activate fan-tta
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .
pytest -q tests                                     # 264 tests; 25 of them run once COCO is in data/coco2017
```

Ultralytics downloads the detector weights (`yolov8s.pt`) the first time you run anything. We checked that the file it
downloads today is the same one we used (MD5 `31fd9a2593ff42e8bfd7478c2314ed40`).

**Data.** Put COCO val2017 in `data/coco2017/` (the `val2017/` images and
`annotations/instances_val2017.json`); every config looks there. The night experiments also need NightOwls in
`data/nightowls/` and EuroCity Persons (night) in `data/ecp/`. See `configs/cross_domain_*.yaml` for the exact paths.

## Reproduce the paper's numbers (CPU, a few seconds)

```bash
python reproduce/make_tables.py         # main results, strategy summary and the image-fix baselines
python reproduce/make_ablation_and_figures.py  # ablation table, bar chart and heatmap
python reproduce/make_fig1_trace.py           # Figure 1, the measured trace over a stream
```

These read the summaries in `experiment_results/artifacts/` and the re-runs in `experiment_results/runs/`, and write
the numbers, tables and figures to `outputs/paper/`. When we ran them on a fresh copy, every table and number came out
exactly as in the paper.

## Run the experiments (GPU)

Here is a single run: FAN under Gaussian noise of severity 0.9, seed 42.

```bash
python -m sdmr.run_stream_experiment --config configs/v7_noise_0_9.yaml --strategy fan --seed 42
# writes outputs/v7_noise_0_9/fan/results.json, where mean_ap50 should be 0.6755 (0.6606 for no_adapt)
```

You can swap in any of these for `--strategy`: `no_adapt`, `fan`, `bn_adapt`, `periodic_ft`, `tent`, `eata`,
`cotta`, `sar`, `rotta`, and the five image fixes `gaussian_smooth`, `median_filter`, `clahe`, `gamma_correct` and
`input_whiten`.

| shift | config | shift | config |
|---|---|---|---|
| Noise 0.3, 0.5, 0.7 | `v7_noise_0_3`, `v7_base`, `v7_noise_0_7` | Brightness 0.7, 0.9 | `v7_brightness_0_7`, `v7_brightness_0_9` |
| Noise 0.9, 1.0 | `v7_noise_0_9`, `v7_noise_1_0` | Contrast 0.7, Fog 0.7 | `v7_contrast_0_7`, `v7_fog_0_7` |
| Blur 0.5, 0.7, 0.9 | `v7_blur_0_5`, `v7_blur_0_7`, `v7_blur_0_9` | Gradual, Recurring, Mixed | `coco_gradual_brightness`, `coco_recurring_blur`, `coco_mixed_shifts` |

In the paper, `no_adapt`, `fan`, `bn_adapt`, `tent` and `eata` were run with seeds 42, 123 and 456 on all 15 shifts,
and `cotta`, `sar`, `rotta` and `periodic_ft` with seed 42.

To repeat the extra experiments we added for the final version (results go to `outputs/runs/`):

```bash
bash reproduce/run_input_baselines.sh   # the five image fixes on all 15 shifts (Gaussian smoothing with 3 seeds)
bash reproduce/run_ablation.sh          # FAN settings on Noise 0.5 and Blur 0.9
bash reproduce/run_fig1_trace.sh        # the Figure 1 stream (Noise 0.9, seed 42)
FAN_RUNS=outputs/runs python reproduce/make_tables.py   # rebuild the tables from your own runs
```

### How closely the results reproduce

We re-ran eight results from the paper in our pinned environment. Seven matched to four decimal places: the frozen
detector at Noise 0.5 (69.8973), Blur 0.9 (62.5615) and Noise 0.9 (66.0575); FAN at Noise 0.5 (70.0008) and Noise 0.9
(67.5465); BN Adapt at Noise 0.9 (52.4235); and CoTTA at Noise 0.9 (66.0575, the same as the frozen detector on every
single frame). The eighth, FAN at Blur 0.9, gave 63.53 where the paper reports 63.41. The frozen detector, FAN and
Gaussian smoothing at Noise 0.9 also gave the same numbers on a CPU. We haven't re-run the other results.

## FAN settings

We used the same settings for every shift, with no tuning per condition. They live in the `fan:` block of each config.

| setting | value |
|---|---|
| backbone layers FAN watches | 2, 4, 6 |
| anchor update rate | 0.05 |
| correction strength | 0.4 |
| largest allowed correction | 4.0 times |
| channel drift threshold | 0.3 |
| minimum layer drift before acting | 0.05 |
| noise-to-blur ratio that switches on input smoothing | 5.0 |
| frames to wait before acting | 5 |
| confidence check | on |

## Licence

Our code is under the MIT licence (`LICENSE`). The detector comes from Ultralytics YOLOv8, which has its own AGPL-3.0
licence covering the `ultralytics` package and its weights. The COCO annotations are CC BY 4.0 and the images follow
the Flickr terms. NightOwls and EuroCity Persons are for research use under their own terms.

## Citing our work

If this code or the benchmark helps your research, we'd really appreciate it if you cite our paper:

```bibtex
@inproceedings{sumit2026fan,
  title     = {When {TTA} Meets Frozen Detectors: A Benchmark and Feature-Anchoring Solution for Streaming Object Detection},
  author    = {Sumit, Shahriar Shakir and Tahtali, Murat and Mirjalili, Seyedali},
  booktitle = {Proceedings of the 18th Asian Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026}
}
```

## Questions and feedback

We're happy to hear from you. Open an issue on GitHub, or write to Shahriar Shakir Sumit at
shahriar9121@gmail.com or m.sumit@unsw.edu.au.
Pull requests are welcome too.

## Acknowledgement

Experiments were performed on the computational facility of the National Computational Infrastructure (NCI) through
the UNSW HPC Scheme (DOI: 10.26190/PMN5-7J50).
