# Earlier plotting and aggregation scripts

We used these scripts while writing the paper to aggregate runs and draw tables and figures. They read the summary
JSON files from `./outputs/` (the same files as in `experiment_results/artifacts/`) and write into `./paper/`. We keep them for
reference only. To rebuild the paper's tables and figures, use the scripts in `reproduce/` (see the top-level
README).

Note: `generate_final_figures.py` and `generate_extra_figures.py` draw an early, schematic version of Figure 1 with
synthetic curves. The paper uses a measured per-frame trace instead (`reproduce/make_fig1_trace.py`, with the data from
`reproduce/run_fig1_trace.sh`).

`generate_multiseed_scatter.py` reads `experiment_results/artifacts/multiseed_full_v12.json` and writes
`outputs/paper/figures/multiseed_scatter.pdf`; it redraws the scatter figure with the same points.
