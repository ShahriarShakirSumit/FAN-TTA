"""Locations used by the reproduction scripts (relative to the repository; FAN_RUNS overrides the run folder).

experiment_results/artifacts/   JSON summaries of the paper's main campaigns (multi-seed v13 and the single-seed comprehensive run)
experiment_results/runs/        version-pinned re-runs: input-space baselines, hyper-parameter ablation, Figure 1 trace
outputs/paper/                  everything the scripts produce: numbers (.json), tables (.tex) and figures (.pdf)
"""
import gzip, json, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ART = REPO / "experiment_results" / "artifacts"
RUNS = Path(os.environ.get("FAN_RUNS", REPO / "experiment_results" / "runs"))
RESULTS = Path(os.environ.get("FAN_PAPER_OUT", REPO / "outputs" / "paper")); TABLES = RESULTS / "tables"; FIGURES = RESULTS / "figures"
for d in (TABLES, FIGURES): d.mkdir(parents=True, exist_ok=True)

def load_json(path):
    """Read path, or path + '.gz' when only the compressed copy is shipped."""
    p = Path(path)
    if p.exists(): return json.load(open(p))
    return json.load(gzip.open(str(p) + ".gz", "rt"))
