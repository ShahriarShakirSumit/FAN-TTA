#!/usr/bin/env python3
"""Figure 1: MEASURED streaming trace (replaces the schematic drawn by scripts/legacy/generate_final_figures.py).
Per-frame AP50 from the version-pinned re-run (reproduce/run_fig1_trace.sh; Noise 0.9, seed 42), which reproduces Table 3 exactly.
Shown as a 30-frame centred moving average. Layout/style follow the original figure."""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from repro_paths import RUNS, FIGURES, load_json
B = str(RUNS)
def trace(s):
    fr = load_json(f"{B}/repro_fig1_{s}/{s}/frame_results.json"); fr.sort(key=lambda f: f["frame_idx"])
    return np.array([f["ap50"] for f in fr]) * 100, np.array([f["shift_severity"] if f["is_shifted"] else 0 for f in fr])
plt.rcParams.update({"font.size": 10, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
na, sev = trace("no_adapt"); fan, _ = trace("fan"); bn, _ = trace("bn_adapt"); co, _ = trace("cotta")
sm = lambda x: uniform_filter1d(x, size=30, mode="nearest"); t = np.arange(len(na))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.2), height_ratios=[1, 1.6], sharex=True)
a1.fill_between(t, sev, alpha=0.3, color="#e74c3c"); a1.plot(t, sev, color="#e74c3c", lw=1.5)
a1.set_ylabel("Noise\nseverity", fontsize=9); a1.set_ylim(-0.05, 1.05)
for x in (300, 600):
    a1.axvline(x, color="gray", ls="--", alpha=0.7, lw=0.8); a2.axvline(x, color="gray", ls="--", alpha=0.7, lw=0.8)
for x, lab, c in ((150, "Clean", "#27ae60"), (450, "Shifted (Gaussian noise, $\\sigma = 0.9$)", "#e74c3c"), (750, "Clean", "#27ae60")):
    a1.text(x, 0.93, lab, ha="center", fontsize=9, color=c, fontweight="bold")
a2.plot(t, sm(na), color="#7f7f7f", lw=2.2, label="No Adapt")
a2.plot(t, sm(co), color="#1f77b4", lw=1.0, ls=(0, (4, 3)), label="CoTTA (identical to No Adapt)")
a2.plot(t, sm(fan), color="#2ca02c", lw=2.0, label="FAN (ours)")
a2.plot(t, sm(bn), color="#d62728", lw=1.5, ls="--", label="BN Adapt")
a2.set_ylabel("AP50 (%)\n30-frame mean", fontsize=9); a2.set_xlabel("Frame index", fontsize=9); a2.grid(alpha=0.2)
a2.legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=2)
plt.tight_layout(); plt.savefig(FIGURES / "timeline_example.pdf", bbox_inches="tight"); plt.close()
seg = lambda x, a, b: x[a:b].mean()
print(f"shifted means: NA {seg(na,300,600):.2f} FAN {seg(fan,300,600):.2f} BN {seg(bn,300,600):.2f} CoTTA {seg(co,300,600):.2f}; "
      f"clean: NA {np.r_[na[:300],na[600:]].mean():.2f} FAN {np.r_[fan[:300],fan[600:]].mean():.2f} BN {np.r_[bn[:300],bn[600:]].mean():.2f}; "
      f"CoTTA==NoAdapt per frame: {np.array_equal(co, na)}")
