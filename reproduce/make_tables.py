#!/usr/bin/env python3
"""FAN-TTA: every table and number in the paper, from ONE execution per strategy.
Sources: multiseed_v13_5strat.json (No-Adapt, FAN, TENT, EATA, BN-Adapt; seeds 42/123/456) -- the single execution
behind Table 3's seed-42 columns AND the multi-seed tables; comprehensive_v6v9v10_results.json (CoTTA, SAR, RoTTA,
Periodic-FT, single seed 42 -- their only execution); experiment_results/runs (input-space family, pinned env).
Asserts that the two sources agree for every strategy except FAN (so only FAN's seed-42 column changes)."""
import json, os
from pathlib import Path
from statistics import mean
from scipy.stats import wilcoxon
from repro_paths import ART, RUNS, TABLES, RESULTS
MS = json.load(open(ART / "multiseed_v13_5strat.json")); CP = json.load(open(ART / "comprehensive_v6v9v10_results.json"))
BASE = str(RUNS)
C = ["noise_0_3", "noise_0_5", "noise_0_7", "noise_0_9", "noise_1_0", "blur_0_5", "blur_0_7", "blur_0_9", "brightness_0_7",
     "brightness_0_9", "contrast_0_7", "fog_0_7", "gradual_brightness", "recurring_blur", "mixed_shifts"]
DISP = dict(zip(C, ["Noise 0.3", "Noise 0.5", "Noise 0.7", "Noise 0.9", "Noise 1.0", "Blur 0.5", "Blur 0.7", "Blur 0.9", "Bright 0.7",
                    "Bright 0.9", "Contrast 0.7", "Fog 0.7", "Gradual", "Recurring", "Mixed"]))
for c in C:            # only FAN may differ between the two executions
    for s in ("no_adapt", "tent", "eata", "bn_adapt"):
        assert abs(MS[c][s]["42"] - CP[c][s]["ap50"] * 100) < 1e-6, (c, s)
def base(strat, c, s=42):
    f = f"{BASE}/{strat}__{c}__s{s}/{strat}/results.json"
    return json.load(open(f))["aggregate_metrics"]["mean_ap50"] * 100
V = {c: {"no_adapt": MS[c]["no_adapt"]["42"], "fan": MS[c]["fan"]["42"], "bn_adapt": MS[c]["bn_adapt"]["42"],
         "tent": MS[c]["tent"]["42"], "eata": MS[c]["eata"]["42"],
         **{s: CP[c][s]["ap50"] * 100 for s in ("cotta", "sar", "rotta", "periodic_ft")},
         **{b: base(b, c) for b in ("gaussian_smooth", "median_filter", "clahe", "gamma_correct", "input_whiten")}} for c in C}
A = ["no_adapt", "fan", "bn_adapt", "gaussian_smooth", "median_filter", "clahe", "gamma_correct", "input_whiten"]
B = ["no_adapt", "tent", "eata", "cotta", "sar", "rotta", "periodic_ft"]
H = {"no_adapt": "No Adapt", "fan": "FAN (ours)", "bn_adapt": "BN Adapt", "gaussian_smooth": "Gauss.\\ smooth", "median_filter": "Median",
     "clahe": "CLAHE", "gamma_correct": "Gamma", "input_whiten": "Whitening", "tent": "TENT", "eata": "EATA", "cotta": "CoTTA",
     "sar": "SAR", "rotta": "RoTTA", "periodic_ft": "Periodic FT"}
def panel(cols, bold):
    L = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", "Experiment & " + " & ".join(H[s] for s in cols) + " \\\\", "\\midrule"]
    for c in C:
        top = max(round(V[c][s], 2) for s in cols)
        L.append(DISP[c] + " & " + " & ".join((f"\\textbf{{{V[c][s]:.2f}}}" if bold and round(V[c][s], 2) == top else f"{V[c][s]:.2f}") for s in cols) + " \\\\")
    L.append("\\midrule"); L.append("\\textbf{Average} & " + " & ".join(f"{mean(V[c][s] for c in C):.2f}" for s in cols) + " \\\\")
    L.append("$\\Delta$ vs No Adapt & " + " & ".join(f"{mean(V[c][s] - V[c]['no_adapt'] for c in C):+.3f}" for s in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; return "\n".join(L)
tab = ("\\begin{table*}[t]\n\\centering\n\\caption{\\textbf{AP50 (\\%) under 15 synthetic distribution shifts on COCO 2017 person detection "
       "(YOLOv8s, seed 42).} (a)~Frozen-compatible methods: FAN's competitors under the FD-TTA constraints, including five training-free "
       "input-space baselines (Gaussian smoothing with FAN's base kernel applied to every frame, median filtering, CLAHE, gamma correction "
       "and per-image whitening); best per row in bold. (b)~Structurally incompatible TTA methods, reported as diagnostic evidence of the "
       "failure modes in Section~\\ref{sec:intro}. The five multi-seed strategies (No Adapt, FAN, TENT, EATA, BN Adapt) are the seed-42 runs "
       "of the multi-seed campaign (supplementary material); CoTTA, SAR, RoTTA and Periodic FT were run at seed 42 only.}\n"
       "\\label{tab:main_results}\n\\small\n\\textbf{(a) Frozen-compatible methods}\\\\[2pt]\n\\resizebox{\\textwidth}{!}{%\n" + panel(A, True) +
       "\n}\n\\\\[6pt]\n\\textbf{(b) Structurally incompatible TTA methods (diagnostic)}\\\\[2pt]\n\\resizebox{0.86\\textwidth}{!}{%\n" + panel(B, False) + "\n}\n\\end{table*}\n")
(TABLES / "table1_main_results.tex").write_text(tab)
# ---- statistics (one-sided Wilcoxon on 3-seed means, as in the paper) ----
m3 = lambda c, s: mean(MS[c][s].values())
g3 = {c: mean(base("gaussian_smooth", c, s) for s in (42, 123, 456)) for c in C}
def wil(a, b):
    r = wilcoxon(a, b, alternative="greater"); d = [x - y for x, y in zip(a, b)]
    return dict(W=float(r.statistic), p=float(r.pvalue), wins=sum(x > 0 for x in d), mean=mean(d))
fan3 = [m3(c, "fan") for c in C]
W = {s: wil(fan3, [m3(c, s) for c in C]) for s in ("no_adapt", "tent", "eata", "bn_adapt")}
W["gaussian_smooth"] = wil(fan3, [g3[c] for c in C])
s42 = {b: dict(wins=sum(V[c]["fan"] > V[c][b] for c in C), mean=mean(V[c]["fan"] - V[c][b] for c in C),
               vs_noadapt_wins=sum(V[c][b] > V[c]["no_adapt"] for c in C)) for b in ("gaussian_smooth", "median_filter", "clahe", "gamma_correct", "input_whiten")}
best = [c for c in C if round(V[c]["fan"], 2) >= max(round(V[c][s], 2) for s in A + B)]
best_a = [c for c in C if round(V[c]["fan"], 2) >= max(round(V[c][s], 2) for s in A)]
out = dict(fan_s42_mean=mean(V[c]["fan"] for c in C), fan_s42_delta=mean(V[c]["fan"] - V[c]["no_adapt"] for c in C),
           fan_s42_below_noadapt=[(c, round(V[c]["fan"] - V[c]["no_adapt"], 2)) for c in C if V[c]["fan"] < V[c]["no_adapt"]],
           fan_best_of_all_s42=len(best), fan_best_of_frozen_s42=len(best_a),
           fan3_delta=mean(m3(c, "fan") - m3(c, "no_adapt") for c in C), fan3_max_gain=max((m3(c, "fan") - m3(c, "no_adapt"), c) for c in C),
           fan3_losses=[(c, round(m3(c, "fan") - m3(c, "no_adapt"), 2)) for c in C if m3(c, "fan") < m3(c, "no_adapt")],
           wilcoxon=W, input_baselines_s42=s42, fan_minus_gauss_3seed={c: m3(c, "fan") - g3[c] for c in C},
           gauss3_mean=mean(g3.values()), fan3_mean=mean(fan3))
json.dump(out, open(RESULTS / "paper_numbers.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("fan_minus_gauss_3seed", "wilcoxon", "input_baselines_s42")}, indent=0, default=str))
for k, v in W.items(): print(f"Wilcoxon FAN vs {k:<16} W={v['W']:.0f} p={v['p']:.3g} wins={v['wins']}/15 mean={v['mean']:+.2f}")
for k, v in s42.items(): print(f"s42 FAN vs {k:<16} wins={v['wins']}/15 mean={v['mean']:+.2f} | {k} beats No-Adapt {v['vs_noadapt_wins']}/15")

# ---- input-space baseline summary table (main text) ----
names = {"gaussian_smooth": "Gaussian smoothing", "median_filter": "Median filter", "clahe": "CLAHE", "gamma_correct": "Gamma correction", "input_whiten": "Per-image whitening"}
S = ["\\begin{table}[t]", "\\centering", "\\caption{Frozen-compatible input-space baselines. Gaussian smoothing uses all three seeds and is compared with "
     "FAN's three-seed means (one-sided Wilcoxon, $n = 15$); the others use seed 42 and are compared with FAN's seed-42 run. Reference means: "
     f"FAN {out['fan3_mean']:.2f} (three seeds) and {out['fan_s42_mean']:.2f} (seed 42); No Adapt {mean(m3(c,'no_adapt') for c in C):.2f} and "
     f"{mean(V[c]['no_adapt'] for c in C):.2f}.}}", "\\label{tab:input_baselines}", "\\small", "\\begin{tabular}{lccccc}", "\\toprule",
     "Baseline & Seeds & Mean AP50 & FAN $-$ baseline & FAN wins & $>$ No Adapt \\\\", "\\midrule"]
g = W["gaussian_smooth"]
g_na3 = sum(g3[c] > m3(c, "no_adapt") for c in C); out["gauss3_beats_noadapt"] = g_na3
S.append(f"Gaussian smoothing & 3 & {out['gauss3_mean']:.2f} & {g['mean']:+.2f} ($p = {g['p']:.2f}$) & {g['wins']}/15 & {g_na3}/15 \\\\")
for b in ("median_filter", "clahe", "gamma_correct", "input_whiten"):
    S.append(f"{names[b]} & 1 & {mean(V[c][b] for c in C):.2f} & {s42[b]['mean']:+.2f} & {s42[b]['wins']}/15 & {s42[b]['vs_noadapt_wins']}/15 \\\\")
S += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(TABLES / "table_input_baselines.tex").write_text("\n".join(S) + "\n")
fg = out["fan_minus_gauss_3seed"]; fam = lambda ks: (min(fg[k] for k in ks), max(fg[k] for k in ks))
print("FAN-Gauss blur", fam(["blur_0_5", "blur_0_7", "blur_0_9"]), "fog", fg["fog_0_7"], "noise0.3-0.7", fam(["noise_0_3", "noise_0_5", "noise_0_7"]))

# ---- strategy summary (Table 2 of the paper): FAN row from the same single execution, printed ----
d = [V[c]["fan"] - V[c]["no_adapt"] for c in C]
w_, l_, t_ = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d), sum(abs(x) <= 1e-9 for x in d)
print("taxonomy FAN row:", round(mean(d), 3), round(min(d), 3), round(max(d), 3), f"{w_}/{l_}/{t_}")

# ---- supplementary: per-condition AP50 of FAN and the input-space baselines (seed 42) + FAN - smoothing (3 seeds) ----
P = ["\\begin{table}[htbp]", "\\centering", "\\caption{Per-condition AP50 (\\%) of No Adapt, FAN and the five input-space baselines at seed 42, "
     "and the three-seed difference between FAN and Gaussian smoothing.}", "\\label{tab:supp_input_baselines}", "\\scriptsize",
     "\\begin{tabular}{lccccccc|c}", "\\toprule",
     "Condition & No Adapt & FAN & Gauss. & Median & CLAHE & Gamma & Whiten & FAN$-$Gauss.\\ (3 seeds) \\\\", "\\midrule"]
for c in C:
    P.append(f"{DISP[c]} & " + " & ".join(f"{V[c][s]:.2f}" for s in ("no_adapt", "fan", "gaussian_smooth", "median_filter", "clahe", "gamma_correct", "input_whiten"))
             + f" & {out['fan_minus_gauss_3seed'][c]:+.2f} \\\\")
P += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(TABLES / "table_supp_input_baselines.tex").write_text("\n".join(P) + "\n")
