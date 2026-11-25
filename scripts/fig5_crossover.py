"""Fig. 5: the variability relation with ten reference readings, single column (3.36 in), Times; vector PDF.
(a) systolic MAE per band of within-group SBP standard deviation for the mean of ten readings, the absolute
objective and the pairwise objective (reports/s45_crossover_all.json); (b) pairwise minus absolute with 95% intervals,
pooled (s45) and within each dataset (reports/s54_variability_within_source_all.json). Same data as fig_crossover_v3."""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "fig5" / "Figure_5.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.linewidth": 0.7, "pdf.fonttype": 42, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.labelsize": 8.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
C_PW, C_AB, C_CAL, C_TX, C_BG, C_UCI = "#1f5fa8", "#d9622b", "#6c757d", "#343a40", "#f4f7fb", "#8a8f98"
S45 = json.loads((ROOT / "reports/s45_crossover_all.json").read_text())
S54 = json.loads((ROOT / "reports/s54_variability_within_source_all.json").read_text())["results"]
B45 = ["0-3", "3-5", "5-8", "8-12", "12-18", "18-+"]
B54 = ["0 to 3", "3 to 5", "5 to 8", "8 to 12", "12 to 18", "above 18"]
LAB = ["0–3", "3–5", "5–8", "8–12", "12–18", "≥18"]
bands = S45["bands"]
x = np.arange(len(B45))

fig = plt.figure(figsize=(3.36, 3.9))
ax = fig.add_axes([0.17, 0.575, 0.80, 0.37])
ax.set_facecolor(C_BG)
for key, lab, col, mk, mfc in (("cal_only", "mean of 10 readings", C_CAL, "x", C_CAL),
                               ("absolute", "absolute objective", C_AB, "o", "white"),
                               ("paired", "pairwise objective", C_PW, "s", C_PW)):
    ax.plot(x, [bands[b][key] for b in B45], color=col, marker=mk, ms=4, lw=1.4, label=lab, mfc=mfc, mew=1)
ax.set_xticks(x)
ax.set_xticklabels([])
ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_ylabel("SBP MAE (mmHg)")
ax.legend(loc="upper left", fontsize=8, frameon=False, handlelength=1.6, labelspacing=0.25)
ax.grid(axis="y", color="#dee2e6", lw=0.5)
ax.set_axisbelow(True)
ax.text(-0.17, 1.04, "(a)", transform=ax.transAxes, fontsize=9.5, fontweight="bold")
for s_ in ax.spines.values():
    s_.set_color("#6c757d")

bx = fig.add_axes([0.17, 0.12, 0.80, 0.37])
bx.set_facecolor(C_BG)
bx.axhline(0, color=C_TX, lw=0.7)
series = [(f"pooled, {S45['n_groups']:,} groups", [bands[b]["paired_minus_absolute"] for b in B45],
           [bands[b]["difference_ci"] for b in B45], x, C_TX, "D", C_TX)]
for name, col, mfc, off in (("UCI records", C_UCI, "white", -0.22), ("VitalDB subjects", C_PW, C_PW, 0.22)):
    bb = S54[name]["bands"]
    idx = [i for i, b in enumerate(B54) if b in bb]
    series.append((f"{name}, $n$ = {S54[name]['n_groups']:,}", [bb[B54[i]]["delta"] for i in idx],
                   [bb[B54[i]]["ci"] for i in idx], np.array(idx) + off, col, "o", mfc))
for lab, d, ci, xx, col, mk, mfc in series:
    d, ci = np.array(d), np.array(ci)
    bx.errorbar(xx, d, yerr=[d - ci[:, 0], ci[:, 1] - d], fmt=mk, ms=3.8, color=col, mfc=mfc, mec=col,
                lw=0.9, capsize=1.8, label=lab)
bx.set_xticks(x)
bx.set_xticklabels(LAB)
bx.set_xlim(-0.5, len(x) - 0.5)
bx.set_xlabel("within-group SBP standard deviation (mmHg)")
bx.set_ylabel("pairwise − absolute (mmHg)")
bx.legend(loc="lower left", fontsize=8, frameon=False, handlelength=1.2, labelspacing=0.2)
bx.text(-0.42, 0.10, "pairwise worse", fontsize=8, color=C_UCI, va="bottom", ha="left")
bx.text(-0.42, -0.10, "pairwise better", fontsize=8, color=C_UCI, va="top", ha="left")
bx.text(-0.17, 1.04, "(b)", transform=bx.transAxes, fontsize=9.5, fontweight="bold")
for s_ in bx.spines.values():
    s_.set_color("#6c757d")
fig.savefig(OUT, dpi=600)
fig.savefig(OUT.with_suffix(".png"), dpi=220)
print("saved", OUT)
