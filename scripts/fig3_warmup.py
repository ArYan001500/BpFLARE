"""Fig. 3: FLARE error versus unlabeled warm-up length on VitalDB CalFree (s96) and on the UCI records (s105), with the
absolute network on the same windows; printed double-column width (6.99 in), Times; vector PDF."""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "fig3" / "Figure_3.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
w96 = json.loads((ROOT / "reports" / "s96_flare_warmup.json").read_text())
u105 = json.loads((ROOT / "reports" / "s105_flare_uci.json").read_text())
mpl.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9,
                     "axes.linewidth": 0.7, "pdf.fonttype": 42, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "axes.labelsize": 9, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
C_FL, C_AB, C_PUB, C_TX, C_BG = "#1f5fa8", "#d9622b", "#8a8f98", "#495057", "#f4f7fb"
PUB = {"SBP": (12.27, 12.37), "DBP": (7.89, 7.94)}

fig = plt.figure(figsize=(6.99, 4.3))
pos = {(0, 0): [0.075, 0.57, 0.40, 0.33], (0, 1): [0.575, 0.57, 0.40, 0.33],
       (1, 0): [0.075, 0.10, 0.40, 0.33], (1, 1): [0.575, 0.10, 0.40, 0.33]}
letters = iter("abcd")
for col, t in enumerate(("SBP", "DBP")):
    # VitalDB
    ax = fig.add_axes(pos[(0, col)])
    ks = ["10s", "30s", "50s", "60s", "100s", "200s", "600s", "1200s"]
    x = np.array([w96[t][k]["M"] * 10 for k in ks], float)
    m = np.array([w96[t][k]["FLARE"][0] for k in ks])
    lo = np.array([w96[t][k]["FLARE"][1][0] for k in ks])
    hi = np.array([w96[t][k]["FLARE"][1][1] for k in ks])
    ab = np.array([w96[t][k]["absolute_same_windows"] for k in ks])
    ax.set_facecolor(C_BG)
    ax.fill_between(x, lo, hi, color=C_FL, alpha=0.18, lw=0)
    ax.plot(x, ab, color=C_AB, lw=1.3, ls=(0, (4, 2)), marker="s", ms=3.2, label="absolute network")
    ax.plot(x, m, color=C_FL, lw=1.8, marker="o", ms=4, mec="white", mew=0.6, label="FLARE, 95% interval")
    for v in PUB[t]:
        ax.axhline(v, color=C_PUB, lw=0.9, ls=(0, (1, 1.5)))
    ax.text(1150, max(PUB[t]) + (0.08 if t == "SBP" else 0.05), f"published {PUB[t][0]:.2f}, {PUB[t][1]:.2f}",
            ha="right", va="bottom", fontsize=8, color=C_TX)
    i200 = ks.index("200s")
    ax.axvline(200, color=C_FL, lw=0.7, ls=(0, (2, 2)), alpha=0.7)
    ax.plot(200, m[i200], "o", ms=9, mfc="none", mec=C_FL, mew=1.2)
    ax.annotate(f"{m[i200]:.2f} at 200 s", (200, m[i200]), xytext=(0, -16), textcoords="offset points",
                ha="center", fontsize=8.5, color=C_FL, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 60, 200, 600, 1200])
    ax.set_xticklabels(["10 s", "30 s", "1 min", "200 s", "10 min", "20 min"])
    ax.minorticks_off()
    ax.set_ylabel(f"{t} MAE (mmHg)")
    ax.set_title(f"VitalDB CalFree, {'systolic' if t == 'SBP' else 'diastolic'}", fontsize=9.5, loc="left")
    ax.text(-0.15, 1.08, f"({next(letters)})", transform=ax.transAxes, fontsize=10, fontweight="bold")
    if col == 0:
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    for s_ in ax.spines.values():
        s_.set_color("#6c757d")
for col, t in enumerate(("SBP", "DBP")):
    # UCI
    ax = fig.add_axes(pos[(1, col)])
    Ms = [5, 10, 20]
    x = np.array(Ms, float) * 10
    m = np.array([u105[t][f"M{k}"]["FLARE"][0] for k in Ms])
    lo = np.array([u105[t][f"M{k}"]["FLARE"][1][0] for k in Ms])
    hi = np.array([u105[t][f"M{k}"]["FLARE"][1][1] for k in Ms])
    am = np.array([u105[t][f"M{k}"]["absolute"][0] for k in Ms])
    alo = np.array([u105[t][f"M{k}"]["absolute"][1][0] for k in Ms])
    ahi = np.array([u105[t][f"M{k}"]["absolute"][1][1] for k in Ms])
    ax.set_facecolor(C_BG)
    ax.fill_between(x, alo, ahi, color=C_AB, alpha=0.15, lw=0)
    ax.fill_between(x, lo, hi, color=C_FL, alpha=0.18, lw=0)
    ax.plot(x, am, color=C_AB, lw=1.3, ls=(0, (4, 2)), marker="s", ms=3.2)
    ax.plot(x, m, color=C_FL, lw=1.8, marker="o", ms=4, mec="white", mew=0.6)
    ax.annotate(f"{m[-1]:.2f}", (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=C_FL, fontweight="bold")
    ax.annotate(f"{am[-1]:.2f}", (x[-1], am[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=C_AB)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["50 s", "100 s", "200 s"])
    ax.minorticks_off()
    ax.set_xlim(40, 260)
    lo_y, hi_y = min(lo.min(), alo.min()), max(hi.max(), ahi.max())
    pad = 0.15 * (hi_y - lo_y)
    ax.set_ylim(lo_y - pad, hi_y + pad)
    ax.set_xlabel("unlabeled warm-up (log scale)")
    ax.set_ylabel(f"{t} MAE (mmHg)")
    ax.set_title(f"UCI records, {'systolic' if t == 'SBP' else 'diastolic'}", fontsize=9.5, loc="left")
    ax.text(-0.15, 1.08, f"({next(letters)})", transform=ax.transAxes, fontsize=10, fontweight="bold")
    for s_ in ax.spines.values():
        s_.set_color("#6c757d")
fig.savefig(OUT, dpi=600)
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("saved", OUT)
