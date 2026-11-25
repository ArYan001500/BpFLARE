"""Fig. 2: per-subject calibration-free errors of FLARE (gated) and the absolute network on the 144 CalFree subjects,
drawn at the printed double-column width (6.99 in) in Times; vector PDF. Data: runs/s93/s104_per_subject_*.npz."""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "fig2" / "Figure_2.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9,
                     "axes.linewidth": 0.7, "pdf.fonttype": 42, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "axes.labelsize": 9, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
C_FL, C_AB, C_TX, C_BG = "#1f5fa8", "#d9622b", "#495057", "#f4f7fb"
z = {t: np.load(ROOT / "runs" / "s93" / f"s104_per_subject_{t}_calfree.npz") for t in ("SBP", "DBP")}
rng = np.random.default_rng(7)
import json  # noqa: E402
BR = json.loads((ROOT / "reports" / "s104_flare_branches.json").read_text())  # same intervals as the text

fig = plt.figure(figsize=(6.99, 2.55))
axs = [fig.add_axes([0.065, 0.17, 0.255, 0.70]), fig.add_axes([0.385, 0.17, 0.255, 0.70]),
       fig.add_axes([0.735, 0.17, 0.25, 0.70])]
summary = {}
for k, (t, lim) in enumerate((("SBP", 34), ("DBP", 26))):
    ax = axs[k]
    fl, ab = z[t]["flare"], z[t]["absolute"]
    better = fl < ab
    ax.set_facecolor(C_BG)
    ax.fill_between([0, lim], [0, 0], [0, lim], color=C_FL, alpha=0.06, lw=0)
    ax.plot([0, lim], [0, lim], color=C_TX, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.scatter(ab[better], fl[better], s=16, color=C_FL, alpha=0.85, lw=0.4, ec="white", zorder=3,
               label=f"lower with FLARE ({better.sum()})")
    ax.scatter(ab[~better], fl[~better], s=16, facecolor="white", ec=C_AB, lw=0.9, zorder=3,
               label=f"higher with FLARE ({(~better).sum()})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"absolute network, {t} MAE (mmHg)")
    ax.set_ylabel(f"FLARE, {t} MAE (mmHg)")
    ax.legend(loc="upper left", fontsize=8, frameon=False, handletextpad=0.2, borderaxespad=0.2)
    ax.text(-0.20, 1.04, "(a)" if k == 0 else "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold")
    for s_ in ax.spines.values():
        s_.set_color("#6c757d")
    d = fl - ab
    ii = rng.integers(0, len(d), (2000, len(d)))
    m_, (lo_, hi_) = BR[t]["calfree"]["FLARE_minus_absolute"]
    summary[t] = (d, m_, (lo_, hi_))
ax = axs[2]
ax.set_facecolor(C_BG)
for k, t in enumerate(("SBP", "DBP")):
    d, m, (lo, hi) = summary[t]
    x = k + 0.20 * rng.uniform(-1, 1, len(d))
    ax.scatter(x, d, s=8, color=np.where(d < 0, C_FL, C_AB), alpha=0.5, lw=0, zorder=2)
    ax.add_patch(plt.Rectangle((k - 0.30, lo), 0.60, hi - lo, color=C_FL, alpha=0.30, lw=0, zorder=3))
    ax.plot([k - 0.30, k + 0.30], [m, m], color=C_FL, lw=2.2, zorder=4)
    fmt = lambda v: f"{v:+.2f}".replace("-", "\u2212")  # noqa: E731
    ax.text(k, 10.2, f"{fmt(m)}\n[{fmt(lo)}, {fmt(hi)}]", ha="center", va="center", fontsize=8, color=C_FL,
            linespacing=1.05, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_FL, lw=0.6), zorder=5)
ax.axhline(0, color=C_TX, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["SBP", "DBP"])
ax.set_xlim(-0.55, 1.55)
ax.set_ylim(-14.2, 12.0)
ax.set_ylabel("FLARE \u2212 absolute MAE (mmHg)")
ax.text(-0.30, 1.04, "(c)", transform=ax.transAxes, fontsize=10, fontweight="bold")
for s_ in ax.spines.values():
    s_.set_color("#6c757d")
fig.savefig(OUT, dpi=600)
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("saved", OUT, {t: (round(summary[t][1], 3), np.round(summary[t][2], 3)) for t in summary})
