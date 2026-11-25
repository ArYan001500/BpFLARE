"""Fig. 4: errors against the arterial reference for the absolute network and FLARE (gated) on the 144 CalFree
subjects. Density from a 60,000-pair random sample; mean error, 95% limits of agreement, SD and the FLARE regression
line from all 2,736,000 window-draw pairs (s106). Printed double-column width (6.99 in), Times; vector PDF."""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "fig4" / "Figure_4.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9,
                     "axes.linewidth": 0.7, "pdf.fonttype": 42, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "axes.labelsize": 9, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
C_FL, C_AB, C_TX, C_REG = "#1f5fa8", "#d9622b", "#343a40", "#c2255c"
fig = plt.figure(figsize=(6.99, 4.5))
L, Wd, Hd = [0.075, 0.405, 0.735], 0.25, 0.33
B = [0.585, 0.11]
letters = iter("abcdef")
for row, t in enumerate(("SBP", "DBP")):
    z = np.load(ROOT / "runs" / "s93" / f"s106_agreement_sample_{t}.npz")
    ref = z["ref"]
    rng_y = (-60, 60) if t == "SBP" else (-40, 40)
    rng_x = (50, 220) if t == "SBP" else (25, 120)
    for col, (name, e, me, sd, c, cmap) in enumerate(
            (("Absolute network", z["err_abs"], float(z["me_abs"]), float(z["sd_abs"]), C_AB, "Oranges"),
             ("FLARE", z["err_flare"], float(z["me_flare"]), float(z["sd_flare"]), C_FL, "Blues"))):
        ax = fig.add_axes([L[col], B[row], Wd, Hd])
        ax.hexbin(ref, e, gridsize=45, extent=(*rng_x, *rng_y), cmap=cmap, mincnt=1, bins="log", lw=0)
        ax.axhline(me, color=C_TX, lw=1.1)
        for v in (me - 1.96 * sd, me + 1.96 * sd):
            ax.axhline(v, color=C_TX, lw=0.9, ls=(0, (4, 2)))
        ax.axhline(0, color="#adb5bd", lw=0.6)
        if name == "FLARE":
            k, b0 = z["slope_flare"]
            xs = np.array(rng_x, float)
            ax.plot(xs, k * xs + b0, color=C_REG, lw=1.4)
            ax.text(0.97, 0.94, f"slope {k:.2f}".replace("-", "\u2212"), transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=C_REG, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ced4da", lw=0.5))
        ax.text(0.03, 0.94, f"ME {me:+.2f}, SD {sd:.2f}".replace("-", "−"), transform=ax.transAxes,
                fontsize=8, va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ced4da", lw=0.5))
        ax.set_xlim(*rng_x)
        ax.set_ylim(*rng_y)
        ax.set_xlabel(f"reference {t} (mmHg)")
        if col == 0:
            ax.set_ylabel(f"{t} error (mmHg)")
        ax.set_title("   " + name, fontsize=9.5, loc="left", color=c, fontweight="bold")
        ax.text(-0.17 if col == 0 else -0.14, 1.08, f"({next(letters)})", transform=ax.transAxes, fontsize=10,
                fontweight="bold")
        for s_ in ax.spines.values():
            s_.set_color("#6c757d")
    ax = fig.add_axes([L[2], B[row], Wd, Hd])
    bins = np.linspace(*rng_y, 61)
    for name, e, c in (("absolute network", z["err_abs"], C_AB), ("FLARE", z["err_flare"], C_FL)):
        ax.hist(e, bins=bins, density=True, histtype="stepfilled", color=c, alpha=0.12)
        ax.hist(e, bins=bins, density=True, histtype="step", color=c, lw=1.3, label=name)
    ax.axvline(0, color="#adb5bd", lw=0.6)
    ax.set_xlim(*rng_y)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.38)
    ax.set_xlabel(f"{t} error (mmHg)")
    ax.set_ylabel("density")
    ax.set_title("   Error distribution", fontsize=9.5, loc="left", color=C_TX, fontweight="bold")
    ax.text(-0.17, 1.08, f"({next(letters)})", transform=ax.transAxes, fontsize=10, fontweight="bold")
    if row == 0:
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    for s_ in ax.spines.values():
        s_.set_color("#6c757d")
fig.savefig(OUT, dpi=600)
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("saved", OUT)
