"""Fig. 1: FLARE overview (two stages and the fluctuation gate), drawn at the printed double-column width (6.99 in)
in Times; vector PDF. Real signals and predictions from runs/fig1/fig1_data.npz (fig1_data.py); branch statistics
from reports/s104_flare_branches.json; cohort errors from the main tables."""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "fig1" / "Figure_1.pdf"
d = np.load(ROOT / "runs" / "fig1" / "fig1_data.npz")
br = json.loads((ROOT / "reports" / "s104_flare_branches.json").read_text())["SBP"]["calfree"]

mpl.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2,
                     "ytick.major.size": 2.2})
C_IN, C_L, C_T, C_G, C_R = "#46637f", "#6a4c9c", "#1f8a70", "#b4356b", "#1f5fa8"
C_FL, C_AB, C_GR, C_TX, C_HEART = "#1f5fa8", "#d9622b", "#8a8f98", "#495057", "#d6336c"
W, H = 6.99, 4.90
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def light(c, f):
    c = np.array(mpl.colors.to_rgb(c))
    return tuple(c + (1 - c) * f)


def shadow_box(x, y, w, h, fc, ec, r=0.06, lw=0.8, sh=0.025, z=2):
    ax.add_patch(FancyBboxPatch((x + sh, y - sh), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="#00000022",
                                ec="none", zorder=z - 0.1))
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def gradient_box(x, y, w, h, c, r=0.06, z=3, top=0.95, bottom=0.78, lw=0.8):
    p = shadow_box(x, y, w, h, "none", c, r=r, lw=lw, z=z + 0.2)
    g = np.linspace(0, 1, 64)[:, None]
    c0, c1 = np.array(light(c, top)), np.array(light(c, bottom))
    img = c1[None, None, :] * (1 - g[..., None]) + c0[None, None, :] * g[..., None]
    im = ax.imshow(np.repeat(img, 4, 1), extent=(x, x + w, y, y + h), origin="lower", aspect="auto", zorder=z + 0.15,
                   interpolation="bicubic")
    im.set_clip_path(p)
    return p


def trapezoid(cx, cy, w, h, c, label=None, fs=8.5, z=4):
    k = 0.28 * h
    pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2 + k), (cx + w / 2, cy + h / 2 - k), (cx - w / 2, cy + h / 2)]
    ax.add_patch(Polygon([(px + 0.025, py - 0.025) for px, py in pts], fc="#00000026", ec="none", zorder=z - 0.1))
    p = Polygon(pts, fc="none", ec=c, lw=0.9, zorder=z + 0.2)
    ax.add_patch(p)
    g = np.linspace(0, 1, 64)[None, :]
    c0, c1 = np.array(mpl.colors.to_rgb(c)), np.array(light(c, 0.45))
    img = c0[None, None, :] * (1 - g[..., None]) + c1[None, None, :] * g[..., None]
    im = ax.imshow(np.repeat(img, 8, 0), extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), origin="lower",
                   aspect="auto", zorder=z)
    im.set_clip_path(p)
    if label:
        ax.text(cx, cy, label, ha="center", va="center", color="white", fontsize=fs, zorder=z + 1, linespacing=1.0)


def diamond(cx, cy, w, h, c, label, z=5):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon([(px + 0.03, py - 0.03) for px, py in pts], fc="#00000030", ec="none", zorder=z - 0.1))
    p = Polygon(pts, fc="none", ec=c, lw=1.0, zorder=z + 0.2)
    ax.add_patch(p)
    g = np.linspace(0, 1, 64)[:, None]
    c0, c1 = np.array(light(c, 0.50)), np.array(mpl.colors.to_rgb(c))
    img = c1[None, None, :] * (1 - g[..., None]) + c0[None, None, :] * g[..., None]
    im = ax.imshow(np.repeat(img, 4, 1), extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), origin="lower",
                   aspect="auto", zorder=z)
    im.set_clip_path(p)
    ax.text(cx, cy, label, ha="center", va="center", color="white", fontsize=9, zorder=z + 1)


def arrow(x0, y0, x1, y1, c="#3b3f46", lw=1.0, ms=8, z=6, cs="arc3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, zorder=z,
                                 connectionstyle=cs, shrinkA=0, shrinkB=0))


def panel(x, y, w, h, letter, title, c):
    shadow_box(x, y, w, h, light(c, 0.93), light(c, 0.55), r=0.09, lw=0.8, sh=0.03, z=1)
    ax.add_patch(FancyBboxPatch((x, y + h - 0.28), w, 0.28, boxstyle="round,pad=0,rounding_size=0.09", fc=c,
                                ec="none", zorder=1.5))
    ax.add_patch(plt.Rectangle((x, y + h - 0.28), w, 0.12, fc=c, ec="none", zorder=1.5))
    ax.text(x + 0.09, y + h - 0.14, f"({letter})", color="white", fontsize=9.5, fontweight="bold", va="center",
            zorder=2)
    ax.text(x + 0.36, y + h - 0.14, title, color="white", fontsize=9.5, fontweight="bold", va="center", zorder=2)


def window_card(x, y, w, h, ecg, ppg, z=5, lab=None):
    shadow_box(x, y, w, h, "white", "#9aa3ad", r=0.03, lw=0.6, sh=0.02, z=z)
    t = np.linspace(x + 0.03, x + w - 0.03, len(ecg))
    ax.plot(t, y + h * (0.55 + 0.38 * ecg), color="#2b8a3e", lw=0.45, zorder=z + 0.5)
    ax.plot(t, y + h * (0.08 + 0.38 * ppg), color=C_R, lw=0.55, zorder=z + 0.5)
    if lab:
        ax.text(x - 0.05, y + h / 2, lab, ha="right", va="center", fontsize=9, zorder=z + 1)


def heart(cx, cy, s, c=C_HEART, z=7):
    u = np.linspace(0, 2 * np.pi, 200)
    hx = 16 * np.sin(u) ** 3
    hy = 13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u)
    ax.fill(cx + s * hx / 17 + 0.012, cy + s * hy / 17 - 0.012, color="#00000030", zorder=z - 0.1, lw=0)
    ax.fill(cx + s * hx / 17, cy + s * hy / 17, color=c, zorder=z, lw=0)


ecg, ppg = d["ecg"][:500], d["ppg"][:500]
ecg2, ppg2 = d["ecg2"][:500], d["ppg2"][:500]
G = 0.07
yT, hT = 2.56, 2.28          # top row
yB, hB = 0.06, 2.42          # bottom row
xa, wa = 0.06, 1.22
xb, wb = xa + wa + G, 2.50
xc = xb + wb + G
wc = W - xc - 0.06
xd, wd = 0.06, 4.52
xe = xd + wd + G
we = W - xe - 0.06

# ------------------------------------------------------------------ (a) input
panel(xa, yT, wa, hT, "a", "Input", C_IN)
mx, my, mw, mh = xa + 0.11, 3.66, wa - 0.22, 0.80
shadow_box(mx, my, mw, mh, "#20252b", "#0e1114", r=0.06, lw=0.8, sh=0.03, z=3)
ax.add_patch(FancyBboxPatch((mx + 0.06, my + 0.06), mw - 0.12, mh - 0.12, boxstyle="round,pad=0,rounding_size=0.03",
                            fc="#0f1a24", ec="#2f3d4a", lw=0.5, zorder=3.2))
tt = np.linspace(mx + 0.37, mx + mw - 0.09, 500)
ax.plot(tt, my + 0.48 + 0.20 * ecg, color="#5bd17a", lw=0.6, zorder=4)
ax.plot(tt, my + 0.12 + 0.22 * ppg, color="#6fb6ff", lw=0.7, zorder=4)
ax.text(mx + 0.10, my + 0.57, "ECG", color="#5bd17a", fontsize=8, va="center", zorder=4)
ax.text(mx + 0.10, my + 0.23, "PPG", color="#6fb6ff", fontsize=8, va="center", zorder=4)
ax.add_patch(Polygon([(xa + wa / 2 - 0.11, my - 0.10), (xa + wa / 2 + 0.11, my - 0.10), (xa + wa / 2 + 0.05, my),
                      (xa + wa / 2 - 0.05, my)], fc="#3a4048", ec="none", zorder=3))
cx0, cy0 = xa + wa / 2, 3.28
shadow_box(cx0 - 0.27, cy0 - 0.10, 0.44, 0.20, "#e9ecef", "#868e96", r=0.05, lw=0.6, sh=0.015, z=4)
ax.plot([cx0 + 0.17, cx0 + 0.30, cx0 + 0.30], [cy0, cy0, cy0 - 0.15], color="#868e96", lw=0.8, zorder=4)
ax.plot([cx0 - 0.33, cx0 + 0.33], [cy0 - 0.17, cy0 + 0.17], color="#c92a2a", lw=1.6, zorder=5, solid_capstyle="round")
ax.text(cx0, cy0 - 0.30, "no cuff reading", ha="center", fontsize=8.5, color="#c92a2a", style="italic")
ax.text(cx0, 2.80, "20 random windows", ha="center", fontsize=9, fontweight="bold")
ax.text(cx0, 2.63, "200 s, no pressure label", ha="center", fontsize=8)

# ------------------------------------------------------------------ (b) stage 1
panel(xb, yT, wb, hT, "b", "Stage 1: subject level", C_L)
fy = 4.02
arrow(xa + wa - 0.05, fy, xb + 0.10, fy)
for k in range(3):
    window_card(xb + 0.12 + 0.06 * k, fy - 0.10 - 0.06 * k, 0.42, 0.32, ecg if k % 2 == 0 else ecg2,
                ppg if k % 2 == 0 else ppg2, z=5 + k)
ax.text(xb + 0.36, fy - 0.30, "warm-up\nwindows", ha="center", va="top", fontsize=8.5, linespacing=1.0)
trapezoid(xb + 0.95, fy, 0.44, 0.62, C_L, label="frozen\nencoder", fs=8)
ax.text(xb + 0.95, fy - 0.36, r"$\phi(x)$, 512-d", ha="center", va="top", fontsize=8, color=C_L)
arrow(xb + 0.68, fy, xb + 0.73, fy, ms=7)
gradient_box(xb + 1.27, fy - 0.17, 0.36, 0.34, C_L, r=0.05, top=0.92, bottom=0.72)
ax.text(xb + 1.45, fy, "mean", ha="center", va="center", fontsize=8.5, zorder=6)
arrow(xb + 1.17, fy, xb + 1.27, fy, ms=7)
gradient_box(xb + 1.75, fy - 0.17, 0.40, 0.34, C_L, r=0.05, top=0.30, bottom=0.0)
ax.text(xb + 1.95, fy, "ridge", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold", zorder=6)
arrow(xb + 1.63, fy, xb + 1.75, fy, ms=7)
ax.text(xb + 2.15, fy - 0.36, "fitted on 194\nother subjects", ha="center", va="top", fontsize=8,
        linespacing=1.0, color=C_TX)
gradient_box(xb + 0.16, 2.70, wb - 0.84, 0.52, C_L, r=0.06, top=0.96, bottom=0.82, lw=0.9)
ax.text(xb + 0.16 + (wb - 0.84) / 2, 3.07, r"subject level $\hat{L}_s$", ha="center", va="center", fontsize=9, color=C_L,
        fontweight="bold", zorder=6)
ax.text(xb + 0.16 + (wb - 0.84) / 2, 2.84, rf"{float(d['L_hat']):.1f} mmHg (example subject)", ha="center",
        va="center", fontsize=8, zorder=6)
arrow(xb + 1.95, fy - 0.17, xb + 1.62, 3.22, c=C_L, ms=8)

# ------------------------------------------------------------------ (c) stage 2
panel(xc, yT, wc, hT, "c", "Stage 2: within-subject tracker $f$", C_T)
window_card(xc + 0.30, 4.06, 0.40, 0.30, ecg, ppg, lab=r"$x_i$")
window_card(xc + 0.30, 3.46, 0.40, 0.30, ecg2, ppg2, lab=r"$x_j$")
trapezoid(xc + 0.98, 4.21, 0.26, 0.38, C_T, label=r"$f$", fs=9.5)
trapezoid(xc + 0.98, 3.61, 0.26, 0.38, C_T, label=r"$f$", fs=9.5)
arrow(xc + 0.70, 4.21, xc + 0.85, 4.21, ms=6)
arrow(xc + 0.70, 3.61, xc + 0.85, 3.61, ms=6)
ax.text(xc + 0.98, 3.91, "shared", ha="center", va="center", fontsize=8, color=C_T, style="italic")
gradient_box(xc + 1.32, 3.50, 0.62, 0.82, C_T, r=0.05, top=0.96, bottom=0.80)
ax.text(xc + 1.63, 3.91, "$f_i - f_j$\nvs\n$y_i - y_j$", ha="center", va="center", fontsize=9, zorder=6,
        linespacing=1.15)
arrow(xc + 1.12, 4.21, xc + 1.32, 4.06, ms=6)
arrow(xc + 1.12, 3.61, xc + 1.32, 3.76, ms=6)
ax.text(xc + 2.42, 3.98, "Huber loss\non differences", ha="center", va="center", fontsize=8.5, linespacing=1.05)
arrow(xc + 1.94, 3.91, xc + 2.05, 3.91, ms=6)
ax.text(xc + 2.42, 3.62, "same patient;\nno absolute target", ha="center", va="center", fontsize=8,
        style="italic", color=C_TX, linespacing=1.05)
# inference row: the same f applied to the warm-up windows and to a scored window
ax.plot([xc + 0.12, xc + wc - 0.12], [3.24, 3.24], color=light(C_T, 0.45), lw=0.8, ls=(0, (3, 2)), zorder=3)
ax.text(xc + 0.12, 3.33, "inference", ha="left", va="center", fontsize=8, color=C_T, fontweight="bold", zorder=6)
window_card(xc + 0.08, 2.78, 0.26, 0.22, ecg, ppg, z=5)
ax.text(xc + 0.26, 2.68, "window $x$", ha="center", va="center", fontsize=8, color=C_TX)
trapezoid(xc + 0.46, 2.89, 0.17, 0.26, C_T, label="$f$", fs=8)
arrow(xc + 0.35, 2.89, xc + 0.375, 2.89, ms=4)
gradient_box(xc + 0.60, 2.70, 0.74, 0.38, C_T, r=0.04, top=0.97, bottom=0.82)
ax.text(xc + 0.97, 2.97, r"$\hat{d}(x)=f(x)$", ha="center", va="center", fontsize=8, zorder=6)
ax.text(xc + 0.97, 2.80, r"$-\,\bar{f}_{\mathrm{warm}}$", ha="center", va="center", fontsize=8, zorder=6)
arrow(xc + 0.545, 2.89, xc + 0.60, 2.89, ms=4)
gradient_box(xc + 1.52, 2.70, 0.56, 0.20, C_T, r=0.03, top=0.97, bottom=0.82)
ax.text(xc + 1.80, 2.80, r"mean: $\bar{f}_{\mathrm{warm}}$", ha="center", va="center", fontsize=8, zorder=6)
gradient_box(xc + 1.52, 2.96, 0.56, 0.20, C_T, r=0.03, top=0.97, bottom=0.82)
ax.text(xc + 1.80, 3.06, r"SD: $v_s\rightarrow$(d)", ha="center", va="center", fontsize=8, zorder=6)
arrow(xc + 1.52, 2.80, xc + 1.34, 2.80, ms=5)
trapezoid(xc + 2.34, 2.93, 0.17, 0.26, C_T, label="$f$", fs=8)
arrow(xc + 2.255, 2.97, xc + 2.08, 3.06, ms=5)
arrow(xc + 2.255, 2.89, xc + 2.08, 2.80, ms=5)
for k in range(3):
    shadow_box(xc + 2.53 + 0.035 * k, 2.86 - 0.035 * k, 0.24, 0.17, "white", "#9aa3ad", r=0.02, lw=0.5, sh=0.01,
               z=5 + k)
arrow(xc + 2.52, 2.93, xc + 2.425, 2.93, ms=5)
ax.text(xc + 2.68, 2.66, "warm-up", ha="center", va="center", fontsize=8, color=C_TX)
ax.text(xc + 0.12, 4.47, "training", ha="left", va="center", fontsize=8, color=C_T, fontweight="bold", zorder=6)

# ------------------------------------------------------------------ (d) fluctuation gate
panel(xd, yB, wd, hB, "d", "Fluctuation gate: two branches", C_G)
gradient_box(xd + 0.12, 0.98, 0.98, 0.62, C_G, r=0.06, top=0.96, bottom=0.84)
ax.text(xd + 0.61, 1.46, "fluctuation index", ha="center", va="center", fontsize=8.5, color=C_G, fontweight="bold",
        zorder=6)
ax.text(xd + 0.61, 1.24, r"$v_s=\mathrm{SD}$ of $f$", ha="center", va="center", fontsize=8.5, zorder=6)
ax.text(xd + 0.61, 1.07, "over the warm-up", ha="center", va="center", fontsize=8, zorder=6)
u = np.linspace(0, 4 * np.pi, 120)
ax.plot(xd + 0.16 + 0.42 * u / u[-1], 1.86 + 0.13 * np.sin(u) * (0.6 + 0.4 * np.cos(0.7 * u)), color=C_G, lw=1.0)
ax.plot(xd + 0.64 + 0.42 * u / u[-1], 1.86 + 0.03 * np.sin(1.3 * u), color=C_G, lw=1.0)
ax.text(xd + 0.37, 1.66, "moving", ha="center", fontsize=8, color=C_TX)
ax.text(xd + 0.85, 1.66, "flat", ha="center", fontsize=8, color=C_TX)
ax.text(xd + 0.61, 0.82, "label-free;\n$v_s$ computed in (c)", ha="center", va="top", fontsize=8, style="italic",
        color=C_TX, linespacing=1.05)
dx, dy = xd + 1.58, 1.29
arrow(xd + 1.10, dy, dx - 0.30, dy, ms=7)
diamond(dx, dy, 0.62, 0.58, C_G, r"$v_s \leq \tau$ ?")
mh_, ml_ = br["mostly_high"], br["mostly_low"]
hb_x, hb_w = xd + 2.12, 1.36
gradient_box(hb_x, 1.40, hb_w, 0.66, C_FL, r=0.06, top=0.95, bottom=0.83)
ax.text(hb_x + hb_w / 2, 1.91, "High fluctuation", ha="center", va="center", fontsize=8.5, color=C_FL,
        fontweight="bold", zorder=6)
ax.text(hb_x + hb_w / 2, 1.70, r"$A_s = \hat{L}_s$", ha="center", va="center", fontsize=9, zorder=6)
ax.text(hb_x + hb_w / 2, 1.50, f"SD {mh_['true_SD_median']:.1f}; gain {-mh_['FLARE_minus_absolute'][0]:.2f} mmHg",
        ha="center", va="center", fontsize=8, color=C_TX, zorder=6)
arrow(dx, dy + 0.29, hb_x, 1.73, ms=7, cs="angle,angleA=90,angleB=180,rad=0")
ax.text(dx - 0.16, 1.72, "no", fontsize=8, color=C_TX, style="italic")
lb_y = 0.16
gradient_box(hb_x, lb_y, hb_w, 0.92, C_G, r=0.06, top=0.95, bottom=0.83)
ax.text(hb_x + hb_w / 2, lb_y + 0.80, "Low fluctuation", ha="center", va="center", fontsize=8.5, color=C_G,
        fontweight="bold", zorder=6)
ax.text(hb_x + hb_w / 2, lb_y + 0.59, r"$A_s = \hat{L}_s + b\,(E_s - \bar{E})$", ha="center", va="center", fontsize=9,
        zorder=6)
heart(hb_x + 0.11, lb_y + 0.34, 0.065)
ax.text(hb_x + 0.21, lb_y + 0.34, "ECG and PPG features\n(heart rate, HRV, PAT, …)", ha="left", va="center",
        fontsize=8, zorder=8, linespacing=1.0)
ax.text(hb_x + hb_w / 2, lb_y + 0.10, f"SD {ml_['true_SD_median']:.1f}; gain {-ml_['FLARE_minus_absolute'][0]:.2f} mmHg",
        ha="center", va="center", fontsize=8, color=C_TX, zorder=6)
arrow(dx, dy - 0.29, hb_x, lb_y + 0.59, ms=7, cs="angle,angleA=-90,angleB=180,rad=0")
ax.text(dx - 0.20, 0.80, "yes", fontsize=8, color=C_TX, style="italic")
arrow(xb + 0.16 + (wb - 0.84) / 2, 2.70, hb_x + 0.35, 2.06, c=C_L, lw=1.1, ms=8)
ax.text(hb_x + 0.45, 2.13, r"$\hat{L}_s$ (both branches)", color=C_L, fontsize=8.5, va="center")
ox, oy = xd + 4.06, 1.12
ax.add_patch(Circle((ox + 0.02, oy - 0.02), 0.14, fc="#00000026", ec="none", zorder=5))
ax.add_patch(Circle((ox, oy), 0.14, fc="white", ec="#3b3f46", lw=1.0, zorder=6))
ax.text(ox, oy - 0.005, "+", ha="center", va="center", fontsize=13, zorder=7)
arrow(hb_x + hb_w, 1.73, ox - 0.10, oy + 0.10, ms=7, cs="arc3,rad=-0.25")
arrow(hb_x + hb_w, lb_y + 0.59, ox - 0.10, oy - 0.10, ms=7, cs="arc3,rad=0.25")
arrow(xc + 0.70, 2.70, ox + 0.04, oy + 0.14, c=C_T, lw=1.1, ms=8)
ax.text(ox + 0.18, 2.00, r"$\hat{d}(x)$", fontsize=9, color=C_T, ha="right")
ax.text(ox, 0.62, r"$\hat{y}(x) = \hat{d}(x) + A_s$", ha="center", fontsize=8.5)
ax.text(ox, 0.33, "absolute pressure", ha="center", fontsize=8, style="italic", color=C_TX)

# ------------------------------------------------------------------ (e) result
panel(xe, yB, we, hB, "e", "Result", C_R)
arrow(ox + 0.14, oy, xe + 0.06, oy, ms=8)
sx = fig.add_axes([(xe + 0.42) / W, 1.44 / H, (we - 0.54) / W, 0.64 / H])
lim = 45
sel = np.random.default_rng(0).choice(len(d["dev_ref"]), 160, replace=False)
r_ab = np.corrcoef(d["dev_ref"], d["dev_abs"])[0, 1]
r_fl = np.corrcoef(d["dev_ref"], d["dev_pair"])[0, 1]
sx.scatter(d["dev_ref"][sel], d["dev_abs"][sel], s=3.5, color=C_AB, alpha=0.55, lw=0, label=f"absolute ($r$ = {r_ab:.2f})")
sx.scatter(d["dev_ref"][sel], d["dev_pair"][sel], s=3.5, color=C_FL, alpha=0.75, lw=0, label=f"FLARE ($r$ = {r_fl:.2f})")
for yy, c in ((d["dev_abs"], C_AB), (d["dev_pair"], C_FL)):
    k = np.polyfit(d["dev_ref"], yy, 1)[0]
    sx.plot([-lim, lim], [-lim * k, lim * k], color=c, lw=1.1)
sx.plot([-lim, lim], [-lim, lim], color=C_TX, lw=0.7, ls=(0, (3, 2)))
sx.set_xlim(-lim, lim)
sx.set_ylim(-lim, lim + 46)
sx.set_xticks([-40, 0, 40])
sx.set_yticks([-40, 0, 40])
sx.set_xlabel("reference deviation (mmHg)", fontsize=8, labelpad=1)
sx.set_ylabel("estimate", fontsize=8, labelpad=0)
for s_ in sx.spines.values():
    s_.set_color("#6c757d")
sx.legend(loc="upper left", fontsize=8, frameon=False, handletextpad=0.05, borderaxespad=0.1, markerscale=2.2)
ax.text(xe + we / 2, 1.02, "one subject: changes less flattened", ha="center", fontsize=8, style="italic", color=C_TX)
mxa = fig.add_axes([(xe + 0.98) / W, 0.50 / H, (we - 1.10) / W, 0.36 / H])
items = [("Absolute network", 12.77, 11.85, 13.72, C_AB), ("Published", 12.27, None, None, C_GR),
         ("FLARE", 11.28, 10.45, 12.17, C_FL)]
for i, (lab, m, lo, hi, c) in enumerate(items[::-1]):
    if lo is not None:
        mxa.plot([lo, hi], [i, i], color=c, lw=1.4, solid_capstyle="round")
    mxa.plot(m, i, "o", color=c, ms=4.2 if lab != "FLARE" else 5.0, mec="white", mew=0.5)
    if lab == "Published":
        mxa.plot(12.37, i, "o", color=c, ms=4.2, mec="white", mew=0.5)
    mxa.annotate(lab, xy=(0, i), xycoords=("axes fraction", "data"), xytext=(-4, 0), textcoords="offset points",
                 ha="right", va="center", fontsize=8 if lab != "FLARE" else 8.5,
                 color=c if lab in ("FLARE", "Absolute network") else C_TX,
                 fontweight="bold" if lab == "FLARE" else "normal")
mxa.set_xlim(10.0, 14.0)
mxa.set_ylim(-0.6, 2.6)
mxa.set_yticks([])
mxa.set_xticks([10, 12, 14])
for k in ("left", "right", "top"):
    mxa.spines[k].set_visible(False)
mxa.spines["bottom"].set_color("#6c757d")
mxa.set_xlabel("SBP MAE (mmHg)", fontsize=8, labelpad=1)
mxa.patch.set_alpha(0)
ax.text(xe + we / 2, 0.12, "144 subjects; lower error in 100", ha="center", fontsize=8.5, color=C_FL, fontweight="bold")
fig.savefig(OUT, dpi=600)
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("saved", OUT)
