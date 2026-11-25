"""Every number the reframed manuscript quotes for the two-stage cuffless system, with intervals.

s63 to s69 established the result piece by piece, each run answering one question and several of
them without intervals. A paper needs one run that produces the headline table, the warm-up table,
the agreement statistics and the modality check under a single protocol, so that no two numbers in
the manuscript come from runs that differ in some detail a reader cannot see.

The protocol is the strict one from s64. A subject contributes M warm-up windows, drawn at random
because PulseDB shuffles windows within a subject. The level is estimated from those M windows
alone, the offset is the difference between that level and the tracker's mean over the same M
windows, and only the remaining windows are scored. No reference pressure is used anywhere.

Stage one is a ridge from averaged RAW_w encoder features to a subject's mean pressure, fitted on
the 194 validation subjects and never refitted. Stage two is the mean of the four pairwise
checkpoints. The calibration-free comparator is the absolute-loss network scored on exactly the
same windows, so every paired difference compares like with like.

Four outputs:
  warm-up table     M = 1, 3, 5, 6, 10, 20, 60, 120 windows, MAE with subject-level 95% intervals
  headline          M = 20, MAE, paired difference against the calibration-free network
  agreement         M = 20, mean error, SD, limits of agreement, BHS cumulative percentages
  PPG only          M = 20, the same system with the ECG channel set to zero throughout, because
                    both protocol-matched published results use the photoplethysmogram alone
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
RUNS = ROOT / "runs" / "s4_pat_control"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
from network_loader import load_q1_module  # noqa: E402
from s29_level_information import pick_alpha, ridge  # noqa: E402

FOUR_TRACKERS = ("WS_PAIRED", "WS_PAIRED_s43", "BB_INCEPTION_PAIRED", "BB_TCN_PAIRED")
# BP_SINGLE=1 reruns the analysis for the primary single tracker; reports get the suffix _single
SINGLE = os.environ.get("BP_SINGLE") == "1"
TRACKERS = ("WS_PAIRED",) if SINGLE else FOUR_TRACKERS
SUFFIX = "_single" if SINGLE else ""
M_SWEEP = (1, 3, 5, 6, 10, 20, 60, 120)
M_HEADLINE = 20
N_DRAWS, N_BOOT = 50, 2000
MIN_TEST_WINDOWS, MIN_FIT_WINDOWS = 100, 20
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
SEED = 42
TARGETS = ((0, "SBP"), (1, "DBP"))


@torch.no_grad()
def encode(tag, X, rows, device, q1, drop_ecg, batch=512):
    ck = torch.load(RUNS / f"{tag}_weights.pt", map_location="cpu", weights_only=False)
    model = q1.RawResNet1D(dropout=0.5)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    mean, std = ck["signal_mean"], ck["signal_std"]
    out = np.empty((len(rows), 512), dtype=np.float32)
    for s in range(0, len(rows), batch):
        sel = rows[s: s + batch]
        x = (np.asarray(X[sel], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        if drop_ecg:
            x[:, 0] = 0.0                       # exactly as s35 --drop-ecg trained these models
        e = model.encoder(torch.from_numpy(x).to(device))
        out[s: s + batch] = torch.cat([e.mean(-1), e.amax(-1)], dim=1).float().cpu().numpy()
    return out


def level_scalar(fv, ft, Sv, val_rows, Yv, fit_subs):
    """Per-window scalar whose mean over any window set is that set's level estimate."""
    G = np.stack([fv[Sv[val_rows] == g].mean(0) for g in fit_subs]).astype(np.float64)
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    out_t = {}
    for j, _ in TARGETS:
        yv = np.array([Yv[val_rows][Sv[val_rows] == g, j].mean() for g in fit_subs])
        a = pick_alpha((G - mu) / sd, yv, ALPHAS, seed=SEED)
        cmu, w, b = ridge((G - mu) / sd, yv, a)
        out_t[j] = ((ft.astype(np.float64) - mu) / sd - cmu) @ w + b
    return out_t


def draws_for(subs, rows_by, M):
    return {int(s): [np.random.default_rng(SEED + 7 * d).choice(rows_by[int(s)], size=M,
                                                                  replace=False)
                     for d in range(N_DRAWS)] for s in subs}


def score(subs, rows_by, draws, Y, track, base, lvl, j, keep_errors=False):
    """Per-subject MAE of the two-stage system and of the comparator on identical windows."""
    two, cf = np.zeros(len(subs)), np.zeros(len(subs))
    signed_two, signed_cf = [], []
    for i, s in enumerate(subs):
        rows = rows_by[int(s)]
        a, c = [], []
        for warm in draws[int(s)]:
            m = np.ones(len(rows), dtype=bool)
            m[np.searchsorted(rows, np.sort(warm))] = False
            ev = rows[m]
            est = track[ev, j] + (lvl[j][warm].mean() - track[warm, j].mean())
            e2, e1 = est - Y[ev, j], base[ev, j] - Y[ev, j]
            a.append(np.abs(e2).mean())
            c.append(np.abs(e1).mean())
            if keep_errors:
                signed_two.append(e2)
                signed_cf.append(e1)
        two[i], cf[i] = float(np.mean(a)), float(np.mean(c))
    if keep_errors:
        return two, cf, np.concatenate(signed_two), np.concatenate(signed_cf)
    return two, cf


def interval(x, boot):
    d = x[boot].mean(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def agreement(e):
    me, sd = float(e.mean()), float(e.std(ddof=1))
    a = np.abs(e)
    return {"mean_error": me, "sd": sd, "loa": [me - 1.96 * sd, me + 1.96 * sd],
            "bhs_pct_within": {"5": float((a <= 5).mean() * 100),
                               "10": float((a <= 10).mean() * 100),
                               "15": float((a <= 15).mean() * 100)},
            "aami_pass": bool(abs(me) <= 5 and sd <= 8), "n_errors": int(e.size)}


def bhs_grade(p):
    """British Hypertension Society grade from cumulative percentages within 5, 10, 15 mmHg."""
    for grade, (g5, g10, g15) in (("A", (60, 85, 95)), ("B", (50, 75, 90)), ("C", (40, 65, 85))):
        if p["5"] >= g5 and p["10"] >= g10 and p["15"] >= g15:
            return grade
    return "D"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()

    Y = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    S = np.load(CONVERTED / "calfree_test" / "S.npy")
    Xt = np.load(CONVERTED / "calfree_test" / "X.npy", mmap_mode="r")
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    Xv = np.load(CONVERTED / "train" / "X.npy", mmap_mode="r")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")

    subs = np.array([s for s in np.unique(S) if (S == s).sum() >= MIN_TEST_WINDOWS])
    rows_by = {int(s): np.flatnonzero(S == s) for s in subs}
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
    boot = np.random.default_rng(7).integers(0, len(subs), size=(N_BOOT, len(subs)))

    load = lambda t: np.asarray(np.load(RUNS / f"{t}_test_pred.npy"), dtype=np.float64)  # noqa: E731
    track = np.mean([load(t) for t in TRACKERS], axis=0)
    base = load("WS_ABS")

    z = np.load(RUNS / "RAW_w_feat_cache.npz")
    lvl = level_scalar(z["val"], z["test"], Sv, val_rows, Yv, fit_subs)
    print(f"{len(subs)} test subjects, {len(fit_subs)} validation subjects fit the level rule, "
          f"{N_DRAWS} warm-up draws each\n", flush=True)

    out = {"warmup": {}, "headline": {}, "agreement": {}, "ppg_only": {}}

    # ---- warm-up table
    print("warm-up  seconds |   SBP MAE  95% CI          |   DBP MAE  95% CI")
    for M in M_SWEEP:
        dr = draws_for(subs, rows_by, M)
        row = {}
        for j, t in TARGETS:
            two, cf = score(subs, rows_by, dr, Y, track, base, lvl, j)
            row[t] = {"mae": float(two.mean()), "ci": interval(two, boot),
                      "calibration_free_mae": float(cf.mean())}
        out["warmup"][str(M)] = {"seconds": M * 10, **row}
        print("  %4d  %6d   | %8.3f  [%.2f, %.2f]  | %8.3f  [%.2f, %.2f]"
              % (M, M * 10, row["SBP"]["mae"], *row["SBP"]["ci"],
                 row["DBP"]["mae"], *row["DBP"]["ci"]), flush=True)

    # ---- headline and agreement at M = 20
    dr = draws_for(subs, rows_by, M_HEADLINE)
    print(f"\nheadline, warm-up {M_HEADLINE} windows")
    for j, t in TARGETS:
        two, cf, e2, e1 = score(subs, rows_by, dr, Y, track, base, lvl, j, keep_errors=True)
        d = two - cf
        out["headline"][t] = {"two_stage": {"mae": float(two.mean()), "ci": interval(two, boot)},
                              "calibration_free": {"mae": float(cf.mean()),
                                                   "ci": interval(cf, boot)},
                              "difference": {"delta": float(d.mean()), "ci": interval(d, boot)}}
        a2, a1 = agreement(e2), agreement(e1)
        a2["bhs_grade"], a1["bhs_grade"] = bhs_grade(a2["bhs_pct_within"]), bhs_grade(
            a1["bhs_pct_within"])
        out["agreement"][t] = {"two_stage": a2, "calibration_free": a1}
        h = out["headline"][t]
        print("  %s two-stage %.3f [%.2f, %.2f]   calibration-free %.3f   difference %+.3f "
              "[%+.2f, %+.2f]" % (t, h["two_stage"]["mae"], *h["two_stage"]["ci"],
                                  h["calibration_free"]["mae"], h["difference"]["delta"],
                                  *h["difference"]["ci"]))
        print("      agreement  ME %+.2f  SD %.2f  LoA [%.1f, %.1f]  within 5/10/15: "
              "%.1f / %.1f / %.1f %%  BHS %s  AAMI %s"
              % (a2["mean_error"], a2["sd"], *a2["loa"], a2["bhs_pct_within"]["5"],
                 a2["bhs_pct_within"]["10"], a2["bhs_pct_within"]["15"], a2["bhs_grade"],
                 "pass" if a2["aami_pass"] else "fail"), flush=True)

    # ---- PPG only: ECG zeroed in both stages, comparator is the PPG-only absolute network
    cache = RUNS / "WS_ABS_ppgonly_feat_cache.npz"
    if cache.is_file():
        zz = np.load(cache)
        fv, ft = zz["val"], zz["test"]
    else:
        print("\nencoding with the PPG-only absolute network ...", flush=True)
        fv = encode("WS_ABS_ppgonly", Xv, val_rows, device, q1, True)
        ft = encode("WS_ABS_ppgonly", Xt, np.arange(len(Y)), device, q1, True)
        np.savez(cache, val=fv, test=ft)
    lvl_p = level_scalar(fv, ft, Sv, val_rows, Yv, fit_subs)
    track_p, base_p = load("WS_PAIRED_ppgonly"), load("WS_ABS_ppgonly")
    print(f"\nPPG only, warm-up {M_HEADLINE} windows (single tracker; no PPG-only ensemble exists)")
    for j, t in TARGETS:
        two, cf = score(subs, rows_by, dr, Y, track_p, base_p, lvl_p, j)
        d = two - cf
        out["ppg_only"][t] = {"two_stage": {"mae": float(two.mean()), "ci": interval(two, boot)},
                              "calibration_free": {"mae": float(cf.mean()),
                                                   "ci": interval(cf, boot)},
                              "difference": {"delta": float(d.mean()), "ci": interval(d, boot)}}
        h = out["ppg_only"][t]
        print("  %s two-stage %.3f [%.2f, %.2f]   calibration-free %.3f   difference %+.3f "
              "[%+.2f, %+.2f]" % (t, h["two_stage"]["mae"], *h["two_stage"]["ci"],
                                  h["calibration_free"]["mae"], h["difference"]["delta"],
                                  *h["difference"]["ci"]), flush=True)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"s70_two_stage_headline{SUFFIX}.json").write_text(json.dumps(
        {"analysis_id": "s70_two_stage_headline",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "protocol": "strict: warm-up windows set level and offset and are excluded from scoring",
         "trackers": list(TRACKERS), "level_features": "RAW_w", "n_test_subjects": int(len(subs)),
         "n_fit_subjects": int(len(fit_subs)), "draws": N_DRAWS, "bootstrap": N_BOOT,
         **out}, indent=2), encoding="utf-8")
    print("\nsaved reports/s70_two_stage_headline.json")


if __name__ == "__main__":
    main()
