"""The obvious reviewer baseline for stage 1: take the level as the mean output of the
calibration-free absolute network over the same warm-up windows, instead of the ridge on its
encoder features fitted on validation subjects. Everything else is the headline system: the same
four-network pairwise tracker, the same 50 draws (seeds as in s68/s70), M = 20, warm-up windows
excluded from scoring. Paired subject-level bootstrap against the headline system.
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

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
RUNS = ROOT / "runs" / "s4_pat_control"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
from s29_level_information import pick_alpha, ridge  # noqa: E402

PAIRED_MODELS = ("WS_PAIRED", "WS_PAIRED_s43", "BB_INCEPTION_PAIRED", "BB_TCN_PAIRED")
M, N_DRAWS, N_BOOT = 20, 50, 2000
MIN_TEST_WINDOWS, MIN_FIT_WINDOWS = 100, 20
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
SEED = 42


def main() -> None:
    Yt = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    St = np.load(CONVERTED / "calfree_test" / "S.npy")
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
    subs = np.array([s for s in np.unique(St) if (St == s).sum() >= MIN_TEST_WINDOWS])
    rows_by = {int(s): np.flatnonzero(St == s) for s in subs}
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
    vrows_by = {int(g): np.flatnonzero(Sv[val_rows] == g) for g in fit_subs}
    vtruth = np.stack([Yv[val_rows][vrows_by[int(g)], :2].mean(0) for g in fit_subs])

    absnet = np.load(RUNS / "s68_abs_predictions.npz")["test_RAW_w"].astype(np.float64)
    track = np.mean([np.load(RUNS / f"{t}_test_pred.npy").astype(np.float64)
                     for t in (("WS_PAIRED",) if os.environ.get("BP_SINGLE") == "1" else PAIRED_MODELS)], axis=0)
    zf = np.load(RUNS / "RAW_w_feat_cache.npz")
    fv, ft = zf["val"].astype(np.float64), zf["test"].astype(np.float64)
    G = np.stack([fv[vrows_by[int(g)]].mean(0) for g in fit_subs])
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    score = {}
    for j in (0, 1):
        a = pick_alpha((G - mu) / sd, vtruth[:, j], ALPHAS, seed=SEED)
        cmu, w, b = ridge((G - mu) / sd, vtruth[:, j], a)
        score[j] = ((ft - mu) / sd - cmu) @ w + b

    tdraws = {int(s): [np.random.default_rng(SEED + 7 * d).choice(
        rows_by[int(s)], size=M, replace=False) for d in range(N_DRAWS)] for s in subs}
    out = {}
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    for j, target in ((0, "SBP"), (1, "DBP")):
        err = {"ridge level (headline)": np.zeros(len(subs)),
               "mean absolute-network output": np.zeros(len(subs))}
        for i, s in enumerate(subs):
            rows = rows_by[int(s)]
            acc = {k: [] for k in err}
            for warm in tdraws[int(s)]:
                keep = np.ones(len(rows), dtype=bool)
                keep[np.searchsorted(rows, np.sort(warm))] = False
                ev = rows[keep]
                for k, lvl in (("ridge level (headline)", score[j][warm].mean()),
                               ("mean absolute-network output", absnet[warm, j].mean())):
                    off = lvl - track[warm, j].mean()
                    acc[k].append(np.abs(track[ev, j] + off - Yt[ev, j]).mean())
            for k in err:
                err[k][i] = np.mean(acc[k])
        d = err["mean absolute-network output"] - err["ridge level (headline)"]
        out[target] = {k: float(v.mean()) for k, v in err.items()}
        out[target]["naive_minus_headline"] = {
            "delta": float(d.mean()),
            "ci": [float(x) for x in np.percentile(d[boot_idx].mean(1), [2.5, 97.5])]}
        print(target, json.dumps(out[target]))
    (REPORTS / ("s76_naive_level_baseline_single.json" if os.environ.get("BP_SINGLE") == "1" else "s76_naive_level_baseline.json")).write_text(json.dumps(
        {"analysis_id": "s76_naive_level_baseline",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "warmup_windows": M, "draws": N_DRAWS, "bootstrap": N_BOOT, "results": out},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
