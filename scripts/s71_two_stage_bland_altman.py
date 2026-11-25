"""Reference and error pairs for the agreement plot of the two-stage cuffless system.

The agreement figure has to show the same thing the agreement row of the manuscript reports, so the
points come from exactly the protocol, draws, level rule and tracker ensemble of s70 at a warm-up of
20 windows, and the summary lines are recomputed here from all points rather than from the sample
that is drawn. The x axis is the reference pressure, not the mean of the two methods, because the
reference is an arterial line rather than a second fallible device.

Every scored window in every draw contributes one point, which is several million per target, so a
fixed random sample is kept for drawing while the statistics use every point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s70_two_stage_headline import (CONVERTED, M_HEADLINE, MIN_FIT_WINDOWS,  # noqa: E402
                                    MIN_TEST_WINDOWS, REPORTS, RUNS, TARGETS, TRACKERS, SUFFIX,
                                    draws_for, level_scalar)

SAMPLE = 60000
SEED = 11


def main() -> None:
    Y = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    S = np.load(CONVERTED / "calfree_test" / "S.npy")
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
    subs = np.array([s for s in np.unique(S) if (S == s).sum() >= MIN_TEST_WINDOWS])
    rows_by = {int(s): np.flatnonzero(S == s) for s in subs}
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])

    track = np.mean([np.asarray(np.load(RUNS / f"{t}_test_pred.npy"), dtype=np.float64)
                     for t in TRACKERS], axis=0)
    z = np.load(RUNS / "RAW_w_feat_cache.npz")
    lvl = level_scalar(z["val"], z["test"], Sv, val_rows, Yv, fit_subs)
    draws = draws_for(subs, rows_by, M_HEADLINE)

    out, summary = {}, {}
    rng = np.random.default_rng(SEED)
    for j, t in TARGETS:
        ref, err = [], []
        for s in subs:
            rows = rows_by[int(s)]
            for warm in draws[int(s)]:
                m = np.ones(len(rows), dtype=bool)
                m[np.searchsorted(rows, np.sort(warm))] = False
                ev = rows[m]
                est = track[ev, j] + (lvl[j][warm].mean() - track[warm, j].mean())
                ref.append(Y[ev, j])
                err.append(est - Y[ev, j])
        ref, err = np.concatenate(ref), np.concatenate(err)
        me, sd = float(err.mean()), float(err.std(ddof=1))
        # the slope of error on reference, the regression-to-the-mean signature
        slope = float(np.polyfit(ref, err, 1)[0])
        summary[t] = {"n_points": int(err.size), "mean_error": me, "sd": sd,
                      "loa": [me - 1.96 * sd, me + 1.96 * sd], "error_on_reference_slope": slope}
        pick = rng.choice(err.size, size=min(SAMPLE, err.size), replace=False)
        out[f"{t}_reference"], out[f"{t}_error"] = ref[pick], err[pick]
        print("%s  n=%d  ME %+.3f  SD %.3f  LoA [%.2f, %.2f]  slope %.3f"
              % (t, err.size, me, sd, *summary[t]["loa"], slope), flush=True)

    np.savez(REPORTS / f"s71_two_stage_bland_altman_points{SUFFIX}.npz", **out)
    (REPORTS / f"s71_two_stage_bland_altman{SUFFIX}.json").write_text(json.dumps(
        {"analysis_id": "s71_two_stage_bland_altman", "warmup_windows": M_HEADLINE,
         "sampled_points_per_target": SAMPLE, "summary": summary}, indent=2), encoding="utf-8")
    print("saved reports/s71_two_stage_bland_altman.json and _points.npz")


if __name__ == "__main__":
    main()
