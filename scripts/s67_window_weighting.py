"""Is there a window weighting that reads a subject's level better than a plain average?

The ridge on top of averaged features is linear, so the level it predicts from a set of windows is
the mean of a per-window scalar plus a constant. That has an awkward consequence: choosing which
windows to keep, with the answer in hand, can hit almost any target inside the range of those
scalars, so an oracle subset search reports a near-zero error and says nothing about whether the
information exists. The ceiling has to be built from something the estimator could actually use.

Two things are measured instead.

The first is how much room sampling noise leaves at all. Averaging ten windows and averaging four
hundred differ only by the noise in the mean, and a weighting rule works inside that same gap, so
the distance between them bounds what any reweighting of a fixed rule can recover.

The second is a weighting that could be deployed. Each window's own level estimate has an error,
that error is predicted from the window's features by a rule fitted on the validation subjects, and
windows expected to be wrong are down-weighted. Beside it sits the same mechanism with the true
per-window errors, which is not deployable but bounds this family of rules honestly, because the
weights stay non-negative and are never chosen to hit the target.
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

FEATURES = "RAW_w"
M_EVAL, N_DRAWS = (10, 20), 50
M_NOISE = (10, 20, 60, 200, 400)
TEMPERATURES = (0.5, 1.0, 2.0)
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
    pred = np.asarray(np.load(RUNS / "WS_PAIRED_test_pred.npy"), dtype=np.float64)

    z = np.load(RUNS / f"{FEATURES}_feat_cache.npz")
    Fv_w = z["val"].astype(np.float64)
    Ft_w = z["test"].astype(np.float64)
    Fv = np.stack([Fv_w[Sv[val_rows] == g].mean(0) for g in fit_subs])
    mu, sd = Fv.mean(0), Fv.std(0)
    sd[sd < 1e-9] = 1.0
    truth = np.stack([Yt[rows_by[int(s)], :2].mean(0) for s in subs])

    results = {}
    for j, target in ((0, "SBP"), (1, "DBP")):
        yv = np.array([Yv[val_rows][Sv[val_rows] == g, j].mean() for g in fit_subs])
        a = pick_alpha((Fv - mu) / sd, yv, ALPHAS, seed=SEED)
        cmu, w, b = ridge((Fv - mu) / sd, yv, a)

        # the level from any set of windows is the mean of this per-window scalar
        def scalar(F):
            return ((F - mu) / sd - cmu) @ w + b

        qv, qt = scalar(Fv_w), scalar(Ft_w)

        # how much room the sampling noise leaves
        print("=" * 80)
        print(f"--- {target} ---  level error against the number of averaged windows")
        noise = {}
        for M in M_NOISE:
            r = np.random.default_rng(SEED)
            est = np.stack([[qt[r.choice(rows_by[int(s)], size=min(M, len(rows_by[int(s)])),
                                         replace=False)].mean() for s in subs]
                            for _ in range(20)])
            noise[M] = float(np.abs(est - truth[None, :, j]).mean())
            print("  %4d windows (%5.1f min)   level MAE %.4f" % (M, M * 10 / 60, noise[M]))
        print("  the whole budget for any reweighting of this rule: %.4f mmHg\n"
              % (noise[M_NOISE[0]] - noise[M_NOISE[-1]]))

        # a deployable quality rule: predict each window's own error, then down-weight the bad ones
        ysub = {int(g): float(Yv[val_rows][Sv[val_rows] == g, j].mean()) for g in fit_subs}
        err_v = np.abs(qv - np.array([ysub[int(g)] for g in Sv[val_rows]]))
        Zv_w = (Fv_w - mu) / sd
        aq = pick_alpha(Zv_w[::7], err_v[::7], ALPHAS, seed=SEED)
        qcmu, qw, qb = ridge(Zv_w[::7], err_v[::7], aq)
        ehat_t = np.maximum((Ft_w - mu) / sd @ qw - qcmu @ qw + qb, 0.1)
        print("  window-error rule: correlation with the true per-window error on validation "
              "%.3f" % float(np.corrcoef(Zv_w[1::7] @ qw - qcmu @ qw + qb, err_v[1::7])[0, 1]))

        err_t = np.abs(qt - np.repeat(truth[:, j], [len(rows_by[int(s)]) for s in subs]))

        for M in M_EVAL:
            draws = {int(s): [np.random.default_rng(SEED + 7 * d).choice(
                rows_by[int(s)], size=M, replace=False) for d in range(N_DRAWS)] for s in subs}
            pos = np.zeros(len(Yt), dtype=np.int64)
            at = 0
            for s in subs:
                pos[rows_by[int(s)]] = np.arange(at, at + len(rows_by[int(s)]))
                at += len(rows_by[int(s)])

            rules = {"uniform average": None}
            for T in TEMPERATURES:
                rules[f"predicted-quality weights, T={T}"] = ("hat", T)
                rules[f"true-error weights, T={T} (not deployable)"] = ("true", T)

            print(f"  warm-up of {M} windows")
            for name, spec in rules.items():
                hat = np.zeros((N_DRAWS, len(subs)))
                for i, s in enumerate(subs):
                    for d, warm in enumerate(draws[int(s)]):
                        q = qt[warm]
                        if spec is None:
                            hat[d, i] = q.mean()
                        else:
                            e = (ehat_t if spec[0] == "hat" else err_t[pos])[warm]
                            wt = np.exp(-e / (spec[1] * e.mean()))
                            hat[d, i] = float((wt * q).sum() / wt.sum())
                lmae = float(np.abs(hat - truth[None, :, j]).mean())
                e2 = np.zeros(len(subs))
                for i, s in enumerate(subs):
                    rows = rows_by[int(s)]
                    v = []
                    for d, warm in enumerate(draws[int(s)]):
                        m_ = np.ones(len(rows), dtype=bool)
                        m_[np.searchsorted(rows, np.sort(warm))] = False
                        ev = rows[m_]
                        v.append(np.abs(pred[ev, j] + hat[d, i] - pred[warm, j].mean()
                                        - Yt[ev, j]).mean())
                    e2[i] = float(np.mean(v))
                print("    %-44s level %.4f   two-stage %.4f" % (name, lmae, e2.mean()))
                results.setdefault(target, {}).setdefault(str(M), {})[name] = {
                    "level_mae": lmae, "two_stage_mae": float(e2.mean())}
            print()
        results[target]["window_noise"] = {str(k): v for k, v in noise.items()}

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "s67_window_weighting.json").write_text(json.dumps(
        {"analysis_id": "s67_window_weighting",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "draws": N_DRAWS, "results": results}, indent=2), encoding="utf-8")
    print("saved reports/s67_window_weighting.json")


if __name__ == "__main__":
    main()
