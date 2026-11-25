"""Round-3 review requests, post hoc.

1. Stage-1 ridge penalty with the standardisation fitted inside each cross-validation fold rather than
   once on all 194 validation subjects. If the selected penalty is unchanged, the final rule (scaler and
   ridge refitted on all 194) is identical and no downstream number changes.
2. The correlation between the stage-1 level and the subject's true mean under the strict protocol of
   s70 (20 warm-up windows, 50 draws), CalFree, with a subject bootstrap interval.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s29_level_information import pick_alpha, ridge  # noqa: E402
from s70_two_stage_headline import (CONVERTED, RUNS, REPORTS, N_DRAWS, N_BOOT, MIN_FIT_WINDOWS, SEED,  # noqa: E402
                                    level_scalar)

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
M = 20


def pick_alpha_foldwise(G, y, alphas, folds=4, seed=0):
    """As s29.pick_alpha (same fold assignment, MAE criterion) but the scaler is fitted per training fold."""
    fold = np.random.default_rng(seed).permutation(len(G)) % folds
    scores = {}
    for a in alphas:
        e = []
        for f in range(folds):
            tr, te = fold != f, fold == f
            mu, sd = G[tr].mean(0), G[tr].std(0)
            sd[sd < 1e-9] = 1.0
            cmu, w, b = ridge((G[tr] - mu) / sd, y[tr], a)
            e.append(np.abs(((G[te] - mu) / sd - cmu) @ w + b - y[te]).mean())
        scores[a] = float(np.mean(e))
    return min(scores, key=scores.get), scores


def cv_scores_global(Gs, y, alphas, folds=4, seed=0):
    fold = np.random.default_rng(seed).permutation(len(Gs)) % folds
    out = {}
    for a in alphas:
        e = []
        for f in range(folds):
            tr, te = fold != f, fold == f
            cmu, w, b = ridge(Gs[tr], y[tr], a)
            e.append(np.abs((Gs[te] - cmu) @ w + b - y[te]).mean())
        out[a] = float(np.mean(e))
    return out


def main():
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
    zf = np.load(RUNS / "RAW_w_feat_cache.npz")
    fv = zf["val"].astype(np.float64)
    G = np.stack([fv[Sv[val_rows] == g].mean(0) for g in fit_subs])
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    res = {"n_fit_subjects": int(len(fit_subs)), "ridge": {}, "level_correlation": {}}
    for j, t in ((0, "SBP"), (1, "DBP"), (2, "MAP")):
        y = np.array([Yv[val_rows][Sv[val_rows] == g, j].mean() for g in fit_subs])
        a_glob = pick_alpha((G - mu) / sd, y, ALPHAS, seed=SEED)
        a_fold, sc_fold = pick_alpha_foldwise(G, y, ALPHAS, seed=SEED)
        sc_glob = cv_scores_global((G - mu) / sd, y, ALPHAS, seed=SEED)
        res["ridge"][t] = {"alpha_global_scaler": a_glob, "alpha_foldwise_scaler": a_fold,
                           "cv_mae_global": sc_glob, "cv_mae_foldwise": sc_fold}
        print(t, "alpha global", a_glob, "fold-wise", a_fold,
              {f"{a:g}": (round(sc_glob[a], 3), round(sc_fold[a], 3)) for a in ALPHAS}, flush=True)

    # level correlation, strict protocol on CalFree
    Y = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    S = np.load(CONVERTED / "calfree_test" / "S.npy")
    lvl = level_scalar(zf["val"], zf["test"], Sv, val_rows, Yv, fit_subs)
    subs = np.unique(S)
    subs = np.array([s for s in subs if (S == s).sum() >= 100])
    rows_by = {int(s): np.flatnonzero(S == s) for s in subs}
    boot = np.random.default_rng(7).integers(0, len(subs), size=(N_BOOT, len(subs)))
    for j, t in ((0, "SBP"), (1, "DBP")):
        true = np.array([Y[rows_by[int(s)], j].mean() for s in subs])
        est = np.zeros((N_DRAWS, len(subs)))
        for i, s in enumerate(subs):
            r = rows_by[int(s)]
            for d in range(N_DRAWS):
                warm = np.random.default_rng(SEED + 7 * d).choice(r, size=M, replace=False)
                est[d, i] = lvl[j][warm].mean()
        r_draws = [np.corrcoef(est[d], true)[0, 1] for d in range(N_DRAWS)]
        rb = np.empty(N_BOOT)
        for b in range(N_BOOT):
            ix = boot[b]
            rb[b] = np.mean([np.corrcoef(est[d, ix], true[ix])[0, 1] for d in range(N_DRAWS)])
        res["level_correlation"][t] = {"pearson_r_mean_over_draws": float(np.mean(r_draws)),
                                       "ci": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
                                       "level_mae": float(np.abs(est - true).mean()),
                                       "population_mean_mae": float(np.abs(Yv[val_rows][:, j].mean() - true).mean())}
        print(t, {k: (np.round(v, 3).tolist() if not isinstance(v, float) else round(v, 3))
                  for k, v in res["level_correlation"][t].items()}, flush=True)
    (REPORTS / "s82_ridge_cv_and_level_r.json").write_text(json.dumps(
        {"analysis_id": "s82_ridge_cv_and_level_r", "created_utc": datetime.now(timezone.utc).isoformat(),
         "post_hoc": True, "results": res}, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
