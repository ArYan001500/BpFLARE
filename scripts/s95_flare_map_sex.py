"""s95: FLARE (seed-42 tracker) for mean arterial pressure and by sex.

MAP: the same three stages with the MAP output channel (fluctuation index from the MAP channel, stage-1 ridge
and ECG ridge for MAP, threshold and ECG weight chosen on validation). Hypotension: windows with reference
MAP < 65 mmHg; a window is flagged when the estimate is below 65 mmHg (sensitivity, specificity; pooled over
windows and draws, descriptive).
Sex: SBP and DBP difference FLARE minus absolute network, per sex recorded in VitalDB (1 = male, 0 = female),
subject bootstrap within sex; interaction = female minus male, bootstrapped within sex. Post hoc, descriptive.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C, R, OUT93 = ROOT / "converted", ROOT / "runs" / "s4_pat_control", ROOT / "runs" / "s93"
sys.path.insert(0, str(ROOT / "scripts"))
import bp_networks as q1  # noqa: E402
sys.modules.setdefault("network_loader", types.SimpleNamespace(load_q1_module=lambda *a: q1))
from s29_level_information import pick_alpha, ridge  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

M, N_DRAWS, N_BOOT, SEED = 20, 50, 2000, 42
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
rng = np.random.default_rng(7)
Y, S = f64(np.load(C / "train" / "Y.npy")), np.load(C / "train" / "S.npy")
tr, va = np.load(C / "train" / "rows_train.npy"), np.load(C / "train" / "rows_validation.npy")
Yt, St, Gt = f64(np.load(C / "calfree_test" / "Y.npy")), np.load(C / "calfree_test" / "S.npy"), np.load(C / "calfree_test" / "G.npy")
Ya = np.concatenate([f64(np.load(C / p / "Y.npy")) for p in ("aami_cal", "aami_test")])
Sa = np.concatenate([np.load(C / p / "S.npy") for p in ("aami_cal", "aami_test")])
Ga = np.concatenate([np.load(C / p / "G.npy") for p in ("aami_cal", "aami_test")])
za, zc = np.load(R / "s78_aami_predictions.npz"), np.load(R / "RAW_w_feat_cache.npz")
cohorts = {"val": (Y[va], S[va]), "calfree": (Yt, St), "aami": (Ya, Sa)}
sexes = {"calfree": Gt, "aami": Ga}
T = {"val": f64(np.load(R / "WS_PAIRED_val_pred_s89.npy")), "calfree": f64(np.load(R / "WS_PAIRED_test_pred.npy")), "aami": f64(za["WS_PAIRED"])}
A = {"calfree": f64(np.load(R / "WS_ABS_test_pred.npy")), "aami": f64(za["WS_ABS"])}
feats = {"val": zc["val"], "calfree": zc["test"], "aami": za["RAW_w_feat"]}
subs = {k: np.array([s for s in np.unique(Sx) if (Sx == s).sum() >= (100 if k == "calfree" else 40)]) for k, (Yx, Sx) in cohorts.items()}
rows_of = {k: {int(s): np.flatnonzero(cohorts[k][1] == s) for s in subs[k]} for k in cohorts}
draws = {k: {s: [np.random.default_rng(SEED + 7 * d).choice(r, M, replace=False) for d in range(N_DRAWS)] for s, r in rows_of[k].items()} for k in cohorts}
E_win = {k: np.load(OUT93 / f"ecg_window_features_{n}.npy") for k, n in (("val", "validation"), ("calfree", "calfree"), ("aami", "aami"))}
E_tr_win = np.load(OUT93 / "ecg_window_features_train_fit.npy")
tr_subs = np.array([s for s in np.unique(S[tr]) if (S[tr] == s).sum() >= 20])
med = np.nanmedian(E_tr_win, 0)
E_tr = np.where(np.isnan(E := np.stack([np.nanmedian(E_tr_win[i * 20:(i + 1) * 20], 0) for i in range(len(tr_subs))])), med, E)


def fit_rule(G, y):
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    a = pick_alpha((G - mu) / sd, y, ALPHAS, seed=SEED)
    cmu, w, b = ridge((G - mu) / sd, y, a)
    return lambda F: ((np.asarray(F, np.float64) - mu) / sd - cmu) @ w + b


def ci(x):
    i = rng.integers(0, len(x), (N_BOOT, len(x)))
    return [round(float(x.mean()), 3), [round(float(v), 3) for v in np.percentile(x[i].mean(1), [2.5, 97.5])]]


def flare(j):
    Yv, Sv = cohorts["val"]
    vs = subs["val"]
    G = np.stack([feats["val"][Sv == s].mean(0) for s in vs]).astype(np.float64)
    yv = np.array([Yv[Sv == s, j].mean() for s in vs])
    score = {k: fit_rule(G, yv)(feats[k]) for k in ("calfree", "aami")}
    score["val"] = np.empty(len(Sv))
    folds = np.random.default_rng(SEED).permutation(len(vs)) % 4
    for k in range(4):
        m = np.isin(Sv, vs[folds == k])
        score["val"][m] = fit_rule(G[folds != k], yv[folds != k])(feats["val"][m])
    sc = StandardScaler().fit(E_tr)
    reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(E_tr), np.array([Y[tr][S[tr] == s, j].mean() for s in tr_subs]))

    def per(key):
        Yx, Sx = cohorts[key]
        out = []
        for s in subs[key]:
            rows = []
            for w in draws[key][int(s)]:
                fw = np.nanmedian(E_win[key][w], 0)
                rows.append((T[key][w, j].std(), score[key][w].mean(), reg.predict(sc.transform(np.where(np.isnan(fw), med, fw)[None]))[0],
                             np.setdiff1d(rows_of[key][int(s)], w), T[key][w, j].mean()))
            out.append(rows)
        return out
    pv = per("val")
    V = np.array([[r[0] for r in x] for x in pv])
    L = np.array([[r[1] for r in x] for x in pv])
    E = np.array([[r[2] for r in x] for x in pv])
    tru = np.array([cohorts["val"][0][rows_of["val"][int(s)], j].mean() for s in vs])
    tau, e_mean = float(np.median(V)), float(E.mean())
    b = float(min(np.linspace(0, 1.5, 31), key=lambda b: np.abs(L + b * (E - e_mean) - tru[:, None])[V <= tau].mean()))
    res = {"tau": round(tau, 3), "b": round(b, 2)}
    per_subject = {}
    for key in ("calfree", "aami"):
        Yx, Sx = cohorts[key]
        new, ab, flags = [], [], []
        for x in per(key):
            m_, a_ = [], []
            for v, l_, e_, sc_, tm in x:
                anc = l_ + (b * (e_ - e_mean) if v <= tau else 0.0)
                pred = T[key][sc_, j] - tm + anc
                m_.append(np.abs(pred - Yx[sc_, j]).mean())
                a_.append(np.abs(A[key][sc_, j] - Yx[sc_, j]).mean())
                if j == 2:
                    flags.append(np.stack([Yx[sc_, j] < 65, pred < 65, A[key][sc_, j] < 65], 1))
            new.append(np.mean(m_))
            ab.append(np.mean(a_))
        new, ab = np.array(new), np.array(ab)
        per_subject[key] = (new, ab)
        res[key] = {"FLARE": ci(new), "absolute": ci(ab), "FLARE_minus_absolute": ci(new - ab)}
        if j == 2:
            fl = np.concatenate(flags)
            hyp = fl[:, 0]
            res[key]["hypotension_MAP_below_65"] = {
                "share_of_windows": round(float(hyp.mean()), 3),
                "FLARE_sensitivity": round(float(fl[hyp, 1].mean()), 3), "FLARE_specificity": round(float(1 - fl[~hyp, 1].mean()), 3),
                "absolute_sensitivity": round(float(fl[hyp, 2].mean()), 3), "absolute_specificity": round(float(1 - fl[~hyp, 2].mean()), 3)}
    return res, per_subject


report = {"protocol": __doc__}
for j, t in ((0, "SBP"), (1, "DBP"), (2, "MAP")):
    res, ps = flare(j)
    if t != "MAP":
        for key in ("calfree", "aami"):
            Yx, Sx = cohorts[key]
            sex = np.array([sexes[key][Sx == s][0] for s in subs[key]])
            d = ps[key][0] - ps[key][1]
            blk = {}
            for name, val in (("female", 0), ("male", 1)):
                blk[name] = {"n": int((sex == val).sum()), "FLARE_minus_absolute": ci(d[sex == val])}
            fi, mi = np.flatnonzero(sex == 0), np.flatnonzero(sex == 1)
            boots = [d[rng.choice(fi, len(fi))].mean() - d[rng.choice(mi, len(mi))].mean() for _ in range(N_BOOT)]
            blk["female_minus_male"] = [round(float(d[fi].mean() - d[mi].mean()), 3), [round(float(v), 3) for v in np.percentile(boots, [2.5, 97.5])]]
            res[key]["by_sex"] = blk
    report[t] = res
    print(t, json.dumps(res), flush=True)
(ROOT / "reports" / "s95_flare_map_sex.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
